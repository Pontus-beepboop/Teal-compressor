//! Watch-folder daemon: monitors an input directory recursively for stable .NEV and .R3D clips
//! and writes compressed copies into an output directory while preserving relative paths.

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use bluecompress::{Backend, Options, Resolution};
use serde::Serialize;

use crate::{codec_name, fourcc, is_clip, ps, q, resolution_name};

pub struct WatchConfig {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub quality: u8,
    pub resolution: Resolution,
    pub backend: Backend,
    pub threads: usize,
    pub poll_seconds: u64,
    pub json: bool,
    pub quiet: bool,
    pub once: bool,
}

#[derive(Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct WatchSnapshot {
    pub enabled: bool,
    pub input_dir: String,
    pub output_dir: String,
    pub quality: u8,
    pub resolution: String,
    pub poll_seconds: u64,
    pub state: String,
    pub current_clip: Option<String>,
    pub fraction: f64,
    pub fps: f64,
    pub done_count: usize,
    pub failed_count: usize,
    pub saved_input_bytes: u64,
    pub saved_output_bytes: u64,
    pub last_clip: Option<String>,
    pub last_error: Option<String>,
}

pub type WatchHook = Arc<dyn Fn(&WatchSnapshot) + Send + Sync>;

type FileStamp = (u64, Option<SystemTime>);

pub fn is_ignored_name(name: &str) -> bool {
    name.starts_with('.')
        || name.eq_ignore_ascii_case("@eadir")
        || name.eq_ignore_ascii_case("#recycle")
        || name.eq_ignore_ascii_case("#snapshot")
        || name.ends_with(".partial")
        || name.ends_with(".tmp")
}

fn same_dir(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

fn walk_watch(dir: &Path, skip_dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
        if is_ignored_name(name) {
            continue;
        }
        if p.is_dir() {
            if same_dir(&p, skip_dir) {
                continue;
            }
            walk_watch(&p, skip_dir, out);
        } else if p.is_file() && is_clip(&p) {
            out.push(p);
        }
    }
}

fn temp_output_path(target: &Path) -> PathBuf {
    let mut s = target.as_os_str().to_owned();
    s.push(".bluecompress.tmp");
    PathBuf::from(s)
}

fn file_stamp(path: &Path) -> Option<FileStamp> {
    let m = std::fs::metadata(path).ok()?;
    if !m.is_file() || m.len() == 0 {
        return None;
    }
    Some((m.len(), m.modified().ok()))
}

fn update_state(state: &Mutex<WatchSnapshot>, hook: Option<&WatchHook>, f: impl FnOnce(&mut WatchSnapshot)) {
    let snap = {
        let mut s = state.lock().unwrap();
        f(&mut s);
        s.clone()
    };
    if let Some(h) = hook {
        h(&snap);
    }
}

pub fn run_watch(
    cfg: WatchConfig,
    cancel: &AtomicBool,
    state: Arc<Mutex<WatchSnapshot>>,
    hook: Option<WatchHook>,
    transcode_lock: Option<Arc<Mutex<()>>>,
) -> u8 {
    let _ = std::fs::create_dir_all(&cfg.input_dir);
    let _ = std::fs::create_dir_all(&cfg.output_dir);
    if same_dir(&cfg.input_dir, &cfg.output_dir) {
        eprintln!("error: watch input and output folders must be different");
        return 2;
    }

    update_state(&state, hook.as_ref(), |s| {
        s.enabled = true;
        s.input_dir = ps(&cfg.input_dir);
        s.output_dir = ps(&cfg.output_dir);
        s.quality = cfg.quality;
        s.resolution = resolution_name(cfg.resolution).into();
        s.poll_seconds = cfg.poll_seconds;
        s.state = "idle".into();
    });

    if !cfg.quiet && !cfg.json {
        println!(
            "[watch] watching {} -> {} (Q{}, {}, poll {}s)",
            cfg.input_dir.display(),
            cfg.output_dir.display(),
            cfg.quality,
            resolution_name(cfg.resolution),
            cfg.poll_seconds
        );
        let _ = std::io::stdout().flush();
    }

    let mut observed: HashMap<PathBuf, FileStamp> = HashMap::new();
    let mut processed: HashSet<(PathBuf, u64, Option<SystemTime>)> = HashSet::new();
    let mut failed_total = 0usize;

    loop {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        let mut clips = Vec::new();
        walk_watch(&cfg.input_dir, &cfg.output_dir, &mut clips);

        let current_set: HashSet<PathBuf> = clips.iter().cloned().collect();
        observed.retain(|k, _| current_set.contains(k));

        let mut ready = Vec::new();
        let mut stabilizing = 0usize;

        for clip in clips {
            let rel = match clip.strip_prefix(&cfg.input_dir) {
                Ok(r) => r.to_path_buf(),
                Err(_) => match clip.file_name() {
                    Some(n) => PathBuf::from(n),
                    None => continue,
                },
            };
            let target_out = cfg.output_dir.join(&rel);
            if std::fs::metadata(&target_out).map(|m| m.len() > 0).unwrap_or(false) {
                observed.remove(&clip);
                continue;
            }
            let Some(stamp) = file_stamp(&clip) else { continue };
            let key = (clip.clone(), stamp.0, stamp.1);
            if processed.contains(&key) {
                continue;
            }
            match observed.get(&clip) {
                Some(&prev) if prev == stamp => {
                    ready.push((clip, rel, target_out, stamp));
                }
                _ => {
                    observed.insert(clip, stamp);
                    stabilizing += 1;
                }
            }
        }

        if ready.is_empty() {
            let next_state = if stabilizing > 0 { "stabilizing" } else { "idle" };
            update_state(&state, hook.as_ref(), |s| {
                if s.state != next_state || s.current_clip.is_some() {
                    s.state = next_state.into();
                    s.current_clip = None;
                    s.fraction = 0.0;
                    s.fps = 0.0;
                }
            });
            if cfg.once && stabilizing == 0 {
                break;
            }
        } else {
            for (clip, rel, target_out, expected_stamp) in ready {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                // Re-check right before transcoding in case the file changed while an earlier clip ran.
                let Some(now_stamp) = file_stamp(&clip) else {
                    observed.remove(&clip);
                    continue;
                };
                if now_stamp != expected_stamp {
                    observed.insert(clip, now_stamp);
                    stabilizing += 1;
                    continue;
                }
                if std::fs::metadata(&target_out).map(|m| m.len() > 0).unwrap_or(false) {
                    observed.remove(&clip);
                    processed.insert((clip, expected_stamp.0, expected_stamp.1));
                    continue;
                }

                let rel_s = ps(&rel);
                let ins = ps(&clip);
                let outs = ps(&target_out);

                if let Some(parent) = target_out.parent().filter(|p| !p.as_os_str().is_empty()) {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        failed_total += 1;
                        processed.insert((clip.clone(), expected_stamp.0, expected_stamp.1));
                        observed.remove(&clip);
                        let msg = format!("cannot create {}: {e}", parent.display());
                        update_state(&state, hook.as_ref(), |s| {
                            s.failed_count += 1;
                            s.last_error = Some(format!("{rel_s}: {msg}"));
                            s.state = "idle".into();
                            s.current_clip = None;
                        });
                        if cfg.json {
                            println!("{{\"event\":\"error\",\"input\":{},\"error\":{}}}", q(&ins), q(&msg));
                        } else {
                            eprintln!("[watch] {rel_s}: error: {msg}");
                        }
                        continue;
                    }
                }

                let tmp_out = temp_output_path(&target_out);
                let _ = std::fs::remove_file(&tmp_out);

                let _guard = transcode_lock.as_ref().map(|l| l.lock().unwrap());
                if cancel.load(Ordering::Relaxed) {
                    break;
                }

                update_state(&state, hook.as_ref(), |s| {
                    s.state = "running".into();
                    s.current_clip = Some(rel_s.clone());
                    s.fraction = 0.0;
                    s.fps = 0.0;
                });

                if let Ok(info) = bluecompress::probe(&clip) {
                    let (ow, oh) = bluecompress::output_size(info.width, info.height, cfg.resolution);
                    if cfg.json {
                        println!(
                            "{{\"event\":\"start\",\"input\":{},\"output\":{},\"codec\":{},\"width\":{},\"height\":{},\
                             \"frames\":{},\"out_width\":{ow},\"out_height\":{oh},\"quality\":{}}}",
                            q(&ins),
                            q(&outs),
                            q(fourcc(&info)),
                            info.width,
                            info.height,
                            info.frames,
                            cfg.quality
                        );
                    } else if !cfg.quiet {
                        println!(
                            "[watch] compressing {rel_s}: {} {}x{}, {} frames @ {:.3} fps -> {ow}x{oh} Q{}",
                            codec_name(&info),
                            info.width,
                            info.height,
                            info.frames,
                            info.fps,
                            cfg.quality
                        );
                    }
                    let _ = std::io::stdout().flush();
                }

                let opt = Options {
                    input: clip.clone(),
                    output: tmp_out.clone(),
                    quality: cfg.quality,
                    resolution: cfg.resolution,
                    backend: cfg.backend,
                    threads: cfg.threads,
                };

                let mut last_emit = Instant::now() - Duration::from_secs(1);
                let result = bluecompress::transcode(&opt, cancel, &mut |fraction, fps| {
                    if fraction >= 1.0 || last_emit.elapsed() >= Duration::from_millis(150) {
                        last_emit = Instant::now();
                        update_state(&state, hook.as_ref(), |s| {
                            s.fraction = fraction;
                            s.fps = fps;
                        });
                    }
                });

                observed.remove(&clip);

                match result {
                    Ok(r) => {
                        if let Err(e) = std::fs::rename(&tmp_out, &target_out) {
                            let _ = std::fs::remove_file(&tmp_out);
                            failed_total += 1;
                            processed.insert((clip, expected_stamp.0, expected_stamp.1));
                            let msg = format!("rename failed {}: {e}", target_out.display());
                            update_state(&state, hook.as_ref(), |s| {
                                s.failed_count += 1;
                                s.last_error = Some(format!("{rel_s}: {msg}"));
                                s.state = "idle".into();
                                s.current_clip = None;
                                s.fraction = 0.0;
                                s.fps = 0.0;
                            });
                            if cfg.json {
                                println!("{{\"event\":\"error\",\"input\":{},\"error\":{}}}", q(&ins), q(&msg));
                            } else {
                                eprintln!("[watch] {rel_s}: error: {msg}");
                            }
                            continue;
                        }
                        processed.insert((clip, expected_stamp.0, expected_stamp.1));
                        let saved = 100.0 * (1.0 - r.output_bytes as f64 / r.input_bytes.max(1) as f64);
                        update_state(&state, hook.as_ref(), |s| {
                            s.done_count += 1;
                            s.saved_input_bytes += r.input_bytes;
                            s.saved_output_bytes += r.output_bytes;
                            s.last_clip = Some(rel_s.clone());
                            s.last_error = None;
                            s.state = "idle".into();
                            s.current_clip = None;
                            s.fraction = 0.0;
                            s.fps = 0.0;
                        });
                        if cfg.json {
                            println!(
                                "{{\"event\":\"done\",\"input\":{},\"output\":{},\"input_bytes\":{},\"output_bytes\":{},\
                                 \"saved_percent\":{saved:.2},\"frames\":{},\"seconds\":{:.3},\"fps\":{:.2},\"backend\":{}}}",
                                q(&ins),
                                q(&outs),
                                r.input_bytes,
                                r.output_bytes,
                                r.frames,
                                r.seconds,
                                r.fps,
                                q(&r.backend)
                            );
                        } else if !cfg.quiet {
                            println!(
                                "[watch] done {rel_s}: {:.2} GB -> {:.2} GB ({saved:.1}% smaller) in {:.1} s, {:.1} fps [{}]",
                                r.input_bytes as f64 / 1e9,
                                r.output_bytes as f64 / 1e9,
                                r.seconds,
                                r.fps,
                                r.backend
                            );
                        }
                        let _ = std::io::stdout().flush();
                    }
                    Err(e) => {
                        let _ = std::fs::remove_file(&tmp_out);
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        failed_total += 1;
                        processed.insert((clip, expected_stamp.0, expected_stamp.1));
                        update_state(&state, hook.as_ref(), |s| {
                            s.failed_count += 1;
                            s.last_error = Some(format!("{rel_s}: {}", e.0));
                            s.state = "idle".into();
                            s.current_clip = None;
                            s.fraction = 0.0;
                            s.fps = 0.0;
                        });
                        if cfg.json {
                            println!("{{\"event\":\"error\",\"input\":{},\"error\":{}}}", q(&ins), q(&e.0));
                        } else {
                            eprintln!("[watch] {rel_s}: error: {e}");
                        }
                    }
                }
            }
            if cfg.once && stabilizing == 0 {
                break;
            }
        }

        let wait_ms = cfg.poll_seconds.max(1) * 1000;
        let mut slept = 0u64;
        while slept < wait_ms {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            let step = (wait_ms - slept).min(150);
            std::thread::sleep(Duration::from_millis(step));
            slept += step;
        }
    }

    if cancel.load(Ordering::Relaxed) {
        return 3;
    }
    if failed_total > 0 {
        1
    } else {
        0
    }
}
