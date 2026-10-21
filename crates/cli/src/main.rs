//! bluecompress: headless command line front end (scripts, build servers, batch jobs).
//!
//!   bluecompress compress [options] <input|folder>... -o <output folder>
//!   bluecompress compress [options] <input> --output <file>
//!   bluecompress probe [--json] <input|folder>...
//!   bluecompress backends [--json]
//!   bluecompress version
//!   bluecompress [options] <input> <output>          (short form of compress)
//!
//! With --json, stdout carries one JSON object per line (events: start, progress, done,
//! skipped, error, summary); human readable messages go to stderr.
//! Exit codes: 0 success, 1 at least one clip failed, 2 usage error, 3 cancelled.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

use bluecompress::{Backend, ClipInfo, Codec, Options, Resolution};

static CANCEL: AtomicBool = AtomicBool::new(false);

fn q(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn ps(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

struct Cli {
    quality: u8,
    resolution: Resolution,
    backend: Backend,
    threads: usize,
    inputs: Vec<String>,
    output_dir: Option<String>,
    output_file: Option<String>,
    suffix: String,
    json: bool,
    quiet: bool,
    overwrite: bool,
    skip_existing: bool,
}

fn usage() -> String {
    format!(
        "bluecompress {}: re-encode Nikon N-RAW (.NEV) and R3D NE (.R3D) TICO RAW clips

usage:
  bluecompress compress [options] <clip|folder>... -o <output folder>
  bluecompress compress [options] <clip> --output <file>
  bluecompress probe [--json] <clip|folder>...
  bluecompress backends [--json]
  bluecompress version
  bluecompress [options] <clip> <output file>

options:
  -q, --quality 1..4      1 lossless repack (pixel-identical), 2 archival (default),
                          3 high, 4 compact
  -r, --resolution R      original (default), 6k or 4k
  -b, --backend B         auto (default), cpu, metal or cuda; --cpu / --gpu also work
  -j, --threads N         worker threads (default: all cores)
  -o, --output-dir DIR    write outputs here with the source file names
      --output FILE       single input: exact output path
      --suffix S          append S to output file names (before the extension)
      --overwrite         replace existing outputs (default: fail)
      --skip-existing     skip clips whose output already exists
      --json              JSON lines on stdout (start, progress, done, skipped, error, summary)
      --quiet             no progress on stderr

Folders are searched recursively for .nev and .r3d files. Originals are never modified;
outputs are written to <name>.partial and renamed when complete.
exit codes: 0 success, 1 a clip failed, 2 usage error, 3 cancelled
",
        bluecompress::VERSION
    )
}

fn is_clip(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("nev") || e.eq_ignore_ascii_case("r3d"))
        .unwrap_or(false)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        match e.file_type() {
            Ok(t) if t.is_dir() => walk(&p, out),
            Ok(t) if t.is_file() && is_clip(&p) => out.push(p),
            // Follow symlinks to files.
            Ok(t) if t.is_symlink() && p.is_file() && is_clip(&p) => out.push(p),
            _ => {}
        }
    }
}

fn expand(inputs: &[String], missing: &mut Vec<String>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for s in inputs {
        let p = PathBuf::from(s);
        if p.is_dir() {
            let mut found = Vec::new();
            walk(&p, &mut found);
            found.sort();
            out.extend(found);
        } else if p.exists() {
            out.push(p);
        } else {
            missing.push(s.clone());
        }
    }
    out
}

fn codec_name(i: &ClipInfo) -> &'static str {
    if i.codec == Codec::R3dNe {
        "R3D NE"
    } else {
        "N-RAW"
    }
}

fn fourcc(i: &ClipInfo) -> &'static str {
    if i.codec == Codec::R3dNe {
        "NR3D"
    } else {
        "NRAW"
    }
}

fn probe_cmd(o: &Cli) -> u8 {
    let mut missing = Vec::new();
    let mut rc = 0;
    for p in expand(&o.inputs, &mut missing) {
        let info = match bluecompress::probe(&p) {
            Ok(i) => i,
            Err(e) => {
                rc = 1;
                if o.json {
                    println!("{{\"event\":\"error\",\"input\":{},\"error\":{}}}", q(&ps(&p)), q(&e.0));
                } else {
                    eprintln!("{}: {}", p.display(), e);
                }
                continue;
            }
        };
        let (w6, h6) = bluecompress::output_size(info.width, info.height, Resolution::K6);
        let (w4, h4) = bluecompress::output_size(info.width, info.height, Resolution::K4);
        if o.json {
            println!(
                "{{\"event\":\"clip\",\"input\":{},\"codec\":{},\"format\":{},\"width\":{},\"height\":{},\
                 \"frames\":{},\"fps\":{},\"file_bytes\":{},\"video_bytes\":{},\"audio_tracks\":{},\
                 \"output_6k\":[{w6},{h6}],\"output_4k\":[{w4},{h4}]}}",
                q(&ps(&p)),
                q(fourcc(&info)),
                q(codec_name(&info)),
                info.width,
                info.height,
                info.frames,
                info.fps,
                info.file_bytes,
                info.video_bytes,
                info.audio_tracks
            );
        } else {
            println!(
                "{}\n  {} {}x{}, {} frames @ {:.3} fps, {:.2} GB, {} audio track(s)\n  6K -> {w6}x{h6}, 4K -> {w4}x{h4}",
                p.display(),
                codec_name(&info),
                info.width,
                info.height,
                info.frames,
                info.fps,
                info.file_bytes as f64 / 1e9,
                info.audio_tracks
            );
        }
    }
    for m in missing {
        rc = 1;
        if o.json {
            println!("{{\"event\":\"error\",\"input\":{},\"error\":\"not found\"}}", q(&m));
        } else {
            eprintln!("{m}: not found");
        }
    }
    rc
}

fn backends_cmd(o: &Cli) -> u8 {
    let all = [("auto", Backend::Auto), ("cpu", Backend::Cpu), ("metal", Backend::Metal), ("cuda", Backend::Cuda)];
    if o.json {
        let items: Vec<String> =
            all.iter().map(|(n, b)| format!("{}:{}", q(n), q(&bluecompress::backend_description(*b)))).collect();
        println!(
            "{{\"event\":\"backends\",\"version\":{},\"backends\":{{{}}}}}",
            q(bluecompress::VERSION),
            items.join(",")
        );
    } else {
        for (n, b) in all {
            println!("{n:<6} {}", bluecompress::backend_description(b));
        }
    }
    0
}

fn compress_cmd(o: &Cli) -> u8 {
    let mut missing = Vec::new();
    let inputs = expand(&o.inputs, &mut missing);
    if inputs.is_empty() && missing.is_empty() {
        eprintln!("error: no .nev or .r3d clips found");
        return 2;
    }
    if o.output_file.is_some() && inputs.len() + missing.len() != 1 {
        eprintln!("error: --output takes exactly one input; use -o <folder> for several");
        return 2;
    }
    if o.output_file.is_none() && o.output_dir.is_none() {
        eprintln!("error: give an output folder with -o (or --output for a single file)");
        return 2;
    }
    let (mut failed, mut done, mut skipped) = (missing.len(), 0usize, 0usize);
    let (mut in_bytes, mut out_bytes, mut seconds) = (0u64, 0u64, 0f64);
    for m in &missing {
        if o.json {
            println!("{{\"event\":\"error\",\"input\":{},\"error\":\"not found\"}}", q(m));
        } else {
            eprintln!("{m}: not found");
        }
    }
    let mut stdout = std::io::stdout();
    for input in &inputs {
        if CANCEL.load(Ordering::Relaxed) {
            break;
        }
        let out = match (&o.output_file, &o.output_dir) {
            (Some(f), _) => PathBuf::from(f),
            (None, Some(d)) => {
                let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let ext = input.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
                Path::new(d).join(format!("{stem}{}{ext}", o.suffix))
            }
            (None, None) => unreachable!(),
        };
        let (ins, outs) = (ps(input), ps(&out));
        if out.exists() && !o.overwrite {
            if o.skip_existing {
                skipped += 1;
                if o.json {
                    println!(
                        "{{\"event\":\"skipped\",\"input\":{},\"output\":{},\"reason\":\"output exists\"}}",
                        q(&ins),
                        q(&outs)
                    );
                } else if !o.quiet {
                    eprintln!("{ins}: skipped, {outs} exists");
                }
                continue;
            }
            failed += 1;
            if o.json {
                println!(
                    "{{\"event\":\"error\",\"input\":{},\"output\":{},\"error\":\"output exists (use --overwrite or --skip-existing)\"}}",
                    q(&ins),
                    q(&outs)
                );
            } else {
                eprintln!("{ins}: {outs} exists (use --overwrite or --skip-existing)");
            }
            continue;
        }
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(parent);
        }

        if let Ok(info) = bluecompress::probe(input) {
            let (ow, oh) = bluecompress::output_size(info.width, info.height, o.resolution);
            if o.json {
                println!(
                    "{{\"event\":\"start\",\"input\":{},\"output\":{},\"codec\":{},\"width\":{},\"height\":{},\
                     \"frames\":{},\"out_width\":{ow},\"out_height\":{oh},\"quality\":{}}}",
                    q(&ins),
                    q(&outs),
                    q(fourcc(&info)),
                    info.width,
                    info.height,
                    info.frames,
                    o.quality
                );
            } else if !o.quiet {
                eprintln!(
                    "{ins}: {} {}x{}, {} frames @ {:.3} fps -> {ow}x{oh} Q{}",
                    codec_name(&info),
                    info.width,
                    info.height,
                    info.frames,
                    info.fps,
                    o.quality
                );
            }
            let _ = stdout.flush();
        }

        let opt = Options {
            input: input.clone(),
            output: out.clone(),
            quality: o.quality,
            resolution: o.resolution,
            backend: o.backend,
            threads: o.threads,
        };
        let mut last_permille = -1i64;
        let result = bluecompress::transcode(&opt, &CANCEL, &mut |fraction, fps| {
            let pm = (fraction * 1000.0) as i64;
            if pm == last_permille {
                return;
            }
            last_permille = pm;
            if o.json {
                if pm % 10 == 0 {
                    println!(
                        "{{\"event\":\"progress\",\"input\":{},\"fraction\":{fraction:.3},\"fps\":{fps:.2}}}",
                        q(&ins)
                    );
                    let _ = std::io::stdout().flush();
                }
            } else if !o.quiet {
                eprint!("\r  {:5.1}%  {fps:6.1} fps", fraction * 100.0);
            }
        });
        if !o.json && !o.quiet {
            eprintln!();
        }
        let r = match result {
            Ok(r) => r,
            Err(e) => {
                if CANCEL.load(Ordering::Relaxed) {
                    break;
                }
                failed += 1;
                if o.json {
                    println!("{{\"event\":\"error\",\"input\":{},\"error\":{}}}", q(&ins), q(&e.0));
                } else {
                    eprintln!("{ins}: error: {e}");
                }
                continue;
            }
        };
        done += 1;
        in_bytes += r.input_bytes;
        out_bytes += r.output_bytes;
        seconds += r.seconds;
        let saved = 100.0 * (1.0 - r.output_bytes as f64 / r.input_bytes.max(1) as f64);
        if o.json {
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
        } else if !o.quiet {
            eprintln!(
                "done: {:.2} GB -> {:.2} GB ({saved:.1}% smaller) in {:.1} s, {:.1} fps [{}]",
                r.input_bytes as f64 / 1e9,
                r.output_bytes as f64 / 1e9,
                r.seconds,
                r.fps,
                r.backend
            );
        }
        let _ = stdout.flush();
    }
    let cancelled = CANCEL.load(Ordering::Relaxed);
    if o.json {
        println!(
            "{{\"event\":\"summary\",\"done\":{done},\"failed\":{failed},\"skipped\":{skipped},\"cancelled\":{cancelled},\
             \"input_bytes\":{in_bytes},\"output_bytes\":{out_bytes},\"seconds\":{seconds:.3}}}"
        );
    } else if inputs.len() > 1 && !o.quiet {
        eprintln!(
            "{done} done, {failed} failed, {skipped} skipped: {:.2} GB -> {:.2} GB",
            in_bytes as f64 / 1e9,
            out_bytes as f64 / 1e9
        );
    }
    if cancelled {
        eprintln!("cancelled");
        return 3;
    }
    if failed > 0 {
        1
    } else {
        0
    }
}

fn bad(what: &str) -> ExitCode {
    eprintln!("error: {what} (see bluecompress --help)");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let _ = ctrlc::set_handler(|| CANCEL.store(true, Ordering::Relaxed));
    let mut o = Cli {
        quality: 2,
        resolution: Resolution::Original,
        backend: Backend::Auto,
        threads: 0,
        inputs: Vec::new(),
        output_dir: None,
        output_file: None,
        suffix: String::new(),
        json: false,
        quiet: false,
        overwrite: false,
        skip_existing: false,
    };
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let mut it = args.into_iter().peekable();
    let cmd = match it.peek().map(String::as_str) {
        Some(c @ ("compress" | "probe" | "backends" | "version")) => {
            let c = c.to_string();
            it.next();
            c
        }
        _ => String::new(),
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{}", usage());
                return ExitCode::SUCCESS;
            }
            "-q" | "--quality" => match it.next().and_then(|v| v.parse::<u8>().ok()).filter(|v| (1..=4).contains(v)) {
                Some(v) => o.quality = v,
                None => return bad("quality must be 1 to 4"),
            },
            "-r" | "--resolution" => {
                o.resolution = match it.next().unwrap_or_default().to_ascii_lowercase().as_str() {
                    "6k" => Resolution::K6,
                    "4k" => Resolution::K4,
                    "original" | "orig" => Resolution::Original,
                    _ => return bad("resolution must be original, 6k or 4k"),
                }
            }
            "-b" | "--backend" => {
                o.backend = match it.next().unwrap_or_default().as_str() {
                    "auto" | "gpu" => Backend::Auto,
                    "cpu" => Backend::Cpu,
                    "metal" => Backend::Metal,
                    "cuda" => Backend::Cuda,
                    _ => return bad("backend must be auto, cpu, metal or cuda"),
                }
            }
            "--cpu" => o.backend = Backend::Cpu,
            "--gpu" => o.backend = Backend::Auto,
            "-j" | "--threads" => match it.next().and_then(|v| v.parse::<usize>().ok()).filter(|v| *v <= 1024) {
                Some(v) => o.threads = v,
                None => return bad("threads must be a number"),
            },
            "-o" | "--output-dir" => match it.next() {
                Some(v) => o.output_dir = Some(v),
                None => return bad("-o needs a folder"),
            },
            "--output" => match it.next() {
                Some(v) => o.output_file = Some(v),
                None => return bad("--output needs a file"),
            },
            "--suffix" => match it.next() {
                Some(v) => o.suffix = v,
                None => return bad("--suffix needs a value"),
            },
            "--overwrite" => o.overwrite = true,
            "--skip-existing" => o.skip_existing = true,
            "--json" => o.json = true,
            "--quiet" => o.quiet = true,
            s if s.len() > 1 && s.starts_with('-') => return bad(&format!("unknown option {s}")),
            _ => o.inputs.push(a),
        }
    }

    let rc = match cmd.as_str() {
        "version" => {
            if o.json {
                println!("{{\"event\":\"version\",\"version\":{}}}", q(bluecompress::VERSION));
            } else {
                println!("{}", bluecompress::VERSION);
            }
            0
        }
        "backends" => backends_cmd(&o),
        _ if o.inputs.is_empty() => {
            eprint!("{}", usage());
            2
        }
        "probe" => probe_cmd(&o),
        _ => {
            // Short form: bluecompress [options] <input> <output file>
            if cmd.is_empty()
                && o.inputs.len() == 2
                && o.output_dir.is_none()
                && o.output_file.is_none()
                && !Path::new(&o.inputs[1]).is_dir()
            {
                o.output_file = o.inputs.pop();
            }
            compress_cmd(&o)
        }
    };
    ExitCode::from(rc)
}
