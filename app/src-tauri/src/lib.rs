//! Tauri shell around the Blue Compressor core. The queue lives in the web UI; this side
//! scans and probes clips, runs jobs one after another on a worker thread, and reports
//! progress with events.
//!
//! Commands: app_info, scan, plan_output, preview, thumbnail, start, cancel, is_running,
//! pick_clips, pick_folder, reveal, initial_paths, confirm_quit.
//! Events: job-start {id, output}, job-progress {id, fraction, fps},
//! job-done {id, ...outcome}, job-error {id, error, cancelled}, queue-finished {},
//! open-paths [paths], quit-requested {}.

mod engine;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, WindowEvent};

use engine::{Backend, Codec, Options, Resolution};

#[derive(Default)]
struct Runner {
    running: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    initial: Mutex<Vec<String>>,
    quitting: AtomicBool,
}

#[derive(Serialize, Clone)]
struct Clip {
    path: String,
    name: String,
    ok: bool,
    error: Option<String>,
    codec: Option<String>,
    width: u32,
    height: u32,
    frames: u32,
    fps: f64,
    bytes: u64,
}

#[derive(Serialize)]
struct AppInfo {
    version: String,
    gpu: String,
    cpu: String,
    platform: String,
}

/// Queue-wide settings; quality and resolution travel with each job.
#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct Settings {
    backend: String,
    output_folder: Option<String>,
}

#[derive(Deserialize, Clone)]
struct JobRequest {
    id: u64,
    path: String,
    quality: u8,
    resolution: String,
}

#[derive(Serialize, Clone)]
struct JobStart {
    id: u64,
    output: String,
}

#[derive(Serialize, Clone)]
struct JobProgress {
    id: u64,
    fraction: f64,
    fps: f64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct JobDone {
    id: u64,
    output: String,
    input_bytes: u64,
    output_bytes: u64,
    frames: u32,
    seconds: f64,
    fps: f64,
    backend: String,
}

#[derive(Serialize, Clone)]
struct JobError {
    id: u64,
    error: String,
    cancelled: bool,
}

fn resolution(s: &str) -> Resolution {
    match s {
        "6k" => Resolution::K6,
        "4k" => Resolution::K4,
        _ => Resolution::Original,
    }
}

fn backend_of(s: &str) -> Backend {
    if s == "cpu" {
        Backend::Cpu
    } else {
        Backend::Auto
    }
}

fn is_clip(p: &Path) -> bool {
    matches!(p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(), Some("nev" | "r3d"))
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("Compressed")) {
            return;
        }
        let Ok(rd) = std::fs::read_dir(path) else { return };
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for e in entries {
            let hidden = e.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.'));
            if !hidden {
                collect(&e, out);
            }
        }
    } else if is_clip(path) {
        out.push(path.to_path_buf());
    }
}

fn describe(path: &Path) -> Clip {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let base = Clip {
        path: path.to_string_lossy().into_owned(),
        name,
        ok: false,
        error: None,
        codec: None,
        width: 0,
        height: 0,
        frames: 0,
        fps: 0.0,
        bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
    };
    match engine::probe(path) {
        Ok(i) => Clip {
            ok: true,
            codec: Some(match i.codec {
                Codec::NRaw => "N-RAW".into(),
                Codec::R3dNe => "R3D NE".into(),
            }),
            width: i.width,
            height: i.height,
            frames: i.frames,
            fps: i.fps,
            bytes: i.file_bytes,
            ..base
        },
        Err(e) => Clip { error: Some(e.to_string()), ..base },
    }
}

/// Short name of a setting, used for output folders: "Q2", "Q3 6K", "Q1 4K".
fn variant(quality: u8, resolution: &str) -> String {
    match resolution {
        "6k" => format!("Q{quality} 6K"),
        "4k" => format!("Q{quality} 4K"),
        _ => format!("Q{quality}"),
    }
}

/// Where a job writes. Clips keep their file names, so edits relink; the setting goes in
/// the folder name instead. Next to the clips that is a "Compressed Q2" (or "Compressed
/// Q3 4K") folder beside each clip. In a chosen folder the files go straight in, and a second
/// version of the same clip goes into a sub-folder named after its setting. Never an
/// existing file, and never the input itself.
fn output_for(input: &Path, folder: Option<&str>, quality: u8, resolution: &str) -> PathBuf {
    let tag = variant(quality, resolution);
    let name = input.file_name().unwrap_or_default();
    let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    };
    let free = |p: &Path| !p.exists() && !same(p, input);
    let candidates: Vec<PathBuf> = match folder {
        Some(f) if !f.is_empty() => vec![Path::new(f).join(name), Path::new(f).join(&tag).join(name)],
        _ => vec![input.parent().unwrap_or(Path::new(".")).join(format!("Compressed {tag}")).join(name)],
    };
    if let Some(p) = candidates.iter().find(|p| free(p)) {
        return p.clone();
    }
    // Same clip and setting again: number the folder, so the file name still matches.
    let last = candidates.last().unwrap();
    let dir = last.parent().unwrap_or(Path::new("."));
    let base = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let parent = dir.parent().unwrap_or(Path::new("."));
    (2..).map(|n| parent.join(format!("{base} ({n})")).join(name)).find(|p| free(p)).unwrap()
}

#[tauri::command]
fn app_info() -> AppInfo {
    AppInfo {
        version: bluecompress::VERSION.into(),
        gpu: engine::backend_description(Backend::Auto),
        cpu: engine::backend_description(Backend::Cpu),
        platform: std::env::consts::OS.into(),
    }
}

/// Expands folders and probes every clip found (run off the main thread by Tauri).
#[tauri::command(async)]
fn scan(paths: Vec<String>) -> Vec<Clip> {
    let mut files = Vec::new();
    for p in &paths {
        collect(Path::new(p), &mut files);
    }
    files.iter().map(|f| describe(f)).collect()
}

/// Coded output size for a clip at a resolution setting.
#[tauri::command]
fn plan_output(width: u32, height: u32, resolution: String) -> (u32, u32) {
    engine::output_size(width, height, self::resolution(&resolution))
}

/// Before/after preview of one frame, or of the region `crop` = [x, y, width, height] in
/// fractions of the frame, rendered `width` pixels wide. Returned as raw bytes: a 32-byte
/// little-endian header (width u32, height u32, frames u32, reserved u32, source frame bytes
/// u64, output frame bytes u64) followed by the before and after RGBA images.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn preview(
    path: String,
    frame: u32,
    quality: u8,
    resolution: String,
    backend: String,
    width: u32,
    crop: Option<[f64; 4]>,
) -> Result<tauri::ipc::Response, String> {
    let res = self::resolution(&resolution);
    let backend = backend_of(&backend);
    let crop = crop.unwrap_or([0.0, 0.0, 1.0, 1.0]);
    let p = tauri::async_runtime::spawn_blocking(move || {
        engine::preview_region(Path::new(&path), frame, quality.clamp(1, 4), res, backend, crop, width.clamp(16, 8192))
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(32 + p.before_rgba.len() + p.after_rgba.len());
    for v in [p.width, p.height, p.frames, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&p.source_frame_bytes.to_le_bytes());
    out.extend_from_slice(&p.output_frame_bytes.to_le_bytes());
    out.extend_from_slice(&p.before_rgba);
    out.extend_from_slice(&p.after_rgba);
    Ok(tauri::ipc::Response::new(out))
}

/// A small picture of the clip for the queue: width, height (u32 little endian), then RGBA.
#[tauri::command]
async fn thumbnail(path: String, width: u32) -> Result<tauri::ipc::Response, String> {
    let t = tauri::async_runtime::spawn_blocking(move || {
        let frames = engine::probe(Path::new(&path)).map(|i| i.frames).unwrap_or(1);
        // A frame a little way in: first frames are often still settling exposure.
        engine::thumbnail(Path::new(&path), frames / 4, width)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(8 + t.rgba.len());
    out.extend_from_slice(&t.width.to_le_bytes());
    out.extend_from_slice(&t.height.to_le_bytes());
    out.extend_from_slice(&t.rgba);
    Ok(tauri::ipc::Response::new(out))
}

#[tauri::command]
fn is_running(state: State<'_, Runner>) -> bool {
    state.running.load(Ordering::SeqCst)
}

#[tauri::command]
fn start(app: AppHandle, state: State<'_, Runner>, jobs: Vec<JobRequest>, settings: Settings) -> Result<(), String> {
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("already running".into());
    }
    if let Some(h) = state.worker.lock().unwrap().take() {
        let _ = h.join();
    }
    state.cancel.store(false, Ordering::SeqCst);
    let running = state.running.clone();
    let cancel = state.cancel.clone();
    let handle = std::thread::spawn(move || {
        let backend = backend_of(&settings.backend);
        for job in jobs {
            if cancel.load(Ordering::SeqCst) {
                break;
            }
            let input = PathBuf::from(&job.path);
            // Name the folder after what happens: a 6K setting on a 6K clip is no resize.
            let resized = engine::probe(&input)
                .map(|i| engine::output_size(i.width, i.height, resolution(&job.resolution)) != (i.width, i.height))
                .unwrap_or(false);
            let res_name = if resized { job.resolution.as_str() } else { "original" };
            let output = output_for(&input, settings.output_folder.as_deref(), job.quality.clamp(1, 4), res_name);
            if let Some(dir) = output.parent() {
                if let Err(e) = std::fs::create_dir_all(dir) {
                    let _ = app.emit(
                        "job-error",
                        JobError {
                            id: job.id,
                            error: format!("cannot create {}: {e}", dir.display()),
                            cancelled: false,
                        },
                    );
                    continue;
                }
            }
            let out_s = output.to_string_lossy().into_owned();
            let _ = app.emit("job-start", JobStart { id: job.id, output: out_s.clone() });
            let options = Options {
                input,
                output,
                quality: job.quality.clamp(1, 4),
                resolution: resolution(&job.resolution),
                backend,
                threads: 0,
            };
            let mut last = Instant::now() - std::time::Duration::from_secs(1);
            let app2 = app.clone();
            let mut on_progress = |fraction: f64, fps: f64| {
                if fraction >= 1.0 || last.elapsed().as_millis() >= 100 {
                    last = Instant::now();
                    let _ = app2.emit("job-progress", JobProgress { id: job.id, fraction, fps });
                }
            };
            match engine::transcode(&options, &cancel, &mut on_progress) {
                Ok(o) => {
                    let _ = app.emit(
                        "job-done",
                        JobDone {
                            id: job.id,
                            output: out_s,
                            input_bytes: o.input_bytes,
                            output_bytes: o.output_bytes,
                            frames: o.frames,
                            seconds: o.seconds,
                            fps: o.fps,
                            backend: o.backend,
                        },
                    );
                }
                Err(e) => {
                    let cancelled = cancel.load(Ordering::SeqCst);
                    let _ = app.emit("job-error", JobError { id: job.id, error: e.to_string(), cancelled });
                }
            }
        }
        running.store(false, Ordering::SeqCst);
        let _ = app.emit("queue-finished", ());
        if app.state::<Runner>().quitting.load(Ordering::SeqCst) {
            app.exit(0);
        }
    });
    *state.worker.lock().unwrap() = Some(handle);
    Ok(())
}

#[tauri::command]
fn cancel(state: State<'_, Runner>) {
    state.cancel.store(true, Ordering::SeqCst);
}

/// Quit after the running job has been cancelled and cleaned up.
#[tauri::command]
fn confirm_quit(app: AppHandle, state: State<'_, Runner>) {
    state.quitting.store(true, Ordering::SeqCst);
    state.cancel.store(true, Ordering::SeqCst);
    if !state.running.load(Ordering::SeqCst) {
        app.exit(0);
    }
}

#[tauri::command(async)]
fn pick_clips(app: AppHandle) -> Vec<String> {
    use tauri_plugin_dialog::DialogExt;
    app.dialog()
        .file()
        .set_title("Add Clips")
        .add_filter("N-RAW and R3D NE clips", &["nev", "NEV", "r3d", "R3D"])
        .blocking_pick_files()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|f| f.into_path().ok().map(|p| p.to_string_lossy().into_owned()))
        .collect()
}

#[tauri::command(async)]
fn pick_folder(app: AppHandle, title: Option<String>) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    app.dialog()
        .file()
        .set_title(title.unwrap_or_else(|| "Choose Folder".into()))
        .blocking_pick_folder()
        .and_then(|f| f.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
}

#[tauri::command]
fn reveal(path: String) -> Result<(), String> {
    tauri_plugin_opener::reveal_item_in_dir(&path).map_err(|e| e.to_string())
}

/// Paths given on the command line or opened before the window was ready.
#[tauri::command]
fn initial_paths(state: State<'_, Runner>) -> Vec<String> {
    std::mem::take(&mut *state.initial.lock().unwrap())
}

pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with('-')).collect();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(Runner { initial: Mutex::new(args), ..Default::default() })
        .invoke_handler(tauri::generate_handler![
            app_info,
            scan,
            plan_output,
            preview,
            thumbnail,
            start,
            cancel,
            is_running,
            pick_clips,
            pick_folder,
            reveal,
            initial_paths,
            confirm_quit
        ])
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let state = window.state::<Runner>();
                if state.running.load(Ordering::SeqCst) && !state.quitting.load(Ordering::SeqCst) {
                    api.prevent_close();
                    let _ = window.emit("quit-requested", ());
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to start Blue Compressor");

    app.run(|handle, event| match event {
        RunEvent::ExitRequested { api, .. } => {
            let state = handle.state::<Runner>();
            if state.running.load(Ordering::SeqCst) && !state.quitting.load(Ordering::SeqCst) {
                api.prevent_exit();
                let _ = handle.emit("quit-requested", ());
            }
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        RunEvent::Opened { urls } => {
            let paths: Vec<String> =
                urls.iter().filter_map(|u| u.to_file_path().ok()).map(|p| p.to_string_lossy().into_owned()).collect();
            if handle.get_webview_window("main").is_some() {
                let _ = handle.emit("open-paths", paths.clone());
            }
            handle.state::<Runner>().initial.lock().unwrap().extend(paths);
        }
        _ => {}
    });
}
