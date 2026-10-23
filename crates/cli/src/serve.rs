//! Embedded Web UI server (`bluecompress serve`): serves `app/ui` over HTTP/1.1, exposes
//! the Tauri command surface at `/api/invoke/<cmd>`, streams queue and watch-folder events
//! over Server-Sent Events (`/api/events`), and provides server-side directory browsing
//! (`/api/fs/list`, `/api/fs/mkdir`) for NAS and Docker deployments.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bluecompress::{Backend, Codec, Options, Resolution};
use serde::{Deserialize, Serialize};

use crate::watch::{
    is_ignored_name, load_persisted_settings, run_watch, save_persisted_settings, WatchConfig, WatchSnapshot,
};
use crate::{is_clip, parse_resolution, ps, resolution_name};

const INDEX_HTML: &str = include_str!("../../../app/ui/index.html");
const STYLES_CSS: &str = include_str!("../../../app/ui/styles.css");
const APP_JS: &str = include_str!("../../../app/ui/app.js");
const ICON_PNG: &[u8] = include_bytes!("../../../app/ui/icon.png");

pub struct ServeConfig {
    pub listen: String,
    pub browse_root: PathBuf,
    pub initial_paths: Vec<String>,
    pub default_quality: u8,
    pub default_resolution: Resolution,
    pub default_backend: Backend,
    pub threads: usize,
    pub watch: Option<WatchConfig>,
}

struct ServerState {
    browse_root: PathBuf,
    default_quality: u8,
    default_resolution: Resolution,
    default_backend: Backend,
    threads: usize,
    running: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
    initial: Mutex<Vec<String>>,
    sse_clients: Arc<Mutex<Vec<TcpStream>>>,
    watch_state: Arc<Mutex<WatchSnapshot>>,
    transcode_lock: Arc<Mutex<()>>,
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
#[serde(rename_all = "camelCase")]
struct AppInfo {
    version: String,
    gpu: String,
    cpu: String,
    platform: String,
    web: bool,
    browse_root: String,
    default_quality: u8,
    default_resolution: String,
    watch: WatchSnapshot,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SetWatchSettingsArgs {
    clear_after_process: Option<bool>,
    quality: Option<u8>,
    resolution: Option<String>,
}

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

#[derive(Deserialize)]
struct StartArgs {
    jobs: Vec<JobRequest>,
    settings: Settings,
}

#[derive(Deserialize)]
struct ScanArgs {
    paths: Vec<String>,
}

#[derive(Deserialize)]
struct PlanOutputArgs {
    width: u32,
    height: u32,
    resolution: String,
}

#[derive(Deserialize)]
struct PreviewArgs {
    path: String,
    frame: u32,
    quality: u8,
    resolution: String,
    backend: String,
    width: u32,
    crop: Option<[f64; 4]>,
}

#[derive(Deserialize)]
struct ThumbnailArgs {
    path: String,
    width: u32,
}

#[derive(Deserialize)]
struct FsListArgs {
    path: Option<String>,
}

#[derive(Deserialize)]
struct FsMkdirArgs {
    path: Option<String>,
    parent: Option<String>,
    name: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FsEntry {
    name: String,
    path: String,
    kind: &'static str,
    codec: Option<&'static str>,
    size: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FsListResponse {
    path: String,
    parent: Option<String>,
    browse_root: String,
    roots: Vec<String>,
    entries: Vec<FsEntry>,
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

fn resolution_of(s: &str) -> Resolution {
    match s {
        "6k" => Resolution::K6,
        "4k" => Resolution::K4,
        _ => Resolution::Original,
    }
}

fn backend_of(s: &str, default_backend: Backend) -> Backend {
    match s {
        "cpu" => Backend::Cpu,
        "metal" => Backend::Metal,
        "cuda" => Backend::Cuda,
        "opencl" => Backend::OpenCl,
        _ => default_backend,
    }
}

fn collect_clips(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("Compressed") || is_ignored_name(n))
        {
            return;
        }
        let Ok(rd) = std::fs::read_dir(path) else { return };
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        for e in entries {
            let skip = e.file_name().and_then(|n| n.to_str()).is_some_and(is_ignored_name);
            if !skip {
                collect_clips(&e, out);
            }
        }
    } else if is_clip(path) {
        out.push(path.to_path_buf());
    }
}

fn describe_clip(path: &Path) -> Clip {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let base = Clip {
        path: ps(path),
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
    match bluecompress::probe(path) {
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

fn variant(quality: u8, resolution: &str) -> String {
    match resolution {
        "6k" => format!("Q{quality} 6K"),
        "4k" => format!("Q{quality} 4K"),
        _ => format!("Q{quality}"),
    }
}

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
    let last = candidates.last().unwrap();
    let dir = last.parent().unwrap_or(Path::new("."));
    let base = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let parent = dir.parent().unwrap_or(Path::new("."));
    (2..).map(|n| parent.join(format!("{base} ({n})")).join(name)).find(|p| free(p)).unwrap()
}

fn broadcast_event<T: Serialize>(clients: &Mutex<Vec<TcpStream>>, event: &str, payload: T) {
    let line = match serde_json::to_string(&serde_json::json!({ "event": event, "payload": payload })) {
        Ok(s) => format!("data: {s}\n\n"),
        Err(_) => return,
    };
    let mut guard = clients.lock().unwrap();
    guard.retain_mut(|stream| stream.write_all(line.as_bytes()).and_then(|_| stream.flush()).is_ok());
}

fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::RootDir => out.push(c.as_os_str()),
            Component::Normal(s) => out.push(s),
            Component::ParentDir => {
                out.pop();
                if out.as_os_str().is_empty() {
                    out.push("/");
                }
            }
            Component::CurDir | Component::Prefix(_) => {}
        }
    }
    if out.as_os_str().is_empty() {
        PathBuf::from("/")
    } else {
        out
    }
}

fn within_root(path: &Path, root: &Path) -> bool {
    let cp = path.canonicalize().unwrap_or_else(|_| normalize_path(path));
    let cr = root.canonicalize().unwrap_or_else(|_| normalize_path(root));
    cp.starts_with(&cr)
}

fn default_start_dir(state: &ServerState) -> PathBuf {
    let root = state.browse_root.canonicalize().unwrap_or_else(|_| normalize_path(&state.browse_root));
    if root == Path::new("/") {
        for candidate in ["/media", "/data", "/volume1", "/watch/in"] {
            let p = Path::new(candidate);
            if p.is_dir() {
                return p.to_path_buf();
            }
        }
    }
    root
}

fn discover_roots(state: &ServerState) -> Vec<String> {
    let root = state.browse_root.canonicalize().unwrap_or_else(|_| normalize_path(&state.browse_root));
    let mut roots = Vec::new();
    let mut push = |p: &Path| {
        if p.is_dir() && within_root(p, &root) {
            let s = ps(p);
            if !roots.contains(&s) {
                roots.push(s);
            }
        }
    };
    push(&root);
    for c in ["/media", "/data", "/mnt", "/volume1", "/volume2", "/watch/in", "/watch/out"] {
        push(Path::new(c));
    }
    let ws = state.watch_state.lock().unwrap();
    if ws.enabled {
        push(Path::new(&ws.input_dir));
        push(Path::new(&ws.output_dir));
    }
    roots
}

fn handle_fs_list(state: &ServerState, requested: Option<&str>) -> Result<FsListResponse, String> {
    let root = state.browse_root.canonicalize().unwrap_or_else(|_| normalize_path(&state.browse_root));
    let target = match requested.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => PathBuf::from(s),
        None => default_start_dir(state),
    };
    let canon = target.canonicalize().map_err(|e| format!("cannot open {}: {e}", target.display()))?;
    if !canon.is_dir() {
        return Err(format!("not a folder: {}", canon.display()));
    }
    if !within_root(&canon, &root) {
        return Err(format!("path is outside browse root ({})", root.display()));
    }
    let parent = if canon != root { canon.parent().filter(|p| within_root(p, &root)).map(ps) } else { None };

    let rd = std::fs::read_dir(&canon).map_err(|e| format!("cannot read {}: {e}", canon.display()))?;
    let mut dirs = Vec::new();
    let mut clips = Vec::new();
    for entry in rd.flatten() {
        let p = entry.path();
        let Some(name) = p.file_name().and_then(|n| n.to_str()).map(str::to_string) else { continue };
        if is_ignored_name(&name) {
            continue;
        }
        if p.is_dir() {
            dirs.push(FsEntry { name, path: ps(&p), kind: "dir", codec: None, size: 0 });
        } else if p.is_file() && is_clip(&p) {
            let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            let codec = if ext.eq_ignore_ascii_case("r3d") { Some("R3D NE") } else { Some("N-RAW") };
            clips.push(FsEntry { name, path: ps(&p), kind: "clip", codec, size });
        }
    }
    dirs.sort_by_key(|a| a.name.to_lowercase());
    clips.sort_by_key(|a| a.name.to_lowercase());
    dirs.extend(clips);

    Ok(FsListResponse { path: ps(&canon), parent, browse_root: ps(&root), roots: discover_roots(state), entries: dirs })
}

fn handle_fs_mkdir(state: &ServerState, args: FsMkdirArgs) -> Result<serde_json::Value, String> {
    let root = state.browse_root.canonicalize().unwrap_or_else(|_| normalize_path(&state.browse_root));
    let target = if let Some(p) = args.path.filter(|s| !s.trim().is_empty()) {
        normalize_path(Path::new(p.trim()))
    } else if let (Some(parent), Some(name)) = (args.parent, args.name) {
        let trimmed = name.trim();
        if trimmed.is_empty() || trimmed.contains('/') || trimmed.contains('\\') || trimmed == "." || trimmed == ".." {
            return Err("invalid folder name".into());
        }
        normalize_path(&Path::new(parent.trim()).join(trimmed))
    } else {
        return Err("missing folder path or name".into());
    };
    if !within_root(&target, &root) {
        return Err(format!("path is outside browse root ({})", root.display()));
    }
    std::fs::create_dir_all(&target).map_err(|e| format!("cannot create {}: {e}", target.display()))?;
    let canon = target.canonicalize().unwrap_or(target);
    Ok(serde_json::json!({ "path": ps(&canon) }))
}

fn start_queue(state: &Arc<ServerState>, args: StartArgs) -> Result<(), String> {
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("already running".into());
    }
    if let Some(h) = state.worker.lock().unwrap().take() {
        let _ = h.join();
    }
    state.cancel.store(false, Ordering::SeqCst);
    let st = state.clone();
    let handle = std::thread::spawn(move || {
        let backend = backend_of(&args.settings.backend, st.default_backend);
        for job in args.jobs {
            if st.cancel.load(Ordering::SeqCst) {
                break;
            }
            let input = PathBuf::from(&job.path);
            let resized = bluecompress::probe(&input)
                .map(|i| {
                    bluecompress::output_size(i.width, i.height, resolution_of(&job.resolution)) != (i.width, i.height)
                })
                .unwrap_or(false);
            let res_name = if resized { job.resolution.as_str() } else { "original" };
            let output = output_for(&input, args.settings.output_folder.as_deref(), job.quality.clamp(1, 4), res_name);
            if let Some(dir) = output.parent() {
                if let Err(e) = std::fs::create_dir_all(dir) {
                    broadcast_event(
                        &st.sse_clients,
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
            let _guard = st.transcode_lock.lock().unwrap();
            if st.cancel.load(Ordering::SeqCst) {
                break;
            }
            let out_s = ps(&output);
            broadcast_event(&st.sse_clients, "job-start", JobStart { id: job.id, output: out_s.clone() });
            let options = Options {
                input,
                output,
                quality: job.quality.clamp(1, 4),
                resolution: resolution_of(&job.resolution),
                backend,
                threads: st.threads,
            };
            let mut last = Instant::now() - Duration::from_secs(1);
            let clients = st.sse_clients.clone();
            let mut on_progress = |fraction: f64, fps: f64| {
                if fraction >= 1.0 || last.elapsed().as_millis() >= 100 {
                    last = Instant::now();
                    broadcast_event(&clients, "job-progress", JobProgress { id: job.id, fraction, fps });
                }
            };
            match bluecompress::transcode(&options, &st.cancel, &mut on_progress) {
                Ok(o) => {
                    broadcast_event(
                        &st.sse_clients,
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
                    let cancelled = st.cancel.load(Ordering::SeqCst);
                    broadcast_event(
                        &st.sse_clients,
                        "job-error",
                        JobError { id: job.id, error: e.to_string(), cancelled },
                    );
                }
            }
        }
        st.running.store(false, Ordering::SeqCst);
        broadcast_event(&st.sse_clients, "queue-finished", serde_json::Value::Null);
    });
    *state.worker.lock().unwrap() = Some(handle);
    Ok(())
}

fn invoke_command(state: &Arc<ServerState>, cmd: &str, body: &[u8]) -> HttpResponse {
    match cmd {
        "app_info" => {
            let info = AppInfo {
                version: bluecompress::VERSION.into(),
                gpu: bluecompress::backend_description(Backend::Auto),
                cpu: bluecompress::backend_description(Backend::Cpu),
                platform: std::env::consts::OS.into(),
                web: true,
                browse_root: ps(&state.browse_root),
                default_quality: state.default_quality,
                default_resolution: resolution_name(state.default_resolution).into(),
                watch: state.watch_state.lock().unwrap().clone(),
            };
            HttpResponse::json(200, &info)
        }
        "watch_status" => {
            let snap = state.watch_state.lock().unwrap().clone();
            HttpResponse::json(200, &snap)
        }
        "set_watch_settings" => {
            let args: SetWatchSettingsArgs = match serde_json::from_slice(body) {
                Ok(a) => a,
                Err(e) => return HttpResponse::text(400, &format!("invalid set_watch_settings args: {e}")),
            };
            let snap = {
                let mut ws = state.watch_state.lock().unwrap();
                if let Some(c) = args.clear_after_process {
                    ws.clear_after_process = c;
                }
                if let Some(q) = args.quality {
                    ws.quality = q.clamp(1, 4);
                }
                if let Some(r) = args.resolution.as_deref().and_then(parse_resolution) {
                    ws.resolution = resolution_name(r).into();
                }
                if ws.enabled {
                    let out_dir = PathBuf::from(&ws.output_dir);
                    let qv = ws.quality;
                    let rv = ws.resolution.clone();
                    let cv = ws.clear_after_process;
                    save_persisted_settings(&out_dir, |p| {
                        if args.quality.is_some() {
                            p.quality = Some(qv);
                        }
                        if args.resolution.is_some() {
                            p.resolution = Some(rv);
                        }
                        if args.clear_after_process.is_some() {
                            p.clear_after_process = Some(cv);
                        }
                    });
                }
                ws.clone()
            };
            broadcast_event(&state.sse_clients, "watch-status", &snap);
            HttpResponse::json(200, &snap)
        }
        "is_running" => HttpResponse::json(200, &state.running.load(Ordering::SeqCst)),
        "initial_paths" => {
            let paths: Vec<String> = std::mem::take(&mut *state.initial.lock().unwrap());
            HttpResponse::json(200, &paths)
        }
        "scan" => {
            let args: ScanArgs = match serde_json::from_slice(body) {
                Ok(a) => a,
                Err(e) => return HttpResponse::text(400, &format!("invalid scan args: {e}")),
            };
            let mut files = Vec::new();
            for p in &args.paths {
                collect_clips(Path::new(p), &mut files);
            }
            let clips: Vec<Clip> = files.iter().map(|f| describe_clip(f)).collect();
            HttpResponse::json(200, &clips)
        }
        "plan_output" => {
            let args: PlanOutputArgs = match serde_json::from_slice(body) {
                Ok(a) => a,
                Err(e) => return HttpResponse::text(400, &format!("invalid plan_output args: {e}")),
            };
            let (w, h) = bluecompress::output_size(args.width, args.height, resolution_of(&args.resolution));
            HttpResponse::json(200, &(w, h))
        }
        "preview" => {
            let args: PreviewArgs = match serde_json::from_slice(body) {
                Ok(a) => a,
                Err(e) => return HttpResponse::text(400, &format!("invalid preview args: {e}")),
            };
            let res = resolution_of(&args.resolution);
            let backend = backend_of(&args.backend, state.default_backend);
            let crop = args.crop.unwrap_or([0.0, 0.0, 1.0, 1.0]);
            match bluecompress::preview_region(
                Path::new(&args.path),
                args.frame,
                args.quality.clamp(1, 4),
                res,
                backend,
                crop,
                args.width.clamp(16, 8192),
            ) {
                Ok(p) => {
                    let mut out = Vec::with_capacity(32 + p.before_rgba.len() + p.after_rgba.len());
                    for v in [p.width, p.height, p.frames, 0] {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    out.extend_from_slice(&p.source_frame_bytes.to_le_bytes());
                    out.extend_from_slice(&p.output_frame_bytes.to_le_bytes());
                    out.extend_from_slice(&p.before_rgba);
                    out.extend_from_slice(&p.after_rgba);
                    HttpResponse::bytes(200, "application/octet-stream", out)
                }
                Err(e) => HttpResponse::text(400, &e.to_string()),
            }
        }
        "thumbnail" => {
            let args: ThumbnailArgs = match serde_json::from_slice(body) {
                Ok(a) => a,
                Err(e) => return HttpResponse::text(400, &format!("invalid thumbnail args: {e}")),
            };
            let frames = bluecompress::probe(Path::new(&args.path)).map(|i| i.frames).unwrap_or(1);
            match bluecompress::thumbnail(Path::new(&args.path), frames / 4, args.width) {
                Ok(t) => {
                    let mut out = Vec::with_capacity(8 + t.rgba.len());
                    out.extend_from_slice(&t.width.to_le_bytes());
                    out.extend_from_slice(&t.height.to_le_bytes());
                    out.extend_from_slice(&t.rgba);
                    HttpResponse::bytes(200, "application/octet-stream", out)
                }
                Err(e) => HttpResponse::text(400, &e.to_string()),
            }
        }
        "start" => {
            let args: StartArgs = match serde_json::from_slice(body) {
                Ok(a) => a,
                Err(e) => return HttpResponse::text(400, &format!("invalid start args: {e}")),
            };
            match start_queue(state, args) {
                Ok(()) => HttpResponse::json(200, &serde_json::Value::Null),
                Err(e) => HttpResponse::text(409, &e),
            }
        }
        "cancel" | "confirm_quit" => {
            state.cancel.store(true, Ordering::SeqCst);
            HttpResponse::json(200, &serde_json::Value::Null)
        }
        "reveal" => HttpResponse::json(200, &serde_json::Value::Null),
        _ => HttpResponse::text(404, &format!("unknown command: {cmd}")),
    }
}

struct HttpResponse {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl HttpResponse {
    fn text(status: u16, s: &str) -> Self {
        Self { status, content_type: "text/plain; charset=utf-8", body: s.as_bytes().to_vec() }
    }

    fn bytes(status: u16, content_type: &'static str, body: Vec<u8>) -> Self {
        Self { status, content_type, body }
    }

    fn json<T: Serialize>(status: u16, v: &T) -> Self {
        match serde_json::to_vec(v) {
            Ok(body) => Self { status, content_type: "application/json; charset=utf-8", body },
            Err(e) => Self::text(500, &e.to_string()),
        }
    }

    fn write_to(self, mut stream: TcpStream) {
        let reason = match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            409 => "Conflict",
            _ => "Internal Server Error",
        };
        let header = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
            self.status,
            reason,
            self.content_type,
            self.body.len()
        );
        let _ = stream.write_all(header.as_bytes());
        let _ = stream.write_all(&self.body);
        let _ = stream.flush();
    }
}

fn url_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else if b[i] == b'%' && i + 2 < b.len() {
            let hi = (b[i + 1] as char).to_digit(16);
            let lo = (b[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push(((h << 4) | l) as u8);
                i += 3;
                continue;
            }
            out.push(b[i]);
            i += 1;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        if kv.next() == Some(key) {
            return Some(url_decode(kv.next().unwrap_or("")));
        }
    }
    None
}

fn handle_client(mut stream: TcpStream, state: Arc<ServerState>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));

    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });

    let mut req_line = String::new();
    if reader.read_line(&mut req_line).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let raw_target = parts.next().unwrap_or("/");
    let (path, query) = match raw_target.split_once('?') {
        Some((p, q)) => (p, q),
        None => (raw_target, ""),
    };

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse::<usize>().unwrap_or(0).min(16 * 1024 * 1024);
            }
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }

    match (method, path) {
        ("GET", "/" | "/index.html") => {
            HttpResponse::bytes(200, "text/html; charset=utf-8", INDEX_HTML.as_bytes().to_vec()).write_to(stream);
        }
        ("GET", "/styles.css") => {
            HttpResponse::bytes(200, "text/css; charset=utf-8", STYLES_CSS.as_bytes().to_vec()).write_to(stream);
        }
        ("GET", "/app.js") => {
            HttpResponse::bytes(200, "application/javascript; charset=utf-8", APP_JS.as_bytes().to_vec())
                .write_to(stream);
        }
        ("GET", "/icon.png" | "/favicon.ico") => {
            HttpResponse::bytes(200, "image/png", ICON_PNG.to_vec()).write_to(stream);
        }
        ("GET", "/api/health" | "/healthz") => {
            HttpResponse::json(200, &serde_json::json!({ "ok": true, "version": bluecompress::VERSION }))
                .write_to(stream);
        }
        ("GET", "/api/events") => {
            let _ = stream.set_read_timeout(None);
            let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
            let _ = stream.set_nodelay(true);
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
            if stream.write_all(hdr.as_bytes()).is_err() {
                return;
            }
            let snap = state.watch_state.lock().unwrap().clone();
            if let Ok(s) = serde_json::to_string(&serde_json::json!({ "event": "watch-status", "payload": snap })) {
                let _ = stream.write_all(format!("data: {s}\n\n").as_bytes());
            }
            let _ = stream.flush();
            state.sse_clients.lock().unwrap().push(stream);
        }
        ("GET", "/api/fs/list") => {
            let req_path = query_param(query, "path");
            match handle_fs_list(&state, req_path.as_deref()) {
                Ok(res) => HttpResponse::json(200, &res).write_to(stream),
                Err(e) => HttpResponse::text(400, &e).write_to(stream),
            }
        }
        ("POST", "/api/fs/list") => {
            let req_path = serde_json::from_slice::<FsListArgs>(&body)
                .ok()
                .and_then(|a| a.path)
                .or_else(|| query_param(query, "path"));
            match handle_fs_list(&state, req_path.as_deref()) {
                Ok(res) => HttpResponse::json(200, &res).write_to(stream),
                Err(e) => HttpResponse::text(400, &e).write_to(stream),
            }
        }
        ("POST", "/api/fs/mkdir") => match serde_json::from_slice::<FsMkdirArgs>(&body) {
            Ok(args) => match handle_fs_mkdir(&state, args) {
                Ok(v) => HttpResponse::json(200, &v).write_to(stream),
                Err(e) => HttpResponse::text(400, &e).write_to(stream),
            },
            Err(e) => HttpResponse::text(400, &format!("invalid mkdir args: {e}")).write_to(stream),
        },
        ("GET" | "POST", p) if p.starts_with("/api/invoke/") => {
            let cmd = &p["/api/invoke/".len()..];
            invoke_command(&state, cmd, &body).write_to(stream);
        }
        _ => {
            HttpResponse::text(404, "not found").write_to(stream);
        }
    }
}

pub fn run_serve(cfg: ServeConfig, cancel: &'static AtomicBool) -> u8 {
    let listener = match TcpListener::bind(&cfg.listen) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot bind {}: {e}", cfg.listen);
            return 1;
        }
    };

    let _ = std::fs::create_dir_all(&cfg.browse_root);
    let initial_watch = match &cfg.watch {
        Some(wcfg) => {
            let persisted = load_persisted_settings(&wcfg.output_dir);
            let qv = persisted.quality.filter(|q| (1..=4).contains(q)).unwrap_or(wcfg.quality);
            let rv = persisted.resolution.as_deref().and_then(parse_resolution).unwrap_or(wcfg.resolution);
            let cv = persisted.clear_after_process.unwrap_or(wcfg.clear_after_process);
            WatchSnapshot {
                enabled: true,
                input_dir: ps(&wcfg.input_dir),
                output_dir: ps(&wcfg.output_dir),
                quality: qv,
                resolution: resolution_name(rv).into(),
                poll_seconds: wcfg.poll_seconds,
                clear_after_process: cv,
                state: "idle".into(),
                ..Default::default()
            }
        }
        None => WatchSnapshot::default(),
    };

    let state = Arc::new(ServerState {
        browse_root: cfg.browse_root,
        default_quality: cfg.default_quality,
        default_resolution: cfg.default_resolution,
        default_backend: cfg.default_backend,
        threads: cfg.threads,
        running: Arc::new(AtomicBool::new(false)),
        cancel: Arc::new(AtomicBool::new(false)),
        worker: Mutex::new(None),
        initial: Mutex::new(cfg.initial_paths),
        sse_clients: Arc::new(Mutex::new(Vec::new())),
        watch_state: Arc::new(Mutex::new(initial_watch)),
        transcode_lock: Arc::new(Mutex::new(())),
    });

    println!("Blue Compressor {} web UI listening on http://{}", bluecompress::VERSION, cfg.listen);
    let _ = std::io::stdout().flush();

    if let Some(wcfg) = cfg.watch {
        let watch_state = state.watch_state.clone();
        let sse = state.sse_clients.clone();
        let lock = state.transcode_lock.clone();
        let hook = Arc::new(move |snap: &WatchSnapshot| {
            broadcast_event(&sse, "watch-status", snap);
        });
        std::thread::spawn(move || {
            run_watch(wcfg, cancel, watch_state, Some(hook), Some(lock));
        });
    }

    // Heartbeat and clean shutdown thread.
    {
        let st = state.clone();
        std::thread::spawn(move || {
            let mut ticks = 0u32;
            loop {
                std::thread::sleep(Duration::from_millis(200));
                if cancel.load(Ordering::Relaxed) {
                    st.cancel.store(true, Ordering::SeqCst);
                    if let Some(h) = st.worker.lock().unwrap().take() {
                        let _ = h.join();
                    }
                    std::process::exit(0);
                }
                ticks += 1;
                if ticks >= 75 {
                    ticks = 0;
                    let mut guard = st.sse_clients.lock().unwrap();
                    guard.retain_mut(|stream| stream.write_all(b": ping\n\n").and_then(|_| stream.flush()).is_ok());
                }
            }
        });
    }

    for stream in listener.incoming().flatten() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let st = state.clone();
        std::thread::spawn(move || handle_client(stream, st));
    }
    0
}
