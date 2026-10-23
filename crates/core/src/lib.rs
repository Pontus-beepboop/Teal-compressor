//! Blue Compressor core: lossless repacking, requantization and raw resizing of Nikon
//! N-RAW (.NEV) and R3D NE (.R3D) TICO RAW clips, keeping container, metadata and audio.
//!
//! The public surface is small: [`probe`], [`transcode`], [`output_size`] and
//! [`backend_description`]. A C API with the same shape lives in [`ffi`]
//! (header: `crates/core/include/bluecompress.h`).

// Codec loops index several parallel arrays by the same position; iterators would hide that.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

mod bitio;
pub mod bmff;
pub mod ffi;
pub mod frame;
mod gpu;
pub mod nraw;
mod preview;
pub mod quality;
pub mod requant;
pub mod resample;
pub mod resize;
pub mod tico;
mod transcode;
mod util;

pub use preview::{Preview, Thumbnail};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Video codec of a clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// Nikon N-RAW (.NEV), e.g. Z8, Z9.
    NRaw,
    /// R3D NE (.R3D), e.g. Nikon ZR.
    R3dNe,
}

#[derive(Debug, Clone)]
pub struct ClipInfo {
    pub codec: Codec,
    /// Coded Bayer size.
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: f64,
    pub file_bytes: u64,
    pub video_bytes: u64,
    pub audio_tracks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolution {
    Original,
    /// 6048 x 3402 active (6060 x 3410 coded), only from larger sources.
    K6,
    /// 3840 x 2160 active UHD (3852 x 2168 coded).
    K4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub input: PathBuf,
    pub output: PathBuf,
    /// 1 lossless repack, 2 archival, 3 high, 4 compact.
    pub quality: u8,
    pub resolution: Resolution,
    pub backend: Backend,
    /// 0 = all cores.
    pub threads: usize,
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub frames: u32,
    pub seconds: f64,
    pub fps: f64,
    pub backend: String,
}

/// Reads a clip's container.
pub fn probe(path: &std::path::Path) -> Result<ClipInfo> {
    let input = bmff::InputFile::open(path)?;
    let m = bmff::Movie::parse(&input)?;
    let v = m.video();
    let codec = match v.codec {
        c if c == bmff::fourcc(b"NRAW") => Codec::NRaw,
        c if c == bmff::fourcc(b"NR3D") => Codec::R3dNe,
        c => return Err(Error(format!("not a Nikon N-RAW or R3D NE clip (codec {})", bmff::fourcc_str(c)))),
    };
    Ok(ClipInfo {
        codec,
        width: v.width,
        height: v.height,
        frames: v.samples.len() as u32,
        fps: quality::video_frame_rate(&m),
        file_bytes: input.size(),
        video_bytes: v.samples.iter().map(|s| s.size as u64).sum(),
        audio_tracks: m.tracks.iter().filter(|t| t.handler == bmff::fourcc(b"soun")).count() as u32,
    })
}

/// Output Bayer size for a source size and requested resolution (the source's camera
/// margin of 12 x 8 is kept).
pub fn output_size(width: u32, height: u32, resolution: Resolution) -> (u32, u32) {
    let src = quality::Geometry {
        width,
        height,
        active_width: if width > 12 { width - 12 } else { width },
        active_height: if height > 8 { height - 8 } else { height },
    };
    let g = quality::target_geometry(src, resolution);
    (g.width, g.height)
}

/// Human readable description of a backend ("Metal: Apple GPU", "CPU (12 threads)").
pub fn backend_description(backend: Backend) -> String {
    match backend {
        Backend::Cpu => format!("CPU ({} threads)", transcode::hardware_threads()),
        b => gpu::description(b).unwrap_or_else(|| match b {
            Backend::Metal => "not available (Metal needs macOS)".into(),
            Backend::Cuda => "not available (needs the cuda feature, an NVIDIA GPU and driver, and NVRTC 12.x)".into(),
            _ => format!("CPU only ({} threads)", transcode::hardware_threads()),
        }),
    }
}

/// Transcodes one clip. `progress(fraction, fps)` is called from the calling thread about
/// ten times a second; setting `cancel` stops the job and removes the partial output.
/// The output is written to `<output>.partial` and renamed when complete.
pub fn transcode(options: &Options, cancel: &AtomicBool, progress: &mut dyn FnMut(f64, f64)) -> Result<Outcome> {
    transcode::transcode(options, cancel, progress)
}

/// Renders frame `frame` (clamped) of the clip before and after compression with the given
/// quality and resolution, as a simple debayered preview `width` pixels wide. Both halves
/// go through the same rendering, so differences come from compression alone.
pub fn preview_frame(
    path: &std::path::Path,
    frame: u32,
    quality: u8,
    resolution: Resolution,
    backend: Backend,
    width: u32,
) -> Result<Preview> {
    preview::preview_frame(path, frame, quality, resolution, backend, [0.0, 0.0, 1.0, 1.0], width)
}

/// Like [`preview_frame`] for the region `crop` = [x, y, width, height] in fractions of the
/// frame (for zooming in). The last decoded frame is cached, so panning and zooming around
/// one frame with the same settings costs only the rendering.
pub fn preview_region(
    path: &std::path::Path,
    frame: u32,
    quality: u8,
    resolution: Resolution,
    backend: Backend,
    crop: [f64; 4],
    width: u32,
) -> Result<Preview> {
    preview::preview_frame(path, frame, quality, resolution, backend, crop, width)
}

/// A small picture of frame `frame` (clamped) of a clip, `width` pixels wide, for lists and
/// thumbnails. Decodes one frame; does not disturb the [`preview_region`] cache.
pub fn thumbnail(path: &std::path::Path, frame: u32, width: u32) -> Result<Thumbnail> {
    preview::thumbnail(path, frame, width)
}
