//! Quality presets and output geometry.

use crate::bmff::Movie;
use crate::Resolution;

/// Target ranks, measured on Z8 and ZR footage (rank = Q*25 + 25-R,
/// higher is coarser; the Z8 records around rank 139, the ZR around 117 to 130):
/// - rank 135: 84% (Z8) / 75% (ZR) of the size, mean error ~3/16383 raw, below sensor noise
/// - rank 140: 67% / 58%, shadow noise slightly smoothed
/// - rank 160: 43% / 43%, visible smoothing in deep shadows on the Z8
///
/// Precincts the camera already coded coarser than the target are left untouched.
/// Returns None for the lossless repack.
pub fn rank_for_quality(quality: u8) -> Option<i32> {
    match quality {
        0 | 1 => None,
        2 => Some(135),
        3 => Some(140),
        _ => Some(160),
    }
}

/// Rank used by the re-encoder for resized output. Resized output has no camera coding to
/// preserve; Q1 uses a fine truncation that is transparent to the resampled signal.
pub fn encode_rank_for_quality(quality: u8) -> i32 {
    if let Some(r) = std::env::var("BC_ENCODE_RANK").ok().and_then(|v| v.parse().ok()) {
        return r; // calibration only
    }
    rank_for_quality(quality).unwrap_or(125)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Geometry {
    /// Codestream (Bayer) size.
    pub width: u32,
    pub height: u32,
    /// Image area the decoder shows.
    pub active_width: u32,
    pub active_height: u32,
}

pub fn source_geometry(m: &Movie) -> Geometry {
    let v = m.video();
    let (w, h) = (v.width, v.height);
    Geometry {
        width: w,
        height: h,
        active_width: if m.active_width != 0 {
            m.active_width
        } else if w > 12 {
            w - 12
        } else {
            w
        },
        active_height: if m.active_height != 0 {
            m.active_height
        } else if h > 8 {
            h - 8
        } else {
            h
        },
    }
}

/// Output geometry for a resolution preset. Presets set the active area (6K: 6048x3402,
/// 4K: 3840x2160 UHD) and keep the source's margin; the source geometry is returned when
/// the preset is not smaller than the source.
pub fn target_geometry(src: Geometry, resolution: Resolution) -> Geometry {
    let (aw, ah) = match resolution {
        Resolution::K6 => (6048, 3402),
        Resolution::K4 => (3840, 2160),
        Resolution::Original => return src,
    };
    if src.active_width <= aw {
        return src;
    }
    Geometry {
        active_width: aw,
        active_height: ah,
        width: aw + (src.width - src.active_width),
        height: ah + (src.height - src.active_height),
    }
}

pub fn video_frame_rate(m: &Movie) -> f64 {
    let v = m.video();
    if v.timescale == 0 || v.sample_delta == 0 {
        0.0
    } else {
        v.timescale as f64 / v.sample_delta as f64
    }
}
