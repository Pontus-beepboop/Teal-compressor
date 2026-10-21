//! Nikon per-frame sample wrapper shared by N-RAW (.NEV) and R3D NE (.R3D):
//!
//! ```text
//! NRAW box
//!   NRFM  "0100"
//!   NRFH  NRHM "0100" + NRMT tag boxes (width 0x0100, height 0x0101, EXIF-like
//!         exposure tags, CFA pattern, Nikon maker note IFD, ...)
//!   NRTH  NRTI "0100" + u32 codestream length, NRTA (reserved, zero)
//!   <TICO RAW codestream: SOC ... EOC>
//! ```

use crate::bmff::fourcc;
use crate::{Error, Result};

/// Cameras write codestreams in whole 1 KiB units, zero padded before EOC; outputs keep that
/// layout so they are structured like camera files.
pub const CODESTREAM_ALIGN: usize = 1024;

#[inline]
fn rd32(d: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(d[at..at + 4].try_into().unwrap())
}

fn require(c: bool, what: &str) -> Result<()> {
    if c {
        Ok(())
    } else {
        Err(Error(what.into()))
    }
}

#[derive(Debug, Clone, Default)]
pub struct NrawFrame {
    /// Bytes before the codestream.
    pub header: Vec<u8>,
    /// Offset of the codestream inside the sample.
    pub codestream_offset: usize,
    pub codestream_size: usize,
    /// Offset of NRTI's u32 length inside header.
    pub nrti_offset: Option<usize>,
    /// NRMT width/height tag value offsets inside header.
    pub width_tag: Option<usize>,
    pub height_tag: Option<usize>,
}

impl NrawFrame {
    pub fn parse(s: &[u8]) -> Result<NrawFrame> {
        let size = s.len();
        require(size >= 16 && rd32(s, 4) == fourcc(b"NRAW"), "video sample is not an NRAW frame")?;
        require(rd32(s, 0) as usize == size, "NRAW frame size mismatch")?;
        let mut f = NrawFrame::default();
        let mut at = 8;
        while at + 8 <= size {
            if s[at] == 0xff && s[at + 1] == 0x10 {
                break; // SOC
            }
            let len = rd32(s, at) as usize;
            let kind = rd32(s, at + 4);
            require(len >= 8 && at + len <= size, "bad NRAW child box")?;
            if kind == fourcc(b"NRFH") {
                // NRHM followed by NRMT tag boxes.
                let mut p = at + 8;
                while p + 8 <= at + len {
                    let l = rd32(s, p) as usize;
                    require(l >= 8 && p + l <= at + len, "bad NRFH child")?;
                    if rd32(s, p + 4) == fourcc(b"NRMT") && l >= 16 {
                        let group = u16::from_be_bytes([s[p + 8], s[p + 9]]);
                        let tag = u16::from_be_bytes([s[p + 10], s[p + 11]]);
                        if group == 0x0110 && tag == 0x0100 {
                            f.width_tag = Some(p + l - 4);
                        }
                        if group == 0x0110 && tag == 0x0101 {
                            f.height_tag = Some(p + l - 4);
                        }
                    }
                    p += l;
                }
            } else if kind == fourcc(b"NRTH") {
                let mut p = at + 8;
                while p + 8 <= at + len {
                    let l = rd32(s, p) as usize;
                    require(l >= 8 && p + l <= at + len, "bad NRTH child")?;
                    if rd32(s, p + 4) == fourcc(b"NRTI") && l >= 16 {
                        f.nrti_offset = Some(p + 12);
                    }
                    p += l;
                }
            }
            at += len;
        }
        require(at + 4 <= size && s[at] == 0xff && s[at + 1] == 0x10, "codestream not found in NRAW frame")?;
        f.codestream_offset = at;
        f.codestream_size = size - at;
        f.header = s[..at].to_vec();
        Ok(f)
    }

    /// Builds a new sample around `cs`, zero padding it to whole KiB before EOC (as the
    /// cameras do) and patching the sizes; optionally rewrites the frame size tags.
    pub fn build(&self, cs: &[u8], size: Option<(u32, u32)>) -> Result<Vec<u8>> {
        let n = cs.len();
        require(n >= 4 && cs[n - 2] == 0xff && cs[n - 1] == 0x11, "codestream does not end in EOC")?;
        let padded = n.div_ceil(CODESTREAM_ALIGN) * CODESTREAM_ALIGN;
        let mut out = Vec::with_capacity(self.header.len() + padded);
        out.extend_from_slice(&self.header);
        let cs_at = out.len();
        out.extend_from_slice(&cs[..n - 2]);
        out.resize(cs_at + padded - 2, 0);
        out.extend_from_slice(&[0xff, 0x11]);
        // Lcod sits right after the PIH marker and its length field.
        let limit = (cs_at + 256).min(out.len().saturating_sub(8));
        if let Some(p) = (cs_at..limit).find(|&p| out[p] == 0xff && out[p + 1] == 0x12) {
            out[p + 4..p + 8].copy_from_slice(&(padded as u32).to_be_bytes());
        }
        let total = out.len() as u32;
        out[..4].copy_from_slice(&total.to_be_bytes());
        if let Some(o) = self.nrti_offset {
            out[o..o + 4].copy_from_slice(&(padded as u32).to_be_bytes());
        }
        if let Some((w, h)) = size {
            if let Some(o) = self.width_tag {
                out[o..o + 4].copy_from_slice(&w.to_be_bytes());
            }
            if let Some(o) = self.height_tag {
                out[o..o + 4].copy_from_slice(&h.to_be_bytes());
            }
        }
        Ok(out)
    }
}
