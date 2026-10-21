//! Coefficient-domain transcoding of TICO RAW codestreams: entropy decode, optionally
//! requantize every precinct to a coarser truncation, entropy re-encode without the
//! camera's constant-bitrate padding. No wavelet or colour transform is involved, so a
//! lossless repack reproduces the camera's decoded pixels exactly.

use std::sync::Mutex;

use crate::tico::*;
use crate::util::parallel_for;
use crate::Result;

#[derive(Debug, Clone, Copy)]
pub struct RequantParams {
    /// Minimum truncation rank per precinct (see [`rank_of`]); None keeps the camera's.
    pub target_rank: Option<i32>,
    /// Choose the cheapest GCLI coding mode per band instead of the camera's.
    pub optimize_modes: bool,
    /// Drop coded-but-empty bit planes (groups whose magnitudes are all zero).
    pub tidy: bool,
}

impl Default for RequantParams {
    fn default() -> Self {
        RequantParams { target_rank: None, optimize_modes: true, tidy: true }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CodestreamStats {
    pub precincts: u64,
    pub requantized: u64,
    /// Sum of original ranks (for averaging).
    pub rank_sum: i64,
}

/// Ranks order (Q, R) pairs from finest to coarsest; gtli is monotone in rank.
#[inline]
pub fn rank_of(q: i32, r: i32, bands: i32) -> i32 {
    q * bands + (bands - r.clamp(0, bands))
}

#[inline]
pub fn from_rank(rank: i32, bands: i32) -> (i32, i32) {
    (rank / bands, bands - rank % bands)
}

/// Clears coefficients past the band's end and lowers GCLIs of groups left without data.
fn tidy(band: &Band, bl: &mut BandLine) {
    for j in band.width..bl.mag.len() {
        bl.mag[j] = 0;
        bl.neg[j] = 0;
    }
    for g in 0..band.groups {
        let m = &bl.mag[g * GROUP..g * GROUP + GROUP];
        if m[0] | m[1] | m[2] | m[3] == 0 {
            bl.gcli[g] = bl.gtli as u8;
        }
    }
}

fn requant_line(band: &Band, bl: &mut BandLine, new_gtli: i32) {
    let t0 = bl.gtli;
    let t1 = new_gtli;
    bl.gtli = t1;
    for g in 0..band.groups {
        let g0 = bl.gcli[g] as i32;
        let mut r = [0u32; GROUP];
        let mut mx = 0;
        for j in 0..GROUP {
            if g * GROUP + j < band.width {
                r[j] = dequant_uniform(bl.mag[g * GROUP + j], g0, t0);
            }
            mx = mx.max(r[j]);
        }
        let gnew = bit_length(mx);
        let mut any = 0;
        for j in 0..GROUP {
            let i = g * GROUP + j;
            bl.mag[i] = if gnew > t1 { quant_uniform(r[j], gnew, t1) } else { 0 };
            any |= bl.mag[i];
            if bl.mag[i] == 0 {
                bl.neg[i] = 0;
            }
        }
        bl.gcli[g] = if any != 0 { gnew as u8 } else { t1 as u8 };
    }
}

/// Transcodes one codestream, parallel over slices.
pub fn requant_codestream(cs: &[u8], prm: &RequantParams, threads: usize) -> Result<(Vec<u8>, CodestreamStats)> {
    let (header, first_slice) = TicoHeader::parse(cs)?;
    let l = Layout::make(&header, header.width as usize, header.height as usize)?;
    let precincts = find_precincts(cs, first_slice, &l)?;
    let nb = l.bands.len();
    let slices: Vec<Mutex<(Vec<u8>, CodestreamStats)>> =
        (0..l.slices).map(|_| Mutex::new((Vec::new(), CodestreamStats::default()))).collect();
    let lossless = prm.target_rank.is_none();

    parallel_for(l.slices, threads, |s| {
        let mut dec = PredState::default();
        let mut enc = PredState::default();
        dec.reset(nb);
        enc.reset(nb);
        let mut pd = PrecinctData::default();
        let mut scratch = EncodeScratch::default();
        let mut out = Vec::with_capacity(cs.len() / l.slices + 4096);
        out.extend_from_slice(&[0xff, 0x20, 0x00, 0x04, (s >> 8) as u8, s as u8]);
        let mut st = CodestreamStats::default();
        let p0 = s * l.slice_height;
        let p1 = l.precincts.min(p0 + l.slice_height);
        for p in p0..p1 {
            decode_precinct(cs, precincts[p], &l, p, &mut dec, &mut pd, lossless)?;
            let rank0 = rank_of(pd.q, pd.r, nb as i32);
            st.precincts += 1;
            st.rank_sum += rank0 as i64;
            match prm.target_rank {
                Some(target) if target > rank0 => {
                    let (q, r) = from_rank(target, nb as i32);
                    pd.q = q;
                    pd.r = r;
                    st.requantized += 1;
                    for lines in pd.packets.iter_mut() {
                        for bl in lines.iter_mut() {
                            let band = &l.bands[bl.band];
                            let t1 = compute_gtli(band.gain, band.priority, q, r);
                            if t1 != bl.gtli {
                                requant_line(band, bl, t1);
                            }
                            tidy(band, bl);
                        }
                    }
                }
                Some(_) if prm.tidy => {
                    for lines in pd.packets.iter_mut() {
                        for bl in lines.iter_mut() {
                            tidy(&l.bands[bl.band], bl);
                        }
                    }
                }
                _ => {}
            }
            if prm.optimize_modes {
                choose_modes(&l, &mut pd, &enc);
            }
            encode_precinct(&l, &pd, Some(cs), &mut enc, &mut scratch, &mut out)?;
        }
        *slices[s].lock().unwrap() = (out, st);
        Ok(())
    })?;

    let mut total = CodestreamStats::default();
    let mut parts = Vec::with_capacity(l.slices);
    for m in slices {
        let (v, st) = m.into_inner().unwrap();
        total.precincts += st.precincts;
        total.requantized += st.requantized;
        total.rank_sum += st.rank_sum;
        parts.push(v);
    }
    Ok((assemble_codestream(&header, header.width, header.height, &parts), total))
}
