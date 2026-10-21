//! Area (box) resampling of CFA planes with Bayer-phase-correct footprints.

/// Separable taps: for each output index, a contiguous run of input samples and weights.
#[derive(Debug, Clone, Default)]
pub struct Taps {
    /// First input index per output.
    pub first: Vec<i32>,
    /// Taps per output.
    pub count: Vec<i32>,
    /// Offset of each output's weights in `weight`.
    pub offset: Vec<i32>,
    pub weight: Vec<f32>,
}

/// Output plane sample X of a CFA colour at sensor phase `phase` (0 or 1) sits at output
/// sensor pixel 2X+phase. Mapped into the source sensor (pitch ratio `scale`, centres of
/// both sensors aligned), it becomes a box `scale` plane samples wide in that colour's
/// source plane: the area its output plane cell covers. A box never rings, which matters
/// next to clipped highlights (sharper kernels overshoot there).
pub fn make_taps(
    src_plane: usize,
    dst_plane: usize,
    src_full: usize,
    dst_full: usize,
    phase: usize,
    scale: f64,
) -> Taps {
    let mut t = Taps::default();
    let last = src_plane as i64 - 1;
    for x in 0..dst_plane {
        let sensor =
            madd64((2.0 * x as f64 + phase as f64 + 0.5) - dst_full as f64 / 2.0, scale, src_full as f64 / 2.0);
        let centre = (sensor - phase as f64 - 0.5) / 2.0;
        let half = scale.max(1.0) / 2.0;
        let (a, b) = (centre - half, centre + half);
        let (k0, k1) = ((a + 0.5).floor() as i64, (b - 0.5).ceil() as i64);
        let w: Vec<f64> = (k0..=k1).map(|k| (b.min(k as f64 + 0.5) - a.max(k as f64 - 0.5)).max(0.0)).collect();
        let sum: f64 = w.iter().sum();
        // Clamp to the plane: fold weights of out-of-range samples onto the edge sample.
        let first = k0.clamp(0, last);
        let lastk = k1.clamp(0, last);
        let mut folded = vec![0.0f64; (lastk - first + 1) as usize];
        for k in k0..=k1 {
            folded[(k.clamp(0, last) - first) as usize] += w[(k - k0) as usize];
        }
        t.first.push(first as i32);
        t.count.push(folded.len() as i32);
        t.offset.push(t.weight.len() as i32);
        t.weight.extend(folded.iter().map(|v| (v / sum) as f32));
    }
    t
}

/// a * b + c, fused where the CPU has a native FMA (as C/C++ compilers contract it there),
/// plain elsewhere, where `mul_add` would be a slow library call.
#[inline(always)]
fn madd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

#[inline(always)]
fn madd64(a: f64, b: f64, c: f64) -> f64 {
    if cfg!(any(target_arch = "aarch64", target_feature = "fma")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Resamples a float plane (sw x sh) to (dw x dh) with horizontal and vertical taps.
pub fn resample_plane(
    src: &[f32],
    sw: usize,
    sh: usize,
    dst: &mut [f32],
    dw: usize,
    dh: usize,
    tx: &Taps,
    ty: &Taps,
    scratch: &mut Vec<f32>,
) {
    // Horizontal: sh rows of dw.
    scratch.clear();
    scratch.resize(sh * dw, 0.0);
    for y in 0..sh {
        let s = &src[y * sw..(y + 1) * sw];
        let d = &mut scratch[y * dw..(y + 1) * dw];
        for (x, out) in d.iter_mut().enumerate() {
            let (f, n, o) = (tx.first[x] as usize, tx.count[x] as usize, tx.offset[x] as usize);
            let mut acc = 0.0f32;
            for k in 0..n {
                acc = madd(tx.weight[o + k], s[f + k], acc);
            }
            *out = acc;
        }
    }
    // Vertical.
    for y in 0..dh {
        let (f, n, o) = (ty.first[y] as usize, ty.count[y] as usize, ty.offset[y] as usize);
        let d = &mut dst[y * dw..(y + 1) * dw];
        d.fill(0.0);
        for k in 0..n {
            let s = &scratch[(f + k) * dw..(f + k + 1) * dw];
            let wk = ty.weight[o + k];
            for (dv, sv) in d.iter_mut().zip(s) {
                *dv = madd(wk, *sv, *dv);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        // 8K -> 4K and ZR -> 4K style ratios, both CFA phases.
        for (sp, dp, sf, df, scale) in
            [(4134, 1926, 8268, 3852, 8256.0 / 3840.0), (3030, 1926, 6060, 3852, 6048.0 / 3840.0)]
        {
            for phase in 0..2 {
                let t = make_taps(sp, dp, sf, df, phase, scale);
                for x in 0..dp {
                    let (o, n) = (t.offset[x] as usize, t.count[x] as usize);
                    let s: f32 = t.weight[o..o + n].iter().sum();
                    assert!((s - 1.0).abs() < 1e-5, "output {x}: sum {s}");
                    assert!(t.first[x] >= 0 && (t.first[x] + t.count[x]) as usize <= sp);
                }
            }
        }
    }

    #[test]
    fn unit_scale_is_identity() {
        let t = make_taps(100, 100, 200, 200, 0, 1.0);
        for x in 0..100 {
            assert_eq!((t.first[x], t.count[x]), (x as i32, 1));
            assert_eq!(t.weight[t.offset[x] as usize], 1.0);
        }
    }
}
