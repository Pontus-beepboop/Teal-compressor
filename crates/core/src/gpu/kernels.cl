// OpenCL C kernels for the resize pipeline (AMD and other OpenCL GPUs). A port of
// kernels.cu with the same buffer plan, written to give results bit-identical to the
// CPU path (frame.rs / resample.rs): no contraction of a*b+c unless the CPU build fuses
// it too (BC_FMA), and the transfer curve in double precision like Transfer.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL FP_CONTRACT OFF
typedef ulong u64;
#ifdef BC_FMA
#define MADD(a, b, c) fma((a), (b), (c))
#else
#define MADD(a, b, c) ((a) * (b) + (c))
#endif

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---- horizontal 5/3, one level, rows of `stride` starting at `base`
__kernel void ih_even(__global int* buf, __global int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = get_global_id(0);
    const unsigned y = get_global_id(1);
    if (k >= lows || y >= rows) return;
    const u64 r = base + ((u64)(y)) * stride;
    __global const int* hi = buf + r + lows;
    tmp[r + 2 * k] = buf[r + k] - ((hi[clampi(k - 1, 0, highs - 1)] + hi[clampi(k, 0, highs - 1)] + 2) >> 2);
}

__kernel void ih_odd(__global int* buf, __global int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = get_global_id(0);
    const unsigned y = get_global_id(1);
    if (k >= highs || y >= rows) return;
    const u64 r = base + ((u64)(y)) * stride;
    const int e = tmp[r + 2 * k];
    const int right = 2 * k + 2 < len ? tmp[r + 2 * k + 2] : e;
    tmp[r + 2 * k + 1] = buf[r + lows + k] + ((e + right) >> 1);
}

__kernel void fh_high(__global int* buf, __global int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = get_global_id(0);
    const unsigned y = get_global_id(1);
    if (k >= highs || y >= rows) return;
    const u64 r = base + ((u64)(y)) * stride;
    const int e = buf[r + 2 * k];
    const int right = 2 * k + 2 < len ? buf[r + 2 * k + 2] : e;
    tmp[r + lows + k] = buf[r + 2 * k + 1] - ((e + right) >> 1);
}

__kernel void fh_low(__global int* buf, __global int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = get_global_id(0);
    const unsigned y = get_global_id(1);
    if (k >= lows || y >= rows) return;
    const u64 r = base + ((u64)(y)) * stride;
    __global const int* hi = tmp + r + lows;
    tmp[r + k] = buf[r + 2 * k] + ((hi[clampi(k - 1, 0, highs - 1)] + hi[clampi(k, 0, highs - 1)] + 2) >> 2);
}

__kernel void copy_rows(__global int* dst, __global const int* src, unsigned base, unsigned stride, int len, unsigned rows) {
    const int x = get_global_id(0);
    const unsigned y = get_global_id(1);
    if (x >= len || y >= rows) return;
    const u64 i = base + ((u64)(y)) * stride + x;
    dst[i] = src[i];
}

__kernel void copy_span(__global int* dst, __global const int* src, u64 d, u64 s, unsigned n) {
    const unsigned i = get_global_id(0);
    if (i < n) dst[d + i] = src[s + i];
}

// ---- vertical 5/3 (lo, hi, img are element offsets of one component)
int evenRow(__global const int* lo, __global const int* hi, unsigned w, int highRows, int j, unsigned x) {
    return lo[((u64)(j)) * w + x] -
           ((hi[((u64)(clampi(j - 1, 0, highRows - 1))) * w + x] + hi[((u64)(clampi(j, 0, highRows - 1))) * w + x] + 2) >> 2);
}

__kernel void iv(__global const int* coef, __global int* img, u64 lo, u64 hi, u64 im, unsigned w, int ch, int copy) {
    const unsigned x = get_global_id(0);
    const int k = get_global_id(1);
    const int lowRows = (ch + 1) / 2, highRows = ch / 2;
    if (x >= w || k >= lowRows) return;
    __global int* dst = img + im;
    if (copy) {
        dst[((u64)(2 * k)) * w + x] = coef[lo + ((u64)(k)) * w + x];
        if (2 * k + 1 < ch) dst[((u64)(2 * k + 1)) * w + x] = coef[hi + ((u64)(k)) * w + x];
        return;
    }
    const int e = evenRow(coef + lo, coef + hi, w, highRows, k, x);
    dst[((u64)(2 * k)) * w + x] = e;
    if (2 * k + 1 < ch) {
        const int next = 2 * k + 2 < ch ? evenRow(coef + lo, coef + hi, w, highRows, k + 1, x) : e;
        dst[((u64)(2 * k + 1)) * w + x] = coef[hi + ((u64)(k)) * w + x] + ((e + next) >> 1);
    }
}

__kernel void fv_high(__global const int* img, __global int* coef, u64 hi, u64 im, unsigned w, int ch) {
    const unsigned x = get_global_id(0);
    const int k = get_global_id(1);
    if (x >= w || k >= ch / 2) return;
    __global const int* src = img + im;
    const int e = src[((u64)(2 * k)) * w + x];
    const int n = 2 * k + 2 < ch ? src[((u64)(2 * k + 2)) * w + x] : e;
    coef[hi + ((u64)(k)) * w + x] = src[((u64)(2 * k + 1)) * w + x] - ((e + n) >> 1);
}

__kernel void fv_low(__global const int* img, __global int* coef, u64 lo, u64 hi, u64 im, unsigned w, int ch, int copy) {
    const unsigned x = get_global_id(0);
    const int k = get_global_id(1);
    const int lowRows = (ch + 1) / 2, highRows = ch / 2;
    if (x >= w || k >= lowRows) return;
    if (copy) {
        coef[lo + ((u64)(k)) * w + x] = img[im + ((u64)(2 * k)) * w + x];
        if (2 * k + 1 < ch) coef[hi + ((u64)(k)) * w + x] = img[im + ((u64)(2 * k + 1)) * w + x];
        return;
    }
    const int a = coef[hi + ((u64)(clampi(k - 1, 0, highRows - 1))) * w + x];
    const int b = coef[hi + ((u64)(clampi(k, 0, highRows - 1))) * w + x];
    coef[lo + ((u64)(k)) * w + x] = img[im + ((u64)(2 * k)) * w + x] + ((a + b + 2) >> 2);
}

// ---- Star-Tetrix lifting steps; `step` selects one of the ten (1..5 inverse, 6..10 forward).
__kernel void star(__global int* b, unsigned W, unsigned H, int step) {
    const int x = get_global_id(0);
    const int r = get_global_id(1);
    const int w = ((int)(W)), h = ((int)(H));
    if (x >= w || r >= h) return;
    const u64 n = ((u64)(W)) * H;
    __global int* P0 = b;
    __global int* P1 = b + n;
    __global int* P2 = b + 2 * n;
    __global int* P3 = b + 3 * n;
    const int L = x > 0 ? x - 1 : 0, R = x + 1 < w ? x + 1 : w - 1;
    const int U = r > 0 ? r - 1 : 0, D = r + 1 < h ? r + 1 : h - 1;
#define AT(p, rr, xx) p[((u64)(rr)) * W + (xx)]
    switch (step) {
        case 1: AT(P0, r, x) -= (AT(P2, U, x) + AT(P2, U, R) + AT(P2, r, x) + AT(P2, r, R)) >> 3; break;
        case 2: AT(P2, r, x) = AT(P2, r, x) - ((AT(P1, r, x) + AT(P1, D, x) + AT(P3, r, L) + AT(P3, r, x)) >> 3) +
                               ((AT(P0, r, L) + AT(P0, r, x) + AT(P0, D, L) + AT(P0, D, x)) >> 2); break;
        case 3: AT(P0, r, x) -= (AT(P1, r, x) + AT(P1, r, R) + AT(P3, U, x) + AT(P3, r, x)) >> 3; break;
        case 4: AT(P1, r, x) += (AT(P0, r, L) + AT(P0, r, x) + AT(P2, U, x) + AT(P2, r, x)) >> 2; break;
        case 5: AT(P3, r, x) += (AT(P0, r, x) + AT(P0, D, x) + AT(P2, r, x) + AT(P2, r, R)) >> 2; break;
        case 6: AT(P3, r, x) -= (AT(P0, r, x) + AT(P0, D, x) + AT(P2, r, x) + AT(P2, r, R)) >> 2; break;
        case 7: AT(P1, r, x) -= (AT(P0, r, L) + AT(P0, r, x) + AT(P2, U, x) + AT(P2, r, x)) >> 2; break;
        case 8: AT(P0, r, x) += (AT(P1, r, x) + AT(P1, r, R) + AT(P3, U, x) + AT(P3, r, x)) >> 3; break;
        case 9: AT(P2, r, x) = AT(P2, r, x) + ((AT(P1, r, x) + AT(P1, D, x) + AT(P3, r, L) + AT(P3, r, x)) >> 3) -
                               ((AT(P0, r, L) + AT(P0, r, x) + AT(P0, D, L) + AT(P0, D, x)) >> 2); break;
        case 10: AT(P0, r, x) += (AT(P2, U, x) + AT(P2, U, R) + AT(P2, r, x) + AT(P2, r, R)) >> 3; break;
    }
#undef AT
}

// ---- transfer curve, in double precision exactly as Transfer::to_raw_f / to_code
__kernel void to_raw(__global const int* src, __global float* dst, double black, double gain, double knee, u64 n) {
    const u64 i = get_global_id(0);
    if (i >= n) return;
    const double d = (double)((float)src[i]) + 32768.0 - knee;
    double q;
    if (d < 0.0) {
        q = -d * d;
    } else {
        const double t = d * gain / 65536.0;
        q = t * t;
    }
    dst[i] = (float)(black + (q + 163840.0) / 262144.0);
}

__kernel void to_code(__global const float* src, __global int* dst, double black, double gain, double knee, u64 n) {
    const u64 i = get_global_id(0);
    if (i >= n) return;
    const double q = ((double)src[i] - black) * 262144.0 - 163840.0;
    const double d = q >= 0.0 ? sqrt(q) * 65536.0 / gain : -sqrt(-q);
    const double v = fmin(fmax(round(d + knee), 0.0), 65535.0);
    dst[i] = (int)v - 32768;
}

// ---- separable resampling; taps are (first, count, offset) triples plus weights
__kernel void rs_h(__global const float* src, __global float* dst, __global const int* tap, __global const float* wt, unsigned dw, unsigned rows,
                     unsigned srcStride, u64 srcOff, u64 dstOff) {
    const unsigned X = get_global_id(0);
    const unsigned y = get_global_id(1);
    if (X >= dw || y >= rows) return;
    const int first = tap[3 * X], count = tap[3 * X + 1], off = tap[3 * X + 2];
    __global const float* s = src + srcOff + ((u64)(y)) * srcStride + first;
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) acc = MADD(wt[off + k], s[k], acc);
    dst[dstOff + ((u64)(y)) * dw + X] = acc;
}

__kernel void rs_v(__global const float* src, __global float* dst, __global const int* tap, __global const float* wt, unsigned dw, unsigned rows,
                     u64 srcOff, u64 dstOff) {
    const unsigned X = get_global_id(0);
    const unsigned Y = get_global_id(1);
    if (X >= dw || Y >= rows) return;
    const int first = tap[3 * Y], count = tap[3 * Y + 1], off = tap[3 * Y + 2];
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) acc = MADD(wt[off + k], src[srcOff + ((u64)(first + k)) * dw + X], acc);
    dst[dstOff + ((u64)(Y)) * dw + X] = acc;
}

