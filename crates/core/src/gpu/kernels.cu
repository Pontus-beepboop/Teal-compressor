// CUDA kernels for the resize pipeline, compiled at run time with NVRTC. Same kernels
// and buffer plan as kernels.metal and kernels.cl. Written to match the CPU path bit for
// bit: compiled with --fmad=false, a*b+c fused only where the CPU build fuses it
// (BC_FMA), and the transfer curve in double precision like Transfer.
// UNTESTED: never run on an NVIDIA GPU.
typedef unsigned long long u64;
#ifdef BC_FMA
#define MADD(a, b, c) fmaf((a), (b), (c))
#else
#define MADD(a, b, c) __fadd_rn(__fmul_rn((a), (b)), (c))
#endif

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---- horizontal 5/3, one level, rows of `stride` starting at `base`
extern "C" __global__ void ih_even(int* buf, int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= lows || y >= rows) return;
    const u64 r = base + u64(y) * stride;
    const int* hi = buf + r + lows;
    tmp[r + 2 * k] = buf[r + k] - ((hi[clampi(k - 1, 0, highs - 1)] + hi[clampi(k, 0, highs - 1)] + 2) >> 2);
}

extern "C" __global__ void ih_odd(int* buf, int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= highs || y >= rows) return;
    const u64 r = base + u64(y) * stride;
    const int e = tmp[r + 2 * k];
    const int right = 2 * k + 2 < len ? tmp[r + 2 * k + 2] : e;
    tmp[r + 2 * k + 1] = buf[r + lows + k] + ((e + right) >> 1);
}

extern "C" __global__ void fh_high(int* buf, int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= highs || y >= rows) return;
    const u64 r = base + u64(y) * stride;
    const int e = buf[r + 2 * k];
    const int right = 2 * k + 2 < len ? buf[r + 2 * k + 2] : e;
    tmp[r + lows + k] = buf[r + 2 * k + 1] - ((e + right) >> 1);
}

extern "C" __global__ void fh_low(int* buf, int* tmp, unsigned base, unsigned stride, int len, unsigned rows) {
    const int lows = len - len / 2, highs = len / 2;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= lows || y >= rows) return;
    const u64 r = base + u64(y) * stride;
    const int* hi = tmp + r + lows;
    tmp[r + k] = buf[r + 2 * k] + ((hi[clampi(k - 1, 0, highs - 1)] + hi[clampi(k, 0, highs - 1)] + 2) >> 2);
}

extern "C" __global__ void copy_rows(int* dst, const int* src, unsigned base, unsigned stride, int len, unsigned rows) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= len || y >= rows) return;
    const u64 i = base + u64(y) * stride + x;
    dst[i] = src[i];
}

extern "C" __global__ void copy_span(int* dst, const int* src, u64 d, u64 s, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[d + i] = src[s + i];
}

// ---- vertical 5/3 (lo, hi, img are element offsets of one component)
__device__ __forceinline__ int evenRow(const int* lo, const int* hi, unsigned w, int highRows, int j, unsigned x) {
    return lo[u64(j) * w + x] -
           ((hi[u64(clampi(j - 1, 0, highRows - 1)) * w + x] + hi[u64(clampi(j, 0, highRows - 1)) * w + x] + 2) >> 2);
}

extern "C" __global__ void iv(const int* coef, int* img, u64 lo, u64 hi, u64 im, unsigned w, int ch, int copy) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int lowRows = (ch + 1) / 2, highRows = ch / 2;
    if (x >= w || k >= lowRows) return;
    int* dst = img + im;
    if (copy) {
        dst[u64(2 * k) * w + x] = coef[lo + u64(k) * w + x];
        if (2 * k + 1 < ch) dst[u64(2 * k + 1) * w + x] = coef[hi + u64(k) * w + x];
        return;
    }
    const int e = evenRow(coef + lo, coef + hi, w, highRows, k, x);
    dst[u64(2 * k) * w + x] = e;
    if (2 * k + 1 < ch) {
        const int next = 2 * k + 2 < ch ? evenRow(coef + lo, coef + hi, w, highRows, k + 1, x) : e;
        dst[u64(2 * k + 1) * w + x] = coef[hi + u64(k) * w + x] + ((e + next) >> 1);
    }
}

extern "C" __global__ void fv_high(const int* img, int* coef, u64 hi, u64 im, unsigned w, int ch) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || k >= ch / 2) return;
    const int* src = img + im;
    const int e = src[u64(2 * k) * w + x];
    const int n = 2 * k + 2 < ch ? src[u64(2 * k + 2) * w + x] : e;
    coef[hi + u64(k) * w + x] = src[u64(2 * k + 1) * w + x] - ((e + n) >> 1);
}

extern "C" __global__ void fv_low(const int* img, int* coef, u64 lo, u64 hi, u64 im, unsigned w, int ch, int copy) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int lowRows = (ch + 1) / 2, highRows = ch / 2;
    if (x >= w || k >= lowRows) return;
    if (copy) {
        coef[lo + u64(k) * w + x] = img[im + u64(2 * k) * w + x];
        if (2 * k + 1 < ch) coef[hi + u64(k) * w + x] = img[im + u64(2 * k + 1) * w + x];
        return;
    }
    const int a = coef[hi + u64(clampi(k - 1, 0, highRows - 1)) * w + x];
    const int b = coef[hi + u64(clampi(k, 0, highRows - 1)) * w + x];
    coef[lo + u64(k) * w + x] = img[im + u64(2 * k) * w + x] + ((a + b + 2) >> 2);
}

// ---- Star-Tetrix lifting steps; `step` selects one of the ten (1..5 inverse, 6..10 forward).
extern "C" __global__ void star(int* b, unsigned W, unsigned H, int step) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = int(W), h = int(H);
    if (x >= w || r >= h) return;
    const u64 n = u64(W) * H;
    int* P0 = b;
    int* P1 = b + n;
    int* P2 = b + 2 * n;
    int* P3 = b + 3 * n;
    const int L = x > 0 ? x - 1 : 0, R = x + 1 < w ? x + 1 : w - 1;
    const int U = r > 0 ? r - 1 : 0, D = r + 1 < h ? r + 1 : h - 1;
#define AT(p, rr, xx) p[u64(rr) * W + (xx)]
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
extern "C" __global__ void to_raw(const int* src, float* dst, double black, double gain, double knee, u64 n) {
    const u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double d = double(float(src[i])) + 32768.0 - knee;
    double q;
    if (d < 0.0) {
        q = -d * d;
    } else {
        const double t = d * gain / 65536.0;
        q = t * t;
    }
    dst[i] = float(black + (q + 163840.0) / 262144.0);
}

extern "C" __global__ void to_code(const float* src, int* dst, double black, double gain, double knee, u64 n) {
    const u64 i = u64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double q = (double(src[i]) - black) * 262144.0 - 163840.0;
    const double d = q >= 0.0 ? sqrt(q) * 65536.0 / gain : -sqrt(-q);
    const double v = fmin(fmax(round(d + knee), 0.0), 65535.0);
    dst[i] = int(v) - 32768;
}

// ---- separable resampling; taps are (first, count, offset) triples plus weights
extern "C" __global__ void rs_h(const float* src, float* dst, const int* tap, const float* wt, unsigned dw, unsigned rows,
                     unsigned srcStride, u64 srcOff, u64 dstOff) {
    const unsigned X = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= dw || y >= rows) return;
    const int first = tap[3 * X], count = tap[3 * X + 1], off = tap[3 * X + 2];
    const float* s = src + srcOff + u64(y) * srcStride + first;
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) acc = MADD(wt[off + k], s[k], acc);
    dst[dstOff + u64(y) * dw + X] = acc;
}

extern "C" __global__ void rs_v(const float* src, float* dst, const int* tap, const float* wt, unsigned dw, unsigned rows,
                     u64 srcOff, u64 dstOff) {
    const unsigned X = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= dw || Y >= rows) return;
    const int first = tap[3 * Y], count = tap[3 * Y + 1], off = tap[3 * Y + 2];
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) acc = MADD(wt[off + k], src[srcOff + u64(first + k) * dw + X], acc);
    dst[dstOff + u64(Y) * dw + X] = acc;
}

