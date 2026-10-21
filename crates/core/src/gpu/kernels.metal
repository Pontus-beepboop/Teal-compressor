// Metal kernels for the resize pipeline (see gpu/metal.rs). Integer stages match the
// CPU code in frame.rs bit for bit.
#include <metal_stdlib>
using namespace metal;

struct Rows {        // horizontal pass over `rows` rows of `stride` elements
    uint base;       // element offset of the first row
    uint stride;
    uint len;        // active length of this level
    uint rows;
};

struct Vert {        // vertical pass of one component
    uint lo, hi, img; // element offsets: low rows, high rows, image rows
    uint w, h;       // component size (h = image rows)
};

struct Plane { uint w, h; };

struct Curve { float black, gain, knee; uint n; };

struct Taps { uint dw, rows, srcStride, srcOff, dstOff; };

kernel void ih_even(device int* buf [[buffer(0)]], device int* tmp [[buffer(1)]], constant Rows& p [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    const int lows = int(p.len - p.len / 2), highs = int(p.len / 2);
    if (int(g.x) >= lows || g.y >= p.rows) return;
    const uint r = p.base + g.y * p.stride;
    const int k = int(g.x);
    device const int* hi = buf + r + uint(lows);
    const int a = hi[clamp(k - 1, 0, highs - 1)], b = hi[clamp(k, 0, highs - 1)];
    tmp[r + uint(2 * k)] = buf[r + uint(k)] - ((a + b + 2) >> 2);
}

kernel void ih_odd(device int* buf [[buffer(0)]], device int* tmp [[buffer(1)]], constant Rows& p [[buffer(2)]],
                   uint2 g [[thread_position_in_grid]]) {
    const int lows = int(p.len - p.len / 2), highs = int(p.len / 2), len = int(p.len);
    if (int(g.x) >= highs || g.y >= p.rows) return;
    const uint r = p.base + g.y * p.stride;
    const int k = int(g.x);
    const int e = tmp[r + uint(2 * k)];
    const int right = 2 * k + 2 < len ? tmp[r + uint(2 * k + 2)] : e;
    tmp[r + uint(2 * k + 1)] = buf[r + uint(lows + k)] + ((e + right) >> 1);
}

kernel void fh_high(device int* buf [[buffer(0)]], device int* tmp [[buffer(1)]], constant Rows& p [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    const int lows = int(p.len - p.len / 2), highs = int(p.len / 2), len = int(p.len);
    if (int(g.x) >= highs || g.y >= p.rows) return;
    const uint r = p.base + g.y * p.stride;
    const int k = int(g.x);
    const int e = buf[r + uint(2 * k)];
    const int right = 2 * k + 2 < len ? buf[r + uint(2 * k + 2)] : e;
    tmp[r + uint(lows + k)] = buf[r + uint(2 * k + 1)] - ((e + right) >> 1);
}

kernel void fh_low(device int* buf [[buffer(0)]], device int* tmp [[buffer(1)]], constant Rows& p [[buffer(2)]],
                   uint2 g [[thread_position_in_grid]]) {
    const int lows = int(p.len - p.len / 2), highs = int(p.len / 2);
    if (int(g.x) >= lows || g.y >= p.rows) return;
    const uint r = p.base + g.y * p.stride;
    const int k = int(g.x);
    device const int* hi = tmp + r + uint(lows);
    const int a = hi[clamp(k - 1, 0, highs - 1)], b = hi[clamp(k, 0, highs - 1)];
    tmp[r + uint(k)] = buf[r + uint(2 * k)] + ((a + b + 2) >> 2);
}

kernel void copy_rows(device int* dst [[buffer(0)]], device const int* src [[buffer(1)]], constant Rows& p [[buffer(2)]],
                      uint2 g [[thread_position_in_grid]]) {
    if (g.x >= p.len || g.y >= p.rows) return;
    const uint i = p.base + g.y * p.stride + g.x;
    dst[i] = src[i];
}

// Copies one row: dst[base .. base+len) = src[stride .. stride+len) (stride used as source offset).
kernel void copy_span(device int* dst [[buffer(0)]], device const int* src [[buffer(1)]], constant Rows& p [[buffer(2)]],
                      uint g [[thread_position_in_grid]]) {
    if (g >= p.len) return;
    dst[p.base + g] = src[p.stride + g];
}

inline int evenRow(device const int* lo, device const int* hi, uint w, int highRows, int j, uint x) {
    const int a = hi[uint(clamp(j - 1, 0, highRows - 1)) * w + x];
    const int b = hi[uint(clamp(j, 0, highRows - 1)) * w + x];
    return lo[uint(j) * w + x] - ((a + b + 2) >> 2);
}

kernel void iv(device const int* coef [[buffer(0)]], device int* img [[buffer(1)]], constant Vert& v [[buffer(2)]],
               uint2 g [[thread_position_in_grid]]) {
    const int ch = int(v.h), lowRows = (ch + 1) / 2, highRows = ch / 2;
    if (g.x >= v.w || int(g.y) >= lowRows) return;
    device const int* lo = coef + v.lo;
    device const int* hi = coef + v.hi;
    device int* dst = img + v.img;
    const int k = int(g.y);
    const uint x = g.x, w = v.w;
    const int e = evenRow(lo, hi, w, highRows, k, x);
    dst[uint(2 * k) * w + x] = e;
    if (2 * k + 1 < ch) {
        const int next = 2 * k + 2 < ch ? evenRow(lo, hi, w, highRows, k + 1, x) : e;
        dst[uint(2 * k + 1) * w + x] = hi[uint(k) * w + x] + ((e + next) >> 1);
    }
}

kernel void iv_copy(device const int* coef [[buffer(0)]], device int* img [[buffer(1)]], constant Vert& v [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    const int ch = int(v.h), lowRows = (ch + 1) / 2;
    if (g.x >= v.w || int(g.y) >= lowRows) return;
    const int k = int(g.y);
    img[v.img + uint(2 * k) * v.w + g.x] = coef[v.lo + uint(k) * v.w + g.x];
    if (2 * k + 1 < ch) img[v.img + uint(2 * k + 1) * v.w + g.x] = coef[v.hi + uint(k) * v.w + g.x];
}

kernel void fv_high(device const int* img [[buffer(0)]], device int* coef [[buffer(1)]], constant Vert& v [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    const int ch = int(v.h), highRows = ch / 2;
    if (g.x >= v.w || int(g.y) >= highRows) return;
    const int k = int(g.y);
    const uint w = v.w, x = g.x;
    device const int* src = img + v.img;
    const int e = src[uint(2 * k) * w + x];
    const int o = src[uint(2 * k + 1) * w + x];
    const int n = 2 * k + 2 < ch ? src[uint(2 * k + 2) * w + x] : e;
    coef[v.hi + uint(k) * w + x] = o - ((e + n) >> 1);
}

kernel void fv_low(device const int* img [[buffer(0)]], device int* coef [[buffer(1)]], constant Vert& v [[buffer(2)]],
                   uint2 g [[thread_position_in_grid]]) {
    const int ch = int(v.h), lowRows = (ch + 1) / 2, highRows = ch / 2;
    if (g.x >= v.w || int(g.y) >= lowRows) return;
    const int k = int(g.y);
    const uint w = v.w, x = g.x;
    device const int* hi = coef + v.hi;
    const int a = hi[uint(clamp(k - 1, 0, highRows - 1)) * w + x];
    const int b = hi[uint(clamp(k, 0, highRows - 1)) * w + x];
    coef[v.lo + uint(k) * w + x] = img[v.img + uint(2 * k) * w + x] + ((a + b + 2) >> 2);
}

kernel void fv_copy(device const int* img [[buffer(0)]], device int* coef [[buffer(1)]], constant Vert& v [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    const int ch = int(v.h), lowRows = (ch + 1) / 2;
    if (g.x >= v.w || int(g.y) >= lowRows) return;
    const int k = int(g.y);
    coef[v.lo + uint(k) * v.w + g.x] = img[v.img + uint(2 * k) * v.w + g.x];
    if (2 * k + 1 < ch) coef[v.hi + uint(k) * v.w + g.x] = img[v.img + uint(2 * k + 1) * v.w + g.x];
}

// Star-Tetrix lifting. Planes are consecutive in one buffer: p0, p1, p2, p3.
#define ST_PROLOGUE                                                         \
    if (g.x >= pl.w || g.y >= pl.h) return;                                 \
    const int w = int(pl.w), h = int(pl.h), x = int(g.x), r = int(g.y);     \
    const uint n = pl.w * pl.h;                                             \
    device int* P0 = b; device int* P1 = b + n; device int* P2 = b + 2 * n; device int* P3 = b + 3 * n; \
    const int L = x > 0 ? x - 1 : 0, R = x + 1 < w ? x + 1 : w - 1;         \
    const int U = r > 0 ? r - 1 : 0, D = r + 1 < h ? r + 1 : h - 1;         \
    (void)L; (void)R; (void)U; (void)D;
#define AT(p, rr, xx) p[uint(rr) * pl.w + uint(xx)]

kernel void ist1(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int s = AT(P2, U, x) + AT(P2, U, R) + AT(P2, r, x) + AT(P2, r, R);
    AT(P0, r, x) -= s >> 3;
}
kernel void ist2(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int chroma = AT(P1, r, x) + AT(P1, D, x) + AT(P3, r, L) + AT(P3, r, x);
    const int greens = AT(P0, r, L) + AT(P0, r, x) + AT(P0, D, L) + AT(P0, D, x);
    AT(P2, r, x) = AT(P2, r, x) - (chroma >> 3) + (greens >> 2);
}
kernel void ist3(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int chroma = AT(P1, r, x) + AT(P1, r, R) + AT(P3, U, x) + AT(P3, r, x);
    AT(P0, r, x) -= chroma >> 3;
}
kernel void ist4(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int greens = AT(P0, r, L) + AT(P0, r, x) + AT(P2, U, x) + AT(P2, r, x);
    AT(P1, r, x) += greens >> 2;
}
kernel void ist5(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int greens = AT(P0, r, x) + AT(P0, D, x) + AT(P2, r, x) + AT(P2, r, R);
    AT(P3, r, x) += greens >> 2;
}
kernel void fst1(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int greens = AT(P0, r, x) + AT(P0, D, x) + AT(P2, r, x) + AT(P2, r, R);
    AT(P3, r, x) -= greens >> 2;
}
kernel void fst2(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int greens = AT(P0, r, L) + AT(P0, r, x) + AT(P2, U, x) + AT(P2, r, x);
    AT(P1, r, x) -= greens >> 2;
}
kernel void fst3(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int chroma = AT(P1, r, x) + AT(P1, r, R) + AT(P3, U, x) + AT(P3, r, x);
    AT(P0, r, x) += chroma >> 3;
}
kernel void fst4(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int chroma = AT(P1, r, x) + AT(P1, D, x) + AT(P3, r, L) + AT(P3, r, x);
    const int greens = AT(P0, r, L) + AT(P0, r, x) + AT(P0, D, L) + AT(P0, D, x);
    AT(P2, r, x) = AT(P2, r, x) + (chroma >> 3) - (greens >> 2);
}
kernel void fst5(device int* b [[buffer(0)]], constant Plane& pl [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    ST_PROLOGUE
    const int s = AT(P2, U, x) + AT(P2, U, R) + AT(P2, r, x) + AT(P2, r, R);
    AT(P0, r, x) += s >> 3;
}

kernel void to_raw(device const int* src [[buffer(0)]], device float* dst [[buffer(1)]], constant Curve& c [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= c.n) return;
    const float d = float(src[i]) + 32768.0f - c.knee;
    float q;
    if (d < 0.0f) q = -d * d;
    else { const float t = d * c.gain / 65536.0f; q = t * t; }
    dst[i] = c.black + (q + 163840.0f) / 262144.0f;
}

kernel void to_code(device const float* src [[buffer(0)]], device int* dst [[buffer(1)]], constant Curve& c [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= c.n) return;
    const float q = (src[i] - c.black) * 262144.0f - 163840.0f;
    const float d = q >= 0.0f ? sqrt(q) * 65536.0f / c.gain : -sqrt(-q);
    const float v = clamp(round(d + c.knee), 0.0f, 65535.0f);
    dst[i] = int(v) - 32768;
}

// Separable resampling. Taps: per output index (first, count, offset) and weights.
kernel void rs_h(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                 device const int* tap [[buffer(2)]], device const float* wt [[buffer(3)]], constant Taps& t [[buffer(4)]],
                 uint2 g [[thread_position_in_grid]]) {
    if (g.x >= t.dw || g.y >= t.rows) return;
    const int first = tap[3 * g.x], count = tap[3 * g.x + 1], off = tap[3 * g.x + 2];
    device const float* s = src + t.srcOff + g.y * t.srcStride + uint(first);
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) acc += wt[off + k] * s[k];
    dst[t.dstOff + g.y * t.dw + g.x] = acc;
}

kernel void rs_v(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                 device const int* tap [[buffer(2)]], device const float* wt [[buffer(3)]], constant Taps& t [[buffer(4)]],
                 uint2 g [[thread_position_in_grid]]) {
    if (g.x >= t.dw || g.y >= t.rows) return;
    const int first = tap[3 * g.y], count = tap[3 * g.y + 1], off = tap[3 * g.y + 2];
    float acc = 0.0f;
    for (int k = 0; k < count; ++k) acc += wt[off + k] * src[t.srcOff + uint(first + k) * t.dw + g.x];
    dst[t.dstOff + g.y * t.dw + g.x] = acc;
}
