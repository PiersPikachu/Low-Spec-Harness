// kernels.cu - the GPU engine's kernels (ported from engine/gpu/gpu_engine.cu and gpu_experts.cu of the C harness).
// Weights stay compressed in VRAM (huffman4 or rans4 blobs, exactly the vault's bytes); every matvec decodes them on the fly.
// Written for both nvcc --ptx (build time) and NVRTC (run time): no includes, plain extern "C" entry points.

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define LMAX 12
#define NBMAX 16

__device__ __forceinline__ u32 ld4(const u8 *p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
__device__ __forceinline__ float bf(u32 sy, u32 e) { return __uint_as_float((((sy & 0x80u) << 8) | ((e & 255u) << 7) | (sy & 0x7Fu)) << 16); }

// ---------------------------------------------------------------- huffman4
// Y[t][r] (+)= bias[r] + sum_j W[r][j] * X[t][j] for t < n (n <= NB); 4 threads per row (one per interleaved exponent stream),
// each weight decoded once for all n vectors
template <int NB> __device__ void mvb_huff(const u8 *es, const u32 *off, const u8 *sm, const u16 *glut, int rows, int cols,
                                           const float *__restrict__ x, float *__restrict__ y, const float *__restrict__ bias, int acc, int n) {
    __shared__ u16 lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += blockDim.x) lut[i] = glut[i];
    __syncthreads();
    int r = blockIdx.x * (blockDim.x >> 2) + (threadIdx.x >> 2), k = threadIdx.x & 3; bool live = r < rows; float s[NB];
#pragma unroll
    for (int t = 0; t < NB; t++) s[t] = 0.f;
    if (live) {
        u32 pos = off[4 * (u64)r + k]; u64 a = 0; int nb = 0; const u8 *smr = sm + (u64)r * cols; u32 nxt = ld4(es + pos); pos += 4;
        for (int j = k; j < cols; j += 4) {
            if (nb < 12) { a |= (u64)nxt << nb; nb += 32; nxt = ld4(es + pos); pos += 4; }
            u32 e = lut[a & 4095]; int l = e >> 8; a >>= l; nb -= l;
            float w = bf(smr[j], e);
#pragma unroll
            for (int t = 0; t < NB; t++) if (t < n) s[t] += w * x[(u64)t * cols + j];
        }
    }
#pragma unroll
    for (int t = 0; t < NB; t++) {
        float v = s[t]; v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
        if (live && k == 0 && t < n) { float o = v + (bias ? bias[r] : 0.f); u64 yi = (u64)t * rows + r; if (acc) y[yi] += o; else y[yi] = o; }
    }
}
extern "C" __global__ void __launch_bounds__(64) mvb_huff1(const u8 *es, const u32 *off, const u8 *sm, const u16 *lut, int rows, int cols, const float *x, float *y, const float *bias, int acc, int n) { mvb_huff<1>(es, off, sm, lut, rows, cols, x, y, bias, acc, n); }
extern "C" __global__ void __launch_bounds__(64) mvb_huff4(const u8 *es, const u32 *off, const u8 *sm, const u16 *lut, int rows, int cols, const float *x, float *y, const float *bias, int acc, int n) { mvb_huff<4>(es, off, sm, lut, rows, cols, x, y, bias, acc, n); }
extern "C" __global__ void __launch_bounds__(64) mvb_huff16(const u8 *es, const u32 *off, const u8 *sm, const u16 *lut, int rows, int cols, const float *x, float *y, const float *bias, int acc, int n) { mvb_huff<NBMAX>(es, off, sm, lut, rows, cols, x, y, bias, acc, n); }

// one embedding row, 4 threads (one per stream)
extern "C" __global__ void embed_huff(const u8 *es, const u32 *off, const u8 *sm, const u16 *lut, int cols, int tok, float *out) {
    int k = threadIdx.x; u32 pos = off[4 * (u64)tok + k]; u64 a = 0; int nb = 0; const u8 *smr = sm + (u64)tok * cols;
    for (int j = k; j < cols; j += 4) {
        while (nb < 12) { a |= (u64)es[pos++] << nb; nb += 8; }
        u32 e = lut[a & 4095]; int l = e >> 8; a >>= l; nb -= l;
        out[j] = bf(smr[j], e);
    }
}

// ---------------------------------------------------------------- rans4: M = 2^12, tab[slot] = sym | (freq-1)<<8 | (slot-cum)<<20
#define RANS_STEP(xs, p, tab, e) { u32 t_ = tab[xs & 4095u]; e = t_ & 255u; xs = (((t_ >> 8) & 4095u) + 1u) * (xs >> 12) + (t_ >> 20); while (xs < (1u << 23)) xs = (xs << 8) | *p++; }
template <int NB> __device__ void mvb_rans(const u8 *cs, const u32 *off, const u8 *sm, const u32 *gtab, int rows, int cols, int R,
                                           const float *__restrict__ x, float *__restrict__ y, const float *__restrict__ bias, int acc, int n) {
    __shared__ u32 tab[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) tab[i] = gtab[i];
    __syncthreads();
    int chain = blockIdx.x * blockDim.x + threadIdx.x, blk = chain >> 2, k = chain & 3, nblk = (rows + R - 1) / R;
    bool live = blk < nblk; int r0 = blk * R, nr = live ? min(R, rows - r0) : 0; const u8 *p = cs; u32 xs = 0;
    if (live) { p = cs + off[4 * (u64)blk + k]; xs = ld4(p); p += 4; }
    for (int i = 0; i < R; i++) {                      // same trip count for every lane: the shuffles below need the whole warp
        float s[NB];
#pragma unroll
        for (int t = 0; t < NB; t++) s[t] = 0.f;
        if (i < nr) {
            const u8 *smr = sm + (u64)(r0 + i) * cols;
            for (int j = k; j < cols; j += 4) {
                u32 e; RANS_STEP(xs, p, tab, e)
                float w = bf(smr[j], e);
#pragma unroll
                for (int t = 0; t < NB; t++) if (t < n) s[t] += w * x[(u64)t * cols + j];
            }
        }
#pragma unroll
        for (int t = 0; t < NB; t++) {
            float v = s[t]; v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
            if (i < nr && k == 0 && t < n) { int r = r0 + i; float o = v + (bias ? bias[r] : 0.f); u64 yi = (u64)t * rows + r; if (acc) y[yi] += o; else y[yi] = o; }
        }
    }
}
extern "C" __global__ void __launch_bounds__(128) mvb_rans1(const u8 *cs, const u32 *off, const u8 *sm, const u32 *tab, int rows, int cols, int R, const float *x, float *y, const float *bias, int acc, int n) { mvb_rans<1>(cs, off, sm, tab, rows, cols, R, x, y, bias, acc, n); }
extern "C" __global__ void __launch_bounds__(128) mvb_rans4(const u8 *cs, const u32 *off, const u8 *sm, const u32 *tab, int rows, int cols, int R, const float *x, float *y, const float *bias, int acc, int n) { mvb_rans<4>(cs, off, sm, tab, rows, cols, R, x, y, bias, acc, n); }
extern "C" __global__ void __launch_bounds__(128) mvb_rans16(const u8 *cs, const u32 *off, const u8 *sm, const u32 *tab, int rows, int cols, int R, const float *x, float *y, const float *bias, int acc, int n) { mvb_rans<NBMAX>(cs, off, sm, tab, rows, cols, R, x, y, bias, acc, n); }

// embedding row `tok`: the 4 chains of its row block decode up to and including that row
extern "C" __global__ void embed_rans(const u8 *cs, const u32 *off, const u8 *sm, const u32 *tab, int cols, int R, int tok, float *out) {
    int k = threadIdx.x, blk = tok / R, i0 = tok - blk * R;
    const u8 *p = cs + off[4 * (u64)blk + k]; u32 xs = ld4(p); p += 4;
    const u8 *smr = sm + (u64)tok * cols;
    for (int i = 0; i <= i0; i++)
        for (int j = k; j < cols; j += 4) { u32 e; RANS_STEP(xs, p, tab, e) if (i == i0) out[j] = bf(smr[j], e); }
}

// ---------------------------------------------------------------- the rest of a decoder layer
__device__ float block_sum(float v, float *red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31; __syncthreads(); if (l == 0) red[w] = v; __syncthreads();
    float t = 0.f; for (int i = 0; i < (blockDim.x >> 5); i++) t += red[i]; return t;
}
__device__ float block_max(float v, float *red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    int w = threadIdx.x >> 5, l = threadIdx.x & 31; __syncthreads(); if (l == 0) red[w] = v; __syncthreads();
    float t = -1e30f; for (int i = 0; i < (blockDim.x >> 5); i++) t = fmaxf(t, red[i]); return t;
}

// RMSNorm of the blockIdx.x-th group of n values; out may alias x
extern "C" __global__ void rms(const float *x, const float *w, float *out, int n, float eps) {
    __shared__ float red[32]; const float *xx = x + (u64)blockIdx.x * n; float *oo = out + (u64)blockIdx.x * n; float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xx[i] * xx[i];
    ss = block_sum(ss, red); float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) oo[i] = w[i] * (xx[i] * inv);
}

// rotate q in place; rotate k into the cache slot; copy v into the cache slot (blocks 0..nh-1: q heads, nh..: kv heads)
extern "C" __global__ void rope(float *Q, const float *K, const float *V, float *kc, float *vc, const float *cs, const float *sn, int nh, int hd) {
    int h = blockIdx.x, half = hd >> 1;
    if (h < nh) { float *q = Q + h * hd; for (int i = threadIdx.x; i < half; i += blockDim.x) { float a = q[i], b = q[i + half], c = cs[i], s = sn[i]; q[i] = a * c - b * s; q[i + half] = b * c + a * s; } }
    else { int kh = h - nh; const float *k = K + kh * hd; float *ko = kc + kh * hd, *vo = vc + kh * hd; const float *v = V + kh * hd;
        for (int i = threadIdx.x; i < half; i += blockDim.x) { float a = k[i], b = k[i + half], c = cs[i], s = sn[i]; ko[i] = a * c - b * s; ko[i + half] = b * c + a * s; }
        for (int d = threadIdx.x; d < hd; d += blockDim.x) vo[d] = v[d]; }
}

// one block per query head: causal softmax(q.k/sqrt(hd)) @ v over npos cached positions; scores in global scratch (any length)
extern "C" __global__ void attn(const float *Q, const float *kc, const float *vc, float *ATT, float *scr, int nh, int nkv, int hd, int npos, float scale) {
    __shared__ float red[32];
    int h = blockIdx.x, kvh = h / (nh / nkv), kvd = nkv * hd; const float *q = Q + h * hd; float *sc = scr + (u64)h * npos; float mx = -1e30f;
    for (int t = threadIdx.x; t < npos; t += blockDim.x) { const float *k = kc + (u64)t * kvd + kvh * hd; float d = 0.f; for (int i = 0; i < hd; i++) d += q[i] * k[i]; d *= scale; sc[t] = d; mx = fmaxf(mx, d); }
    mx = block_max(mx, red); float sum = 0.f;
    for (int t = threadIdx.x; t < npos; t += blockDim.x) { float e = expf(sc[t] - mx); sc[t] = e; sum += e; }
    sum = block_sum(sum, red); __syncthreads(); float inv = 1.0f / sum;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) { float a = 0.f; for (int t = 0; t < npos; t++) a += sc[t] * vc[(u64)t * kvd + kvh * hd + d]; ATT[h * hd + d] = a * inv; }
}

extern "C" __global__ void silu_mul(float *g, const float *u, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float a = g[i]; g[i] = a / (1.0f + expf(-a)) * u[i]; } }

// ---------------------------------------------------------------- gpt-oss MXFP4 experts (VRAM tier)
// one warp per output row; each lane reads one 32-bit word (8 FP4 codes) at a time (coalesced); FP4 table in shared memory;
// MXF4 (scales as released) or MXF2 (2-bit packed scales + exception rows); up to 8 token vectors per pass
extern "C" __global__ void mxf4(const u8 *sub, int rows, int cols, const float *__restrict__ X, float *__restrict__ Y, int m) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) { const float F[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f}; lut[threadIdx.x] = F[threadIdx.x]; }
    __syncthreads();
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    int nb = cols / 32, nw = cols / 8; const u8 *S = sub + 12 + (u64)row * nb;
    const u32 *Wd = (const u32 *)(sub + 12 + (u64)rows * nb + (u64)row * nb * 16);
    int mx2 = sub[3] == '2'; u32 desc = 0; const u8 *pk = 0, *ex = 0;
    if (mx2) { u32 pb = *(const u32 *)(sub + 12), boff = *(const u32 *)(sub + 20);
        desc = ((const u32 *)(sub + 24))[row]; pk = sub + 24 + 4 * (u64)rows + (u64)row * pb;
        ex = sub + 24 + 4 * (u64)rows + (u64)rows * pb + (u64)(desc & 0x7fffffffu) * nb;
        Wd = (const u32 *)(sub + boff + (u64)row * nb * 16); }
    for (int t0 = 0; t0 < m; t0 += 8) {
        int tn = m - t0 < 8 ? m - t0 : 8; float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int wi = lane; wi < nw; wi += 32) {
            int bi = wi >> 2; int e8 = !mx2 ? (int)__ldg(S + bi) : (desc >> 31) ? (int)__ldg(ex + bi) : (int)(desc & 255u) + ((__ldg(pk + (bi >> 2)) >> ((bi & 3) * 2)) & 3);
            u32 v = __ldg(Wd + wi); float sc = ldexpf(1.0f, e8 - 127); float w[8];
#pragma unroll
            for (int j = 0; j < 4; j++) { u32 byte = (v >> (8 * j)) & 255u; w[2 * j] = lut[byte & 15u]; w[2 * j + 1] = lut[byte >> 4]; }
            for (int t = 0; t < tn; t++) {
                const float4 *xx = (const float4 *)(X + (u64)(t0 + t) * cols + (u64)wi * 8); float4 a = xx[0], b = xx[1];
                float s = w[0] * a.x; s = fmaf(w[1], a.y, s); s = fmaf(w[2], a.z, s); s = fmaf(w[3], a.w, s);
                s = fmaf(w[4], b.x, s); s = fmaf(w[5], b.y, s); s = fmaf(w[6], b.z, s); s = fmaf(w[7], b.w, s);
                acc[t] = fmaf(sc, s, acc[t]);
            }
        }
        for (int t = 0; t < tn; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
            if (lane == 0) Y[(u64)(t0 + t) * rows + row] = v;
        }
    }
}
// clamped SwiGLU of interleaved gate/up (+ bias)
extern "C" __global__ void act_gptoss(const float *GU, const float *bgu, float *A, int m, int I, float lim, float al) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m * I) return;
    int t = i / I, j = i % I; const float *g = GU + (u64)t * 2 * I;
    float gt = g[2 * j] + bgu[2 * j], up = g[2 * j + 1] + bgu[2 * j + 1];
    gt = fminf(gt, lim); up = fminf(fmaxf(up, -lim), lim);
    A[i] = (up + 1.0f) * (gt / (1.0f + expf(-al * gt)));
}
extern "C" __global__ void add_bias(float *D, const float *b, int m, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m * n) D[i] += b[i % n]; }
