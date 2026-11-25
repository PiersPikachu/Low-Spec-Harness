// gpu_engine.cu - G2: whole-model forward pass on the GPU, weights stay Huffman4-compressed in VRAM (mode GPU-RESIDENT).
// Reads the same vault as nvme_engine_u.exe (vault.idx v3/v4 + vault.bin + resident.bin) and speaks the same serve protocol subset:
//   READY / "LOGITS n\n<tokens>" -> "L <vocab floats>" + "END" / QUIT ; also: gpu_engine bench <vault> [--tokens N] [--ctx N]
// Dense decoder family only (same support matrix as the C engine). KV cache is fp32.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <io.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define LMAX 12

struct Cfg { int hid = 0, inter = 0, nl = 0, nh = 0, nkv = 0, hd = 0, vocab = 0, tie = 0, eos = -1, eos2 = -1; double eps = 0, theta = 0;
    int codec = 0, lres = 0, r_ln_in = -1, r_ln_post = -1, r_qn = -1, r_kn = -1, r_bq = -1, r_bk = -1, r_bv = -1, r_bo = -1, r_bg = -1, r_bu = -1, r_bd = -1, rl3 = 0, supported = 0;
    double rfactor = 0, rlow = 0, rhigh = 0, rorig = 0; };
struct Str { uint64_t off, size, padded; };
struct Mat { const uint8_t *es; const uint32_t *off; const uint8_t *sm; const uint16_t *lut; int rows, cols; const uint32_t *tab; int R; };   // tab != null -> rans4 blob

// ---------------------------------------------------------------- kernels
__device__ __forceinline__ uint32_t ld4(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

// y[r] (+)= bias[r] + sum_j W[r][j] * x[j]   -- W is a v3 huffman4 blob; 4 threads per row (one per interleaved exponent stream)
template <int TPB> __global__ void __launch_bounds__(TPB) mv_k(Mat m, const float *__restrict__ x, float *__restrict__ y, const float *__restrict__ bias, int acc) {
    __shared__ uint16_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += TPB) lut[i] = m.lut[i];
    __syncthreads();
    int r = blockIdx.x * (TPB >> 2) + (threadIdx.x >> 2), k = threadIdx.x & 3; bool live = r < m.rows; float s = 0.f; int cols = m.cols;
    if (live) {
        uint32_t pos = m.off[4 * (size_t)r + k]; uint64_t a = 0; int nb = 0; const uint8_t *smr = m.sm + (size_t)r * cols; uint32_t nxt = ld4(m.es + pos); pos += 4;
#pragma unroll 4
        for (int j = k; j < cols; j += 4) {
            if (nb < 12) { a |= (uint64_t)nxt << nb; nb += 32; nxt = ld4(m.es + pos); pos += 4; }
            uint32_t e = lut[a & 4095]; int l = e >> 8; a >>= l; nb -= l;
            uint32_t sy = smr[j]; uint32_t w = ((sy & 0x80u) << 8) | ((e & 255u) << 7) | (sy & 0x7Fu);
            s += __uint_as_float(w << 16) * x[j];
        }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (live && k == 0) { float v = s + (bias ? bias[r] : 0.f); if (acc) y[r] += v; else y[r] = v; }
}

// out[0..cols) = row `tok` of the matrix, decoded to fp32 (embedding lookup); 4 threads, one per stream
__global__ void embed_k(Mat m, int tok, float *out) {
    int k = threadIdx.x, cols = m.cols; uint32_t pos = m.off[4 * (size_t)tok + k]; uint64_t a = 0; int nb = 0; const uint8_t *smr = m.sm + (size_t)tok * cols;
    for (int j = k; j < cols; j += 4) {
        while (nb < 12) { a |= (uint64_t)m.es[pos++] << nb; nb += 8; }
        uint32_t e = m.lut[a & 4095]; int l = e >> 8; a >>= l; nb -= l;
        uint32_t sy = smr[j]; uint32_t w = ((sy & 0x80u) << 8) | ((e & 255u) << 7) | (sy & 0x7Fu);
        out[j] = __uint_as_float(w << 16);
    }
}

// ---------------- rans4 (cube-rANS): exponent rANS, M = 2^12, tab[slot] = sym | (freq-1)<<8 | (slot-cum)<<20 ----------------
#define RANS_STEP(xs, p, tab, e) { uint32_t t_ = tab[xs & 4095u]; e = t_ & 255u; xs = (((t_ >> 8) & 4095u) + 1u) * (xs >> 12) + (t_ >> 20); while (xs < (1u << 23)) xs = (xs << 8) | *p++; }
__global__ void __launch_bounds__(128) mv_rans_k(Mat m, const float *__restrict__ x, float *__restrict__ y, const float *__restrict__ bias, int acc) {
    __shared__ uint32_t tab[4096];
    for (int i = threadIdx.x; i < 4096; i += 128) tab[i] = m.tab[i];
    __syncthreads();
    int chain = blockIdx.x * 128 + threadIdx.x, blk = chain >> 2, k = chain & 3, R = m.R, cols = m.cols, nblk = (m.rows + R - 1) / R;
    bool live = blk < nblk; int r0 = blk * R, nr = live ? min(R, m.rows - r0) : 0; const uint8_t *p = m.es; uint32_t xs = 0;
    if (live) { p = m.es + m.off[4 * (size_t)blk + k]; xs = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); p += 4; }
    for (int i = 0; i < R; i++) {                      // same trip count for every lane: the shuffles below need the whole warp
        float s = 0.f;
        if (i < nr) {
            const uint8_t *smr = m.sm + (size_t)(r0 + i) * cols;
            for (int j = k; j < cols; j += 4) {
                uint32_t e; RANS_STEP(xs, p, tab, e)
                uint32_t sy = smr[j], w = ((sy & 0x80u) << 8) | (e << 7) | (sy & 0x7Fu); s += __uint_as_float(w << 16) * x[j];
            }
        }
        s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
        if (i < nr && k == 0) { int r = r0 + i; float v = s + (bias ? bias[r] : 0.f); if (acc) y[r] += v; else y[r] = v; }
    }
}
// embedding row `tok`: the 4 chains of its row block decode up to and including that row
__global__ void embed_rans_k(Mat m, int tok, float *out) {
    int k = threadIdx.x, R = m.R, cols = m.cols, blk = tok / R, i0 = tok - blk * R; const uint32_t *tab = m.tab;
    const uint8_t *p = m.es + m.off[4 * (size_t)blk + k]; uint32_t xs = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); p += 4;
    const uint8_t *smr = m.sm + (size_t)tok * cols;
    for (int i = 0; i <= i0; i++)
        for (int j = k; j < cols; j += 4) { uint32_t e; RANS_STEP(xs, p, tab, e) if (i == i0) { uint32_t sy = smr[j], w = ((sy & 0x80u) << 8) | (e << 7) | (sy & 0x7Fu); out[j] = __uint_as_float(w << 16); } }
}
static int rans_tab_host(const uint8_t *fq, uint32_t *tab) {
    uint32_t cum = 0;
    for (int s = 0; s < 256; s++) { uint16_t f; memcpy(&f, fq + 2 * s, 2); if (cum + f > 4096) return 0; for (uint32_t k = 0; k < f; k++) tab[cum + k] = (uint32_t)s | ((uint32_t)(f - 1) << 8) | (k << 20); cum += f; }
    return cum == 4096;
}

// ---------------- batched matvec: Y[t][r] (+)= bias[r] + sum_j W[r][j] * X[t][j] for t < n (n <= NB); each weight decoded once ----------------
#define NBMAX 16
template <int TPB, int NB> __global__ void __launch_bounds__(TPB) mvb_k(Mat m, const float *__restrict__ x, float *__restrict__ y, const float *__restrict__ bias, int acc, int n) {
    __shared__ uint16_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += TPB) lut[i] = m.lut[i];
    __syncthreads();
    int r = blockIdx.x * (TPB >> 2) + (threadIdx.x >> 2), k = threadIdx.x & 3; bool live = r < m.rows; int cols = m.cols; float s[NB];
#pragma unroll
    for (int t = 0; t < NB; t++) s[t] = 0.f;
    if (live) {
        uint32_t pos = m.off[4 * (size_t)r + k]; uint64_t a = 0; int nb = 0; const uint8_t *smr = m.sm + (size_t)r * cols; uint32_t nxt = ld4(m.es + pos); pos += 4;
        for (int j = k; j < cols; j += 4) {
            if (nb < 12) { a |= (uint64_t)nxt << nb; nb += 32; nxt = ld4(m.es + pos); pos += 4; }
            uint32_t e = lut[a & 4095]; int l = e >> 8; a >>= l; nb -= l;
            uint32_t sy = smr[j]; float w = __uint_as_float((((sy & 0x80u) << 8) | ((e & 255u) << 7) | (sy & 0x7Fu)) << 16);
#pragma unroll
            for (int t = 0; t < NB; t++) if (t < n) s[t] += w * x[(size_t)t * cols + j];
        }
    }
#pragma unroll
    for (int t = 0; t < NB; t++) {
        float v = s[t]; v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
        if (live && k == 0 && t < n) { float o = v + (bias ? bias[r] : 0.f); size_t yi = (size_t)t * m.rows + r; if (acc) y[yi] += o; else y[yi] = o; }
    }
}
template <int NB> __global__ void __launch_bounds__(128) mvb_rans_k(Mat m, const float *__restrict__ x, float *__restrict__ y, const float *__restrict__ bias, int acc, int n) {
    __shared__ uint32_t tab[4096];
    for (int i = threadIdx.x; i < 4096; i += 128) tab[i] = m.tab[i];
    __syncthreads();
    int chain = blockIdx.x * 128 + threadIdx.x, blk = chain >> 2, k = chain & 3, R = m.R, cols = m.cols, nblk = (m.rows + R - 1) / R;
    bool live = blk < nblk; int r0 = blk * R, nr = live ? min(R, m.rows - r0) : 0; const uint8_t *p = m.es; uint32_t xs = 0;
    if (live) { p = m.es + m.off[4 * (size_t)blk + k]; xs = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); p += 4; }
    for (int i = 0; i < R; i++) {
        float s[NB];
#pragma unroll
        for (int t = 0; t < NB; t++) s[t] = 0.f;
        if (i < nr) {
            const uint8_t *smr = m.sm + (size_t)(r0 + i) * cols;
            for (int j = k; j < cols; j += 4) {
                uint32_t e; RANS_STEP(xs, p, tab, e)
                uint32_t sy = smr[j]; float w = __uint_as_float((((sy & 0x80u) << 8) | (e << 7) | (sy & 0x7Fu)) << 16);
#pragma unroll
                for (int t = 0; t < NB; t++) if (t < n) s[t] += w * x[(size_t)t * cols + j];
            }
        }
#pragma unroll
        for (int t = 0; t < NB; t++) {
            float v = s[t]; v += __shfl_xor_sync(0xffffffffu, v, 1); v += __shfl_xor_sync(0xffffffffu, v, 2);
            if (i < nr && k == 0 && t < n) { int r = r0 + i; float o = v + (bias ? bias[r] : 0.f); size_t yi = (size_t)t * m.rows + r; if (acc) y[yi] += o; else y[yi] = o; }
        }
    }
}

__device__ float block_sum(float v, float *red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    int w = threadIdx.x >> 5, l = threadIdx.x & 31; __syncthreads(); if (l == 0) red[w] = v; __syncthreads();
    float t = 0.f; for (int i = 0; i < (blockDim.x >> 5); i++) t += red[i]; return t;
}
__device__ float block_max(float v, float *red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    int w = threadIdx.x >> 5, l = threadIdx.x & 31; __syncthreads(); if (l == 0) red[w] = v; __syncthreads();
    float t = -INFINITY; for (int i = 0; i < (blockDim.x >> 5); i++) t = fmaxf(t, red[i]); return t;
}

// RMSNorm over blockIdx.x-th group of n values; out may alias x
__global__ void rms_k(const float *x, const float *w, float *out, int n, float eps) {
    __shared__ float red[32]; const float *xx = x + (size_t)blockIdx.x * n; float *oo = out + (size_t)blockIdx.x * n; float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xx[i] * xx[i];
    ss = block_sum(ss, red); float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) oo[i] = w[i] * (xx[i] * inv);
}

// rotate q in place; rotate k into the cache slot; copy v into the cache slot
__global__ void rope_k(float *Q, const float *K, const float *V, float *kc, float *vc, const float *cs, const float *sn, int nh, int hd) {
    int h = blockIdx.x, half = hd >> 1;
    if (h < nh) { float *q = Q + h * hd; for (int i = threadIdx.x; i < half; i += blockDim.x) { float a = q[i], b = q[i + half], c = cs[i], s = sn[i]; q[i] = a * c - b * s; q[i + half] = b * c + a * s; } }
    else { int kh = h - nh; const float *k = K + kh * hd; float *ko = kc + kh * hd, *vo = vc + kh * hd; const float *v = V + kh * hd;
        for (int i = threadIdx.x; i < half; i += blockDim.x) { float a = k[i], b = k[i + half], c = cs[i], s = sn[i]; ko[i] = a * c - b * s; ko[i + half] = b * c + a * s; }
        for (int d = threadIdx.x; d < hd; d += blockDim.x) vo[d] = v[d]; }
}

// one block per query head: causal softmax(q.k/sqrt(hd)) @ v over npos cached positions
__global__ void attn_k(const float *Q, const float *kc, const float *vc, float *ATT, int nh, int nkv, int hd, int npos, float scale) {
    extern __shared__ float sc[]; __shared__ float red[32];
    int h = blockIdx.x, kvh = h / (nh / nkv), kvd = nkv * hd; const float *q = Q + h * hd; float mx = -INFINITY;
    for (int t = threadIdx.x; t < npos; t += blockDim.x) { const float *k = kc + (size_t)t * kvd + kvh * hd; float d = 0.f; for (int i = 0; i < hd; i++) d += q[i] * k[i]; d *= scale; sc[t] = d; mx = fmaxf(mx, d); }
    mx = block_max(mx, red); float sum = 0.f;
    for (int t = threadIdx.x; t < npos; t += blockDim.x) { float e = expf(sc[t] - mx); sc[t] = e; sum += e; }
    sum = block_sum(sum, red); __syncthreads(); float inv = 1.0f / sum;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) { float a = 0.f; for (int t = 0; t < npos; t++) a += sc[t] * vc[(size_t)t * kvd + kvh * hd + d]; ATT[h * hd + d] = a * inv; }
}

__global__ void silu_mul_k(float *g, const float *u, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float a = g[i]; g[i] = a / (1.0f + expf(-a)) * u[i]; } }

// ---------------- rank-1 weight patch: y -= u * (w . x)  (one block) ----------------
__global__ void rank1_k(const float *__restrict__ u, const float *__restrict__ w, const float *__restrict__ x, float *__restrict__ y, int od, int id) {
    __shared__ float red[32]; float d = 0.f;
    for (int j = threadIdx.x; j < id; j += blockDim.x) d += w[j] * x[j];
    d = block_sum(d, red);
    for (int r = threadIdx.x; r < od; r += blockDim.x) y[r] -= d * u[r];
}
struct GPatch { int l, k, od, id; float *u, *w; };
static std::vector<GPatch> G_PATCHES;
static void patch_load(const char *path, const Cfg &c) {
    FILE *f = fopen(path, "rb"); char mg[4]; uint32_t n;
    if (!f || fread(mg, 1, 4, f) != 4 || memcmp(mg, "WPAT", 4) || fread(&n, 4, 1, f) != 1) { fprintf(stderr, "cannot read patch file %s\n", path); exit(1); }
    int q = c.nh * c.hd, kv = c.nkv * c.hd; int od_k[7] = {q, kv, kv, c.hid, c.inter, c.inter, c.hid}, id_k[7] = {c.hid, c.hid, c.hid, q, c.hid, c.hid, c.inter};
    for (uint32_t i = 0; i < n; i++) {
        int32_t h[4]; if (fread(h, 4, 4, f) != 4) { fprintf(stderr, "patch file truncated\n"); exit(1); }
        GPatch p{h[0], h[1], h[2], h[3], nullptr, nullptr};
        if (p.l < 0 || p.l >= c.nl || p.k < 0 || p.k > 6 || p.od != od_k[p.k] || p.id != id_k[p.k]) { fprintf(stderr, "patch does not match this model's shapes\n"); exit(1); }
        std::vector<float> u(p.od), w(p.id);
        if (fread(u.data(), 4, p.od, f) != (size_t)p.od || fread(w.data(), 4, p.id, f) != (size_t)p.id) { fprintf(stderr, "patch file truncated\n"); exit(1); }
        CK(cudaMalloc(&p.u, 4 * (size_t)p.od)); CK(cudaMalloc(&p.w, 4 * (size_t)p.id));
        CK(cudaMemcpy(p.u, u.data(), 4 * (size_t)p.od, cudaMemcpyHostToDevice)); CK(cudaMemcpy(p.w, w.data(), 4 * (size_t)p.id, cudaMemcpyHostToDevice));
        G_PATCHES.push_back(p);
    }
    fclose(f); fprintf(stderr, "gpu engine: applying %u rank-1 weight patches from %s\n", n, path);
}
static void apply_patches(int l, int k, const float *x, float *y) {
    for (auto &p : G_PATCHES) if (p.l == l && p.k == k) rank1_k<<<1, 256>>>(p.u, p.w, x, y, p.od, p.id);
}

// ---------------------------------------------------------------- host: vault loading
static void build_lut(const uint8_t *ln, uint16_t *lut) {
    int cnt[LMAX + 2] = {0}, nxt[LMAX + 2] = {0}, c = 0;
    for (int s = 0; s < 256; s++) if (ln[s]) cnt[ln[s]]++;
    for (int l = 1; l <= LMAX; l++) { c = (c + cnt[l - 1]) << 1; nxt[l] = c; }
    memset(lut, 0, sizeof(uint16_t) << LMAX);
    for (int s = 0; s < 256; s++) { int l = ln[s]; if (!l) continue; int code = nxt[l]++, r = 0; for (int i = 0; i < l; i++) r |= ((code >> i) & 1) << (l - 1 - i);
        for (int k = 0; k < (1 << (LMAX - l)); k++) lut[r | (k << l)] = (uint16_t)(s | (l << 8)); }
}
static bool cfg_kv(Cfg *c, const char *k, double v) {
#define K(n, f) if (!strcmp(k, n)) { c->f = v; return true; }
#define KI(n, f) if (!strcmp(k, n)) { c->f = (int)v; return true; }
    KI("hidden", hid) KI("inter", inter) KI("layers", nl) KI("heads", nh) KI("kv_heads", nkv) KI("head_dim", hd) KI("vocab", vocab) KI("tie", tie) KI("eos", eos) KI("eos2", eos2) K("eps", eps) K("theta", theta)
    KI("lres", lres) KI("r_ln_in", r_ln_in) KI("r_ln_post", r_ln_post) KI("r_qn", r_qn) KI("r_kn", r_kn) KI("r_bq", r_bq) KI("r_bk", r_bk) KI("r_bv", r_bv) KI("r_bo", r_bo)
    KI("r_bg", r_bg) KI("r_bu", r_bu) KI("r_bd", r_bd) KI("rope_llama3", rl3) K("rope_factor", rfactor) K("rope_low", rlow) K("rope_high", rhigh) K("rope_orig", rorig) KI("supported", supported) KI("codec", codec)
    return false;
}
#undef K
#undef KI

struct Eng {
    Cfg c; int ver = 0, ctx = 0, pos = 0, only_layers = -1; std::vector<Str> st; std::vector<Mat> mats[64]; Mat lm, emb; std::vector<void *> dev;
    float *res = 0, *fnrm = 0, *cs = 0, *sn = 0, *H, *XN, *Q, *K, *V, *ATT, *G, *U, *logits, *kc, *vc, *hlog; FILE *dump = 0; size_t vram_weights = 0;
};
static void *dalloc(Eng *E, size_t n) { void *p; cudaError_t e = cudaMalloc(&p, n); if (e != cudaSuccess) { fprintf(stderr, "model does not fit VRAM (needed another %.0f MB): use planner.py to pick a streaming mode\n", n / 1e6); exit(3); } E->dev.push_back(p); return p; }

static int eng_open(Eng *E, const char *vdir, int ctx) {
    char p[600]; snprintf(p, sizeof p, "%s\\vault.idx", vdir); FILE *f = fopen(p, "r"); if (!f) { fprintf(stderr, "cannot open %s\n", p); return -1; }
    char line[800]; Cfg *c = &E->c; E->ctx = ctx;
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "WVAULT", 6)) { E->ver = atoi(line + 7); if (E->ver != 3 && E->ver != 4) { fprintf(stderr, "vault format %d unknown\n", E->ver); return -1; } continue; }
        unsigned long long a, b, d, g; int i; char h[40];
        if (sscanf(line, "stream %d %llu %llu %llu %llu %39s", &i, &a, &b, &d, &g, h) == 6) { if ((int)E->st.size() <= i) E->st.resize(i + 1); E->st[i] = {a, b, d}; continue; }
        char k[64]; double v; if (sscanf(line, "%63s %lf", k, &v) == 2) cfg_kv(c, k, v);
    }
    fclose(f);
    if (E->ver == 3) { c->r_ln_in = 0; c->r_ln_post = c->hid; c->r_qn = 2 * c->hid; c->r_kn = 2 * c->hid + c->hd; c->lres = 2 * c->hid + 2 * c->hd; c->supported = 1; }
    if (!c->supported || c->r_ln_in < 0 || c->r_ln_post < 0 || c->lres <= 0) { fprintf(stderr, "vault layout not supported by this engine (see engine_u.reasons in vault.json)\n"); return -1; }
    if (c->codec != 0 && c->codec != 1) { fprintf(stderr, "unknown weight codec %d\n", c->codec); return -1; }
    if (c->nl > 60 || c->hd > 256 || c->hd % 2) { fprintf(stderr, "unsupported dims\n"); return -1; }
    int lm_s = c->tie ? c->nl : c->nl + 1; if ((int)E->st.size() <= lm_s) { fprintf(stderr, "idx has too few streams\n"); return -1; }
    snprintf(p, sizeof p, "%s\\vault.bin", vdir); FILE *vf = fopen(p, "rb"); if (!vf) { fprintf(stderr, "cannot open vault.bin\n"); return -1; }
    size_t maxs = 0; for (auto &s : E->st) if (s.size > maxs) maxs = s.size; std::vector<uint8_t> host(maxs);
    for (int si = 0; si <= lm_s && (E->only_layers < 0 || si < E->only_layers); si++) {
        Str &s = E->st[si]; _fseeki64(vf, (long long)s.off, SEEK_SET); if (fread(host.data(), 1, s.size, vf) != s.size) { fprintf(stderr, "short read stream %d\n", si); return -1; }
        uint32_t magic, nsub; memcpy(&magic, host.data(), 4); memcpy(&nsub, host.data() + 4, 4); if (magic != 0x534C5657) { fprintf(stderr, "bad stream magic %d\n", si); return -1; }
        uint8_t *d = (uint8_t *)dalloc(E, s.size); E->vram_weights += s.size; CK(cudaMemcpy(d, host.data(), s.size, cudaMemcpyHostToDevice));
        if (c->codec == 1) {
            std::vector<uint32_t> tabs((size_t)nsub * 4096); uint32_t *dt = (uint32_t *)dalloc(E, 4 * tabs.size());
            for (uint32_t i = 0; i < nsub; i++) {
                uint64_t o; memcpy(&o, host.data() + 8 + 8 * i, 8); const uint8_t *b = host.data() + o; uint32_t rows, cols; memcpy(&rows, b, 4); memcpy(&cols, b + 4, 4);
                int R = b[9]; if (b[8] != 12 || R < 1) { fprintf(stderr, "stream %d: unsupported rans4 parameters\n", si); return -1; }
                size_t nblk = (rows + R - 1) / R, p_off = 524, p_cs = p_off + 4 * (4 * nblk + 1); uint32_t cs_len; memcpy(&cs_len, b + p_off + 4 * (4 * nblk), 4); size_t p_sm = p_cs + cs_len + 8;
                if (!rans_tab_host(b + 12, &tabs[(size_t)i * 4096])) { fprintf(stderr, "stream %d: bad rans4 frequency table\n", si); return -1; }
                Mat m = {d + o + p_cs, (const uint32_t *)(d + o + p_off), d + o + p_sm, nullptr, (int)rows, (int)cols, dt + (size_t)i * 4096, R};
                if (si < c->nl) E->mats[si].push_back(m); if (si == c->nl) E->emb = m; if (si == lm_s) E->lm = m;
            }
            CK(cudaMemcpy(dt, tabs.data(), 4 * tabs.size(), cudaMemcpyHostToDevice));
            if (si < c->nl && nsub != 7) { fprintf(stderr, "layer %d has %u matrices, engine needs 7\n", si, nsub); return -1; }
            continue;
        }
        std::vector<uint16_t> luts((size_t)nsub << LMAX); uint16_t *dl = (uint16_t *)dalloc(E, sizeof(uint16_t) * luts.size());
        for (uint32_t i = 0; i < nsub; i++) {
            uint64_t o; memcpy(&o, host.data() + 8 + 8 * i, 8); const uint8_t *b = host.data() + o; uint32_t rows, cols, es_len; memcpy(&rows, b, 4); memcpy(&cols, b + 4, 4);
            size_t no4 = 4 * (size_t)rows + 1, p_off = 264, p_es = p_off + 4 * no4; memcpy(&es_len, b + p_off + 4 * (no4 - 1), 4); size_t p_sm = p_es + es_len + 8;
            build_lut(b + 8, &luts[(size_t)i << LMAX]);
            Mat m = {d + o + p_es, (const uint32_t *)(d + o + p_off), d + o + p_sm, dl + ((size_t)i << LMAX), (int)rows, (int)cols, nullptr, 0};
            if (si < c->nl) E->mats[si].push_back(m); if (si == c->nl) E->emb = m; if (si == lm_s) E->lm = m;
        }
        CK(cudaMemcpy(dl, luts.data(), sizeof(uint16_t) * luts.size(), cudaMemcpyHostToDevice));
        if (si < c->nl && nsub != 7) { fprintf(stderr, "layer %d has %u matrices, engine needs 7\n", si, nsub); return -1; }
    }
    fclose(vf);
    size_t nres = (size_t)c->nl * c->lres + c->hid; std::vector<float> rh(nres); snprintf(p, sizeof p, "%s\\resident.bin", vdir); f = fopen(p, "rb");
    if (!f || fread(rh.data(), 4, nres, f) != nres) { fprintf(stderr, "resident.bin missing or wrong size\n"); return -1; } fclose(f);
    E->res = (float *)dalloc(E, 4 * nres); CK(cudaMemcpy(E->res, rh.data(), 4 * nres, cudaMemcpyHostToDevice)); E->fnrm = E->res + (size_t)c->nl * c->lres;
    int half = c->hd / 2, hid = c->hid, qd = c->nh * c->hd, kvd = c->nkv * c->hd; std::vector<float> cs((size_t)ctx * half), sn((size_t)ctx * half);
    for (int i = 0; i < half; i++) {
        double fr = pow(c->theta, -2.0 * i / c->hd);
        if (c->rl3) { double wl = 2.0 * 3.14159265358979323846 / fr, lo = c->rorig / c->rlow, hi = c->rorig / c->rhigh;
            if (wl > lo) fr /= c->rfactor; else if (wl >= hi) { double sm = (c->rorig / wl - c->rlow) / (c->rhigh - c->rlow); fr = (1.0 - sm) * fr / c->rfactor + sm * fr; } }
        for (int pos = 0; pos < ctx; pos++) { double a = pos * fr; cs[(size_t)pos * half + i] = (float)cos(a); sn[(size_t)pos * half + i] = (float)sin(a); }
    }
    E->cs = (float *)dalloc(E, 4 * cs.size()); E->sn = (float *)dalloc(E, 4 * sn.size()); CK(cudaMemcpy(E->cs, cs.data(), 4 * cs.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(E->sn, sn.data(), 4 * sn.size(), cudaMemcpyHostToDevice));
    E->H = (float *)dalloc(E, 4 * NBMAX * hid); E->XN = (float *)dalloc(E, 4 * NBMAX * hid); E->Q = (float *)dalloc(E, 4 * NBMAX * qd); E->K = (float *)dalloc(E, 4 * NBMAX * kvd); E->V = (float *)dalloc(E, 4 * NBMAX * kvd); E->ATT = (float *)dalloc(E, 4 * NBMAX * qd);
    E->G = (float *)dalloc(E, 4 * (size_t)NBMAX * c->inter); E->U = (float *)dalloc(E, 4 * (size_t)NBMAX * c->inter); E->logits = (float *)dalloc(E, 4 * (size_t)c->vocab);
    E->kc = (float *)dalloc(E, 4 * (size_t)(E->only_layers >= 0 ? E->only_layers : c->nl) * ctx * kvd); E->vc = (float *)dalloc(E, 4 * (size_t)(E->only_layers >= 0 ? E->only_layers : c->nl) * ctx * kvd); E->hlog = (float *)malloc(4 * (size_t)c->vocab);
    return 0;
}

template <int TPB = 64> static void mv(const Mat &m, const float *x, float *y, const float *bias, int acc) {
    if (m.tab) { int chains = 4 * ((m.rows + m.R - 1) / m.R); mv_rans_k<<<(chains + 127) / 128, 128>>>(m, x, y, bias, acc); return; }
    mv_k<TPB><<<(m.rows + (TPB / 4) - 1) / (TPB / 4), TPB>>>(m, x, y, bias, acc); }

static void layer_step(Eng *E, int l, int pos) {
    Cfg *c = &E->c; int hid = c->hid, nh = c->nh, nkv = c->nkv, hd = c->hd, kvd = nkv * hd; float eps = (float)c->eps, scale = 1.0f / sqrtf((float)hd), half = hd / 2;
        const float *rl = E->res + (size_t)l * c->lres; auto R = [&](int o) -> const float * { return o >= 0 ? rl + o : nullptr; }; std::vector<Mat> &S = E->mats[l];
        rms_k<<<1, 256>>>(E->H, R(c->r_ln_in), E->XN, hid, eps);
        mv(S[0], E->XN, E->Q, R(c->r_bq), 0); apply_patches(l, 0, E->XN, E->Q); mv(S[1], E->XN, E->K, R(c->r_bk), 0); apply_patches(l, 1, E->XN, E->K); mv(S[2], E->XN, E->V, R(c->r_bv), 0); apply_patches(l, 2, E->XN, E->V);
        if (c->r_qn >= 0) rms_k<<<nh, 128>>>(E->Q, R(c->r_qn), E->Q, hd, eps);
        if (c->r_kn >= 0) rms_k<<<nkv, 128>>>(E->K, R(c->r_kn), E->K, hd, eps);
        size_t slot = ((size_t)l * E->ctx + pos) * kvd;
        rope_k<<<nh + nkv, 64>>>(E->Q, E->K, E->V, E->kc + slot, E->vc + slot, E->cs + (size_t)pos * (int)half, E->sn + (size_t)pos * (int)half, nh, hd);
        attn_k<<<nh, 128, (pos + 1) * sizeof(float)>>>(E->Q, E->kc + (size_t)l * E->ctx * kvd, E->vc + (size_t)l * E->ctx * kvd, E->ATT, nh, nkv, hd, pos + 1, scale);
        mv(S[3], E->ATT, E->H, R(c->r_bo), 1); apply_patches(l, 3, E->ATT, E->H);
        rms_k<<<1, 256>>>(E->H, R(c->r_ln_post), E->XN, hid, eps);
        mv(S[4], E->XN, E->G, R(c->r_bg), 0); apply_patches(l, 4, E->XN, E->G); mv(S[5], E->XN, E->U, R(c->r_bu), 0); apply_patches(l, 5, E->XN, E->U);
        silu_mul_k<<<(c->inter + 255) / 256, 256>>>(E->G, E->U, c->inter);
        mv(S[6], E->G, E->H, R(c->r_bd), 1); apply_patches(l, 6, E->G, E->H);
}

static void mvb(const Mat &m, const float *x, float *y, const float *bias, int acc, int n) {
    if (m.tab) { int chains = 4 * ((m.rows + m.R - 1) / m.R), g = (chains + 127) / 128;
        if (n == 1) mvb_rans_k<1><<<g, 128>>>(m, x, y, bias, acc, n); else if (n <= 4) mvb_rans_k<4><<<g, 128>>>(m, x, y, bias, acc, n); else mvb_rans_k<NBMAX><<<g, 128>>>(m, x, y, bias, acc, n); return; }
    int g = (m.rows + 15) / 16;
    if (n == 1) mvb_k<64, 1><<<g, 64>>>(m, x, y, bias, acc, n); else if (n <= 4) mvb_k<64, 4><<<g, 64>>>(m, x, y, bias, acc, n); else mvb_k<64, NBMAX><<<g, 64>>>(m, x, y, bias, acc, n);
}
static void patches_b(int l, int k, const float *x, float *y, int n, int in_d, int out_d) { for (int t = 0; t < n; t++) apply_patches(l, k, x + (size_t)t * in_d, y + (size_t)t * out_d); }

// n tokens at positions pos0..pos0+n-1 through layer l; H holds n rows
static void layer_step_b(Eng *E, int l, int pos0, int n) {
    Cfg *c = &E->c; int hid = c->hid, nh = c->nh, nkv = c->nkv, hd = c->hd, qd = nh * hd, kvd = nkv * hd, inter = c->inter, half = hd / 2; float eps = (float)c->eps, scale = 1.0f / sqrtf((float)hd);
    const float *rl = E->res + (size_t)l * c->lres; auto R = [&](int o) -> const float * { return o >= 0 ? rl + o : nullptr; }; std::vector<Mat> &S = E->mats[l];
    rms_k<<<n, 256>>>(E->H, R(c->r_ln_in), E->XN, hid, eps);
    mvb(S[0], E->XN, E->Q, R(c->r_bq), 0, n); patches_b(l, 0, E->XN, E->Q, n, hid, qd);
    mvb(S[1], E->XN, E->K, R(c->r_bk), 0, n); patches_b(l, 1, E->XN, E->K, n, hid, kvd);
    mvb(S[2], E->XN, E->V, R(c->r_bv), 0, n); patches_b(l, 2, E->XN, E->V, n, hid, kvd);
    if (c->r_qn >= 0) rms_k<<<n * nh, 128>>>(E->Q, R(c->r_qn), E->Q, hd, eps);
    if (c->r_kn >= 0) rms_k<<<n * nkv, 128>>>(E->K, R(c->r_kn), E->K, hd, eps);
    for (int t = 0; t < n; t++) { int pos = pos0 + t; size_t slot = ((size_t)l * E->ctx + pos) * kvd;
        rope_k<<<nh + nkv, 64>>>(E->Q + (size_t)t * qd, E->K + (size_t)t * kvd, E->V + (size_t)t * kvd, E->kc + slot, E->vc + slot, E->cs + (size_t)pos * half, E->sn + (size_t)pos * half, nh, hd); }
    for (int t = 0; t < n; t++) { int pos = pos0 + t;
        attn_k<<<nh, 128, (pos + 1) * sizeof(float)>>>(E->Q + (size_t)t * qd, E->kc + (size_t)l * E->ctx * kvd, E->vc + (size_t)l * E->ctx * kvd, E->ATT + (size_t)t * qd, nh, nkv, hd, pos + 1, scale); }
    mvb(S[3], E->ATT, E->H, R(c->r_bo), 1, n); patches_b(l, 3, E->ATT, E->H, n, qd, hid);
    rms_k<<<n, 256>>>(E->H, R(c->r_ln_post), E->XN, hid, eps);
    mvb(S[4], E->XN, E->G, R(c->r_bg), 0, n); patches_b(l, 4, E->XN, E->G, n, hid, inter);
    mvb(S[5], E->XN, E->U, R(c->r_bu), 0, n); patches_b(l, 5, E->XN, E->U, n, hid, inter);
    silu_mul_k<<<(n * inter + 255) / 256, 256>>>(E->G, E->U, n * inter);
    mvb(S[6], E->G, E->H, R(c->r_bd), 1, n); patches_b(l, 6, E->G, E->H, n, inter, hid);
}

static void forward(Eng *E, int tok, int pos, bool want_logits, bool dump_layers) {
    Cfg *c = &E->c; int hid = c->hid, nh = c->nh, nkv = c->nkv, hd = c->hd, qd = nh * hd, kvd = nkv * hd; float eps = (float)c->eps, scale = 1.0f / sqrtf((float)hd), half = hd / 2;
    if (E->emb.tab) embed_rans_k<<<1, 4>>>(E->emb, tok, E->H); else embed_k<<<1, 4>>>(E->emb, tok, E->H);
    for (int l = 0; l < c->nl; l++) {
        layer_step(E, l, pos);
        if (dump_layers && E->dump) { std::vector<float> h(hid); CK(cudaMemcpy(h.data(), E->H, 4 * hid, cudaMemcpyDeviceToHost)); fwrite(h.data(), 4, hid, E->dump); }
    }
    if (want_logits) { rms_k<<<1, 256>>>(E->H, E->fnrm, E->XN, hid, eps); mv(E->lm, E->XN, E->logits, nullptr, 0); CK(cudaMemcpy(E->hlog, E->logits, 4 * (size_t)c->vocab, cudaMemcpyDeviceToHost)); }
    CK(cudaGetLastError());
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint64_t rng_next(uint64_t *s) { uint64_t x = *s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *s = x; return x * 2685821657736338717ull; }
static float rng_f(uint64_t *s) { return (float)(rng_next(s) >> 40) / 16777216.0f; }
static int sample_tok(const float *lgt, int V, float temp, float topp, int topk, uint64_t *rng) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (lgt[i] > lgt[b]) b = i; return b; }
    if (topk < 1 || topk > 128) topk = 128; if (topk > V) topk = V;
    int idx[128]; float val[128]; int k = 0;
    for (int i = 0; i < V; i++) { float v = lgt[i]; if (k == topk && v <= val[k - 1]) continue; int j = (k < topk) ? k++ : k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; } val[j] = v; idx[j] = i; }
    float p[128], sum = 0, mx = val[0]; for (int j = 0; j < k; j++) { p[j] = expf((val[j] - mx) / temp); sum += p[j]; }
    for (int j = 0; j < k; j++) p[j] /= sum;
    float cum = 0; int kk = k; for (int j = 0; j < k; j++) { cum += p[j]; if (cum >= topp) { kk = j + 1; break; } }
    float s2 = 0; for (int j = 0; j < kk; j++) s2 += p[j]; float r = rng_f(rng) * s2, c2 = 0;
    for (int j = 0; j < kk; j++) { c2 += p[j]; if (r <= c2) return idx[j]; }
    return idx[kk - 1];
}

static int dist_tok(const float *lgt, int V, float temp, float topp, int topk, int *ids, float *pr) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (lgt[i] > lgt[b]) b = i; ids[0] = b; pr[0] = 1.0f; return 1; }
    if (topk < 1 || topk > 128) topk = 128; if (topk > V) topk = V;
    int idx[128]; float val[128]; int k = 0;
    for (int i = 0; i < V; i++) { float v = lgt[i]; if (k == topk && v <= val[k - 1]) continue; int j = (k < topk) ? k++ : k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; } val[j] = v; idx[j] = i; }
    float p[128], sum = 0, mx = val[0]; for (int j = 0; j < k; j++) { p[j] = expf((val[j] - mx) / temp); sum += p[j]; }
    for (int j = 0; j < k; j++) p[j] /= sum;
    float cum = 0; int kk = k; for (int j = 0; j < k; j++) { cum += p[j]; if (cum >= topp) { kk = j + 1; break; } }
    float s2 = 0; for (int j = 0; j < kk; j++) s2 += p[j];
    for (int j = 0; j < kk; j++) { ids[j] = idx[j]; pr[j] = p[j] / s2; }
    return kk;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: gpu_engine serve|bench <vault_dir> [--ctx N] [--tokens N]\n"); return 2; }
    std::string cmd = argv[1]; int ctx = 256, ntok = 64, layers = -1; const char *patchf = nullptr;
    for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]); else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) ntok = atoi(argv[++i]); else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]); else if (!strcmp(argv[i], "--patch") && i + 1 < argc) patchf = argv[++i]; else if (argv[i][0] == '-' && i + 1 < argc && argv[i + 1][0] != '-') i++; }
    Eng E; E.only_layers = (cmd == "worker") ? layers : -1; double t0 = now(); if (eng_open(&E, argv[2], ctx)) return 1;
    if (patchf) patch_load(patchf, E.c);
    size_t fr, tot; cudaMemGetInfo(&fr, &tot);
    fprintf(stderr, "gpu engine ready: %d layers, weights %.0f MB in VRAM (compressed), VRAM free now %.0f MB, load %.1fs\n", E.c.nl, E.vram_weights / 1e6, fr / 1e6, now() - t0);
    const char *dh = getenv("NVME_DUMP_H"); if (dh && cmd != "worker") E.dump = fopen(dh, "wb");
    if (cmd == "worker") {      // split mode: layers [0, N) for the CPU engine, binary protocol on stdin/stdout
        if (E.only_layers < 1 || E.only_layers > E.c.nl) { fprintf(stderr, "worker needs --layers N (1..%d)\n", E.c.nl); return 1; }
        _setmode(_fileno(stdin), _O_BINARY); _setmode(_fileno(stdout), _O_BINARY); fwrite("READY\n", 1, 6, stdout); fflush(stdout);
        int hid = E.c.hid, N = E.only_layers; std::vector<float> hin, hl((size_t)N * hid);
        for (;;) {
            uint32_t h[4]; if (fread(h, 4, 4, stdin) != 4 || h[0] == 0x54495551u) break;
            if (h[0] != 0x31574446u) { fprintf(stderr, "worker: bad request header\n"); return 1; }
            int n = (int)h[1], pos0 = (int)h[2]; bool dump = h[3] != 0; hin.resize((size_t)n * hid);
            if (fread(hin.data(), 4, hin.size(), stdin) != hin.size()) break;
            if (pos0 + n > ctx) { fprintf(stderr, "worker: position %d exceeds ctx %d\n", pos0 + n, ctx); return 1; }
            for (int t0 = 0; t0 < n; t0 += NBMAX) {
                int m = n - t0 < NBMAX ? n - t0 : NBMAX;
                CK(cudaMemcpy(E.H, &hin[(size_t)t0 * hid], 4 * (size_t)m * hid, cudaMemcpyHostToDevice));
                for (int l = 0; l < N; l++) { layer_step_b(&E, l, pos0 + t0, m); if (dump && t0 + m == n) CK(cudaMemcpy(&hl[(size_t)l * hid], E.H + (size_t)(m - 1) * hid, 4 * (size_t)hid, cudaMemcpyDeviceToHost)); }
                CK(cudaMemcpy(&hin[(size_t)t0 * hid], E.H, 4 * (size_t)m * hid, cudaMemcpyDeviceToHost));
            }
            CK(cudaGetLastError()); fwrite(hin.data(), 4, hin.size(), stdout); if (dump) fwrite(hl.data(), 4, hl.size(), stdout); fflush(stdout);
        }
        return 0;
    }
    if (cmd == "bench") {
        int V = E.c.vocab; std::vector<int> prompt(8); for (int i = 0; i < 8; i++) prompt[i] = (1000 + 7919 * i) % V; int pos = 0, tok = 0;
        for (int i = 0; i < 8; i++) { forward(&E, prompt[i], pos++, i == 7, false); } tok = 0; for (int i = 1; i < V; i++) if (E.hlog[i] > E.hlog[tok]) tok = i;
        cudaDeviceSynchronize(); double t1 = now(); int n = std::min(ntok, ctx - pos - 1);
        for (int s = 0; s < n; s++) { forward(&E, tok, pos++, true, false); tok = 0; for (int i = 1; i < V; i++) if (E.hlog[i] > E.hlog[tok]) tok = i; }
        double dt = now() - t1; printf("GPU bench: %d tokens in %.3f s = %.1f tok/s (%.2f ms/token), weights %.0f MB VRAM-resident\n", n, dt, n / dt, 1e3 * dt / n, E.vram_weights / 1e6); return 0;
    }
    printf("READY\n"); fflush(stdout);
    std::vector<char> lbuf(1 << 22); char *line = lbuf.data();
    auto read_tokens = [&](int ntok, std::vector<int> &pt) { pt.assign(ntok, 0); if (!fgets(line, 1 << 22, stdin)) return false; char *q = line; for (int i = 0; i < ntok; i++) { char *e; pt[i] = (int)strtol(q, &e, 10); q = e; } return true; };
    while (fgets(line, 1 << 22, stdin)) {
        if (!strncmp(line, "QUIT", 4)) break;
        else if (!strncmp(line, "PING", 4)) { printf("PONG\n"); fflush(stdout); }
        else if (!strncmp(line, "RESET", 5)) { E.pos = 0; printf("OK\n"); fflush(stdout); }
        else if (!strncmp(line, "DRAFTP", 6)) {
            int k = 0, topk = 20, ntok = 0; float temp = 0.7f, topp = 0.8f; unsigned long long seed = 1;
            if (sscanf(line, "DRAFTP %d %f %f %d %llu %d", &k, &temp, &topp, &topk, &seed, &ntok) != 6) { printf("ERR bad DRAFTP\n"); fflush(stdout); continue; }
            std::vector<int> pt; if (!read_tokens(ntok, pt)) break;
            if (ntok < 1 || E.pos + ntok + k > ctx) { printf("ERR ctx %d %d\n", E.pos, ctx); fflush(stdout); continue; }
            uint64_t rng = seed ? seed : 88172645463325252ull;
            for (int i = 0; i < ntok; i++) forward(&E, pt[i], E.pos++, i == ntok - 1, false);
            for (int st = 0; st < k; st++) {
                int ids[128]; float pr[128]; int n = dist_tok(E.hlog, E.c.vocab, temp, topp, topk, ids, pr);
                float r = rng_f(&rng), c = 0; int tok = ids[n - 1];
                for (int j = 0; j < n; j++) { c += pr[j]; if (r < c) { tok = ids[j]; break; } }
                printf("D %d %d", tok, n); for (int j = 0; j < n; j++) printf(" %d %.9g", ids[j], pr[j]); printf("\n");
                forward(&E, tok, E.pos++, st < k - 1, false);
            }
            printf("END draftp %d\n", E.pos); fflush(stdout); }
        else if (!strncmp(line, "SETPOS", 6)) { int p = -1; sscanf(line, "SETPOS %d", &p);
            if (p < 0 || p > E.pos) printf("ERR setpos %d (pos %d)\n", p, E.pos); else { E.pos = p; printf("OK %d\n", p); } fflush(stdout); }
        else if (!strncmp(line, "LOGITS", 6)) {
            int n = 0; sscanf(line, "LOGITS %d", &n); std::vector<int> t; if (!read_tokens(n, t)) break;
            if (n < 1 || E.pos + n > ctx) { printf("ERR ctx %d %d\n", E.pos, ctx); fflush(stdout); continue; }
            for (int i = 0; i < n; i++) forward(&E, t[i], E.pos++, i == n - 1, i == n - 1);
            if (E.dump) fflush(E.dump);
            printf("L"); for (int i = 0; i < E.c.vocab; i++) printf(" %.9g", E.hlog[i]); printf("\nEND dbg 0 %d\n", E.pos); fflush(stdout);
        } else if (!strncmp(line, "GEN", 3)) {
            int max_new = 0, topk = 20, ntok = 0; float temp = 0.7f, topp = 0.8f; unsigned long long seed = 0;
            if (sscanf(line, "GEN %d %f %f %d %llu %d", &max_new, &temp, &topp, &topk, &seed, &ntok) != 6) { printf("ERR bad GEN\n"); fflush(stdout); continue; }
            std::vector<int> pt; if (!read_tokens(ntok, pt)) break;
            if (E.pos + ntok + 1 > ctx) { printf("ERR ctx %d %d\n", E.pos, ctx); fflush(stdout); continue; }
            uint64_t rng = seed ? seed : 88172645463325252ull; double tp0 = now();
            for (int i = 0; i < ntok; i++) forward(&E, pt[i], E.pos++, i == ntok - 1, false);
            double prefill_ms = (now() - tp0) * 1e3; std::vector<double> tms; int tok = sample_tok(E.hlog, E.c.vocab, temp, topp, topk, &rng), new_tok = 0; const char *reason = "max";
            for (;;) {
                printf("T %d\n", tok); fflush(stdout); new_tok++;
                if (tok == E.c.eos || tok == E.c.eos2) { reason = "eos"; break; }
                if (new_tok >= max_new) { forward(&E, tok, E.pos++, false, false); reason = "max"; break; }
                if (E.pos + 1 >= ctx) { reason = "ctx"; break; }
                double t1 = now(); forward(&E, tok, E.pos++, true, false); tok = sample_tok(E.hlog, E.c.vocab, temp, topp, topk, &rng); tms.push_back((now() - t1) * 1e3);
            }
            double dec = 0; for (double v : tms) dec += v; std::vector<double> srt = tms; std::sort(srt.begin(), srt.end());
            printf("S tok_s=%.3f ms_tok=%.0f p50_ms=%.0f stall_pct=0 read_MBps=0 prefill_ms=%.0f streamed_MB_tok=0 resident_MB=%.0f\n", dec > 0 ? 1000.0 * tms.size() / dec : 0.0,
                   tms.empty() ? 0.0 : dec / tms.size(), srt.empty() ? 0.0 : srt[srt.size() / 2], prefill_ms, E.vram_weights / 1e6);
            printf("END %s %d %d\n", reason, new_tok, E.pos); fflush(stdout);
        } else if (line[0] != '\n' && line[0] != '\r') { printf("ERR unknown command\n"); fflush(stdout); }
    }
    return 0;
}
