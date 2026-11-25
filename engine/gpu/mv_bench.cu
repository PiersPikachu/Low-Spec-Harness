// mv_bench.cu - G1: fused Huffman4-decode + bf16 matvec on the GPU, reading a v3 blob exactly as stored in vault.bin.
//   mv_bench <blob.bin> <x.bin> <y_out.bin> [reps]
// One thread per (row, stream): 4 threads decode the 4 interleaved exponent streams of one row,
// pair each exponent with its sign+mantissa byte, multiply by x[j], then reduce across the 4 lanes.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#define LMAX 12

static void build_lut(const uint8_t *ln, uint16_t *lut) {
    int cnt[LMAX + 2] = {0}, nxt[LMAX + 2] = {0}, c = 0;
    for (int s = 0; s < 256; s++) if (ln[s]) cnt[ln[s]]++;
    for (int l = 1; l <= LMAX; l++) { c = (c + cnt[l - 1]) << 1; nxt[l] = c; }
    memset(lut, 0, sizeof(uint16_t) << LMAX);
    for (int s = 0; s < 256; s++) {
        int l = ln[s]; if (!l) continue;
        int code = nxt[l]++, r = 0; for (int i = 0; i < l; i++) r |= ((code >> i) & 1) << (l - 1 - i);
        for (int k = 0; k < (1 << (LMAX - l)); k++) lut[r | (k << l)] = (uint16_t)(s | (l << 8));
    }
}

__global__ void __launch_bounds__(128) mv_huff4(const uint8_t *__restrict__ es, const uint32_t *__restrict__ off, const uint8_t *__restrict__ sm,
                                                const uint16_t *__restrict__ lut_g, const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    __shared__ uint16_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += blockDim.x) lut[i] = lut_g[i];
    __syncthreads();
    int r = blockIdx.x * (blockDim.x >> 2) + (threadIdx.x >> 2), k = threadIdx.x & 3;
    bool live = r < rows; float s = 0.f;
    if (live) {
        uint32_t pos = off[4 * (size_t)r + k]; uint64_t acc = 0; int nb = 0; const uint8_t *smr = sm + (size_t)r * cols;
        for (int j = k; j < cols; j += 4) {
            while (nb < 12) { acc |= (uint64_t)es[pos++] << nb; nb += 8; }
            uint32_t e = lut[acc & 4095]; int l = e >> 8; acc >>= l; nb -= l;
            uint32_t sy = smr[j];
            uint32_t w = ((sy & 0x80u) << 8) | ((e & 255u) << 7) | (sy & 0x7Fu);
            s += __uint_as_float(w << 16) * x[j];
        }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (live && k == 0) y[r] = s;
}


__device__ __forceinline__ uint32_t ld4(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
// v2: 32-bit refills with the next word prefetched, so the global-load latency hides behind the decode of >=2 symbols
// ablation knobs for timing experiments only (results are wrong when any is defined): -DNO_SM -DNO_X -DNO_ES -DNO_LUT
#ifdef NO_SM
#define SMLOAD ((uint32_t)(j & 0xff))
#else
#define SMLOAD smr[j]
#endif
#ifdef NO_X
#define XLOAD 1.0f
#else
#define XLOAD x[j]
#endif
#ifdef NO_ES
#define ESLOAD ((uint32_t)(pos * 2654435761u))
#else
#define ESLOAD ld4(es + pos)
#endif
#ifdef NO_LUT
#define LUTLOAD ((uint32_t)((acc & 4095) * 0x101u))
#else
#define LUTLOAD lut[acc & 4095]
#endif
template <int TPB> __global__ void __launch_bounds__(TPB) mv_huff4_v2(const uint8_t *__restrict__ es, const uint32_t *__restrict__ off, const uint8_t *__restrict__ sm,
                                                const uint16_t *__restrict__ lut_g, const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    __shared__ uint16_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += TPB) lut[i] = lut_g[i];
    __syncthreads();
    int r = blockIdx.x * (TPB >> 2) + (threadIdx.x >> 2), k = threadIdx.x & 3;
    bool live = r < rows; float s = 0.f;
    if (live) {
        uint32_t pos = off[4 * (size_t)r + k]; uint64_t acc = 0; int nb = 0; const uint8_t *smr = sm + (size_t)r * cols;
        uint32_t nxt = ESLOAD; pos += 4;
#pragma unroll 4
        for (int j = k; j < cols; j += 4) {
            if (nb < 12) { acc |= (uint64_t)nxt << nb; nb += 32; nxt = ESLOAD; pos += 4; }
            uint32_t e = LUTLOAD; int l = e >> 8; acc >>= l; nb -= l;
            uint32_t sy = SMLOAD;
            uint32_t w = ((sy & 0x80u) << 8) | ((e & 255u) << 7) | (sy & 0x7Fu);
            s += __uint_as_float(w << 16) * XLOAD;
        }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (live && k == 0) y[r] = s;
}

// v3: multi-symbol lookup table. Exponent codes average ~3 bits, so a 12-bit window usually holds 3 complete codes:
// one shared-memory lookup yields up to 3 symbols (bits 8..31), their total bit length (bits 0..3) and count (bits 4..5).
// The dependent chain (lookup -> shift -> lookup) is ~3x shorter. Summation order per lane is unchanged, so results equal v2 bit for bit.
static void build_lut3(const uint16_t *l1, uint32_t *l3) {
    for (int w = 0; w < (1 << LMAX); w++) {
        int used = 0, n = 0; uint32_t sy[3] = {0, 0, 0}; int ww = w;
        while (n < 3) { uint16_t e = l1[ww & ((1 << LMAX) - 1)]; int l = e >> 8; if (l == 0 || used + l > LMAX) break; sy[n++] = e & 255; used += l; ww >>= l; }
        l3[w] = (uint32_t)used | ((uint32_t)n << 4) | (sy[0] << 8) | (sy[1] << 16) | (sy[2] << 24);
    }
}
#define V3_SYM(SHIFT, IDX) { uint32_t sy_ = smr[j]; uint32_t w_ = ((sy_ & 0x80u) << 8) | ((((e >> (SHIFT)) & 255u)) << 7) | (sy_ & 0x7Fu); s += __uint_as_float(w_ << 16) * x[j]; j += 4; it++; }
template <int TPB> __global__ void __launch_bounds__(TPB) mv_huff4_v3(const uint8_t *__restrict__ es, const uint32_t *__restrict__ off, const uint8_t *__restrict__ sm,
                                                const uint32_t *__restrict__ lut_g, const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    __shared__ uint32_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += TPB) lut[i] = lut_g[i];
    __syncthreads();
    int r = blockIdx.x * (TPB >> 2) + (threadIdx.x >> 2), k = threadIdx.x & 3;
    bool live = r < rows; float s = 0.f;
    if (live) {
        int cnt = k < cols ? (cols - k + 3) >> 2 : 0, it = 0, j = k; uint32_t pos = off[4 * (size_t)r + k]; uint64_t a = 0; int nb = 0; const uint8_t *smr = sm + (size_t)r * cols;
        uint32_t nxt = ld4(es + pos); pos += 4;
        while (it < cnt) {
            if (nb < 12) { a |= (uint64_t)nxt << nb; nb += 32; nxt = ld4(es + pos); pos += 4; }
            uint32_t e = lut[a & 4095]; int used = e & 15, n = (e >> 4) & 3; a >>= used; nb -= used;
            if (n == 0) break;
            V3_SYM(8, 0)
            if (n > 1 && it < cnt) { V3_SYM(16, 1) if (n > 2 && it < cnt) { V3_SYM(24, 2) } }
        }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (live && k == 0) y[r] = s;
}

// v4: the ablation showed the kernel is instruction/latency bound, not DRAM bound (an empty loop with the same ALU work already costs ~1/4 of the time,
// and byte-granular es/sm loads cost the rest). Changes: (1) es read as aligned 32-bit words, (2) sign+mantissa bytes read as one 16-byte vector per 4 symbols,
// (3) x stored pre-permuted so a lane reads its 4 x values as one float4, (4) LUT entry = len | exp<<23 so building the float is 3 logic ops, (5) L1 prefetch ahead.
// Requires cols % 16 == 0 and a 16-byte aligned sm copy. Summation order per lane is unchanged -> bit-identical to v2.
__device__ __forceinline__ uint32_t ldw(const uint8_t *p) { return *(const uint32_t *)p; }
__device__ __forceinline__ void pf(const void *p) { asm volatile("prefetch.global.L1 [%0];" :: "l"(p)); }
static void build_lut4(const uint16_t *l1, uint32_t *l4) { for (int i = 0; i < (1 << LMAX); i++) { uint32_t e = l1[i]; l4[i] = (e >> 8) | ((e & 255u) << 23); } }
#define V4_STEP(I, XC) { if (nb < 12) { a |= (uint64_t)nxt << nb; nb += 32; nxt = ldw(es + wp); wp += 4; } \
    uint32_t e = lut[a & 4095]; a >>= (e & 15u); nb -= (int)(e & 15u); uint32_t t = (sw[I] >> (8 * k)) & 255u; \
    uint32_t f = (e & 0x7F800000u) | ((t & 0x80u) << 24) | ((t & 0x7Fu) << 16); s += __uint_as_float(f) * (XC); }
template <int TPB> __global__ void __launch_bounds__(TPB) mv_huff4_v4(const uint8_t *__restrict__ es, const uint32_t *__restrict__ off, const uint4 *__restrict__ sm4,
                                                const uint32_t *__restrict__ lut_g, const float4 *__restrict__ xp, float *__restrict__ y, int rows, int cols) {
    __shared__ uint32_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += TPB) lut[i] = lut_g[i];
    __syncthreads();
    int r = blockIdx.x * (TPB >> 2) + (threadIdx.x >> 2), k = threadIdx.x & 3; bool live = r < rows; float s = 0.f;
    if (live) {
        int nch = cols >> 4; uint32_t pos = off[4 * (size_t)r + k], a0 = pos & ~3u; int sh = (int)(pos & 3u) * 8;
        uint64_t a = (uint64_t)(ldw(es + a0) >> sh); int nb = 32 - sh; uint32_t nxt = ldw(es + a0 + 4), wp = a0 + 8; const uint4 *smr = sm4 + (size_t)r * nch;
        for (int c = 0; c < nch; c++) {
            pf(smr + c + 8); pf(es + wp + 128);
            uint4 v = smr[c]; float4 xv = xp[4 * c + k]; uint32_t sw[4] = {v.x, v.y, v.z, v.w};
            V4_STEP(0, xv.x) V4_STEP(1, xv.y) V4_STEP(2, xv.z) V4_STEP(3, xv.w)
        }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (live && k == 0) y[r] = s;
}

// v5/v6: ncu on v2 showed 38% occupancy, 14 cycles between issues, 65% of stalls on L1TEX scoreboard, shared-LUT bank conflicts 3.6 wavefronts/lookup.
// v4 cut L1 work in half but its 16 KB LUT halved occupancy, so no speedup. v5 = v4 with a 12-bit packed 8 KB LUT (occupancy back),
// v6 = v5 with RPT rows per thread: independent decode chains interleaved in one thread (ILP) because a single chain is latency bound.
static void build_lut5(const uint16_t *l1, uint16_t *l5) { for (int i = 0; i < (1 << LMAX); i++) { uint32_t e = l1[i]; l5[i] = (uint16_t)((e >> 8) | ((e & 255u) << 4)); } }
template <int TPB, int RPT> __global__ void __launch_bounds__(TPB) mv_huff4_v5(const uint8_t *__restrict__ es, const uint32_t *__restrict__ off, const uint4 *__restrict__ sm4,
                                                const uint16_t *__restrict__ lut_g, const float4 *__restrict__ xp, float *__restrict__ y, int rows, int cols) {
    __shared__ uint16_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += TPB) lut[i] = lut_g[i];
    __syncthreads();
    const int g = threadIdx.x >> 2, k = threadIdx.x & 3, nch = cols >> 4; float s[RPT]; uint64_t a[RPT]; int nb[RPT]; uint32_t nxt[RPT], wp[RPT]; const uint4 *smr[RPT]; bool live[RPT];
#pragma unroll
    for (int q = 0; q < RPT; q++) {
        int r = blockIdx.x * (TPB / 4) * RPT + q * (TPB / 4) + g; live[q] = r < rows; s[q] = 0.f; a[q] = 0; nb[q] = 0; nxt[q] = 0; wp[q] = 0; smr[q] = sm4;
        if (live[q]) { uint32_t pos = off[4 * (size_t)r + k], a0 = pos & ~3u; int sh = (int)(pos & 3u) * 8; a[q] = (uint64_t)(ldw(es + a0) >> sh); nb[q] = 32 - sh; nxt[q] = ldw(es + a0 + 4); wp[q] = a0 + 8; smr[q] = sm4 + (size_t)r * nch; }
    }
    for (int c = 0; c < nch; c++) {
        float4 xv = xp[4 * c + k]; uint4 v[RPT];
#pragma unroll
        for (int q = 0; q < RPT; q++) { pf(smr[q] + c + 8); v[q] = smr[q][c]; }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float xc = i == 0 ? xv.x : i == 1 ? xv.y : i == 2 ? xv.z : xv.w;
#pragma unroll
            for (int q = 0; q < RPT; q++) {
                if (nb[q] < 12) { a[q] |= (uint64_t)nxt[q] << nb[q]; nb[q] += 32; nxt[q] = ldw(es + wp[q]); wp[q] += 4; }
                uint32_t e = lut[a[q] & 4095]; uint32_t l = e & 15u; a[q] >>= l; nb[q] -= (int)l;
                uint32_t sw = i == 0 ? v[q].x : i == 1 ? v[q].y : i == 2 ? v[q].z : v[q].w; uint32_t t = (sw >> (8 * k)) & 255u;
                uint32_t f = ((e & 0xFF0u) << 19) | ((t & 0x80u) << 24) | ((t & 0x7Fu) << 16); s[q] += __uint_as_float(f) * xc;
            }
        }
    }
#pragma unroll
    for (int q = 0; q < RPT; q++) {
        float t = s[q]; t += __shfl_xor_sync(0xffffffffu, t, 1); t += __shfl_xor_sync(0xffffffffu, t, 2);
        int r = blockIdx.x * (TPB / 4) * RPT + q * (TPB / 4) + g; if (live[q] && k == 0) y[r] = t;
    }
}

// v8: TIMING EXPERIMENT ONLY (output is wrong on purpose). Emulates a vault format with 16 interleaved streams per row instead of 4:
// 4x more independent decode chains, each 1/4 as long, identical per-symbol work and memory traffic. Upper bound for a "GPU-friendly" format.
template <int TPB> __global__ void __launch_bounds__(TPB) mv_split16(const uint8_t *__restrict__ es, const uint32_t *__restrict__ off, const uint8_t *__restrict__ sm,
                                                const uint16_t *__restrict__ lut_g, const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    __shared__ uint16_t lut[1 << LMAX];
    for (int i = threadIdx.x; i < (1 << LMAX); i += TPB) lut[i] = lut_g[i];
    __syncthreads();
    int r = blockIdx.x * (TPB >> 4) + (threadIdx.x >> 4), k = threadIdx.x & 15, q = k & 3, p = k >> 2; bool live = r < rows; float s = 0.f;
    if (live) {
        uint32_t o0 = off[4 * (size_t)r + q], len = off[4 * (size_t)r + q + 1] - o0, pos = o0 + p * (len >> 2); int cnt = cols >> 4; int j = q + 4 * (p * cnt);
        uint64_t a = 0; int nb = 0; const uint8_t *smr = sm + (size_t)r * cols; uint32_t nxt = ld4(es + pos); pos += 4;
#pragma unroll 4
        for (int it = 0; it < cnt; it++, j += 4) {
            if (nb < 12) { a |= (uint64_t)nxt << nb; nb += 32; nxt = ld4(es + pos); pos += 4; }
            uint32_t e = lut[a & 4095]; int l = (e >> 8) | 1; a >>= l; nb -= l;
            uint32_t sy = smr[j]; uint32_t w = ((sy & 0x80u) << 8) | ((e & 255u) << 7) | (sy & 0x7Fu); s += __uint_as_float(w << 16) * x[j];
        }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2); s += __shfl_xor_sync(0xffffffffu, s, 4); s += __shfl_xor_sync(0xffffffffu, s, 8);
    if (live && k == 0) y[r] = s;
}

static std::vector<uint8_t> slurp(const char *p) { FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); } fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); std::vector<uint8_t> b(n); if (fread(b.data(), 1, n, f) != (size_t)n) exit(1); fclose(f); return b; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: mv_bench blob.bin x.bin y_out.bin [reps]\n"); return 2; }
    int reps = argc > 4 ? atoi(argv[4]) : 200, variant = argc > 5 ? atoi(argv[5]) : 1, tpb = argc > 6 ? atoi(argv[6]) : 128;
    std::vector<uint8_t> blob = slurp(argv[1]), xb = slurp(argv[2]);
    uint32_t rows, cols; memcpy(&rows, blob.data(), 4); memcpy(&cols, blob.data() + 4, 4);
    if (xb.size() != 4 * (size_t)cols) { fprintf(stderr, "x has %zu floats, matrix has %u cols\n", xb.size() / 4, cols); return 2; }
    size_t no4 = 4 * (size_t)rows + 1, p_off = 264, p_es = p_off + 4 * no4; uint32_t es_len; memcpy(&es_len, blob.data() + p_off + 4 * (no4 - 1), 4);
    size_t p_sm = p_es + es_len + 8; size_t read_bytes = es_len + (size_t)rows * cols;
    uint16_t *lut_h = (uint16_t *)malloc(sizeof(uint16_t) << LMAX); build_lut(blob.data() + 8, lut_h);
    std::vector<uint32_t> lut3_h(1 << LMAX); build_lut3(lut_h, lut3_h.data()); uint32_t *d_lut3; CK(cudaMalloc(&d_lut3, 4 << LMAX)); CK(cudaMemcpy(d_lut3, lut3_h.data(), 4 << LMAX, cudaMemcpyHostToDevice));
    uint32_t *d_lut4 = nullptr; uint8_t *d_smal = nullptr; float *d_xp = nullptr;
    uint16_t *d_lut5 = nullptr;
    if (variant >= 4 && cols % 16 != 0) { fprintf(stderr, "v4 needs cols %% 16 == 0 (cols=%u)\n", cols); return 3; }
    uint8_t *d_blob; uint16_t *d_lut; float *d_x, *d_y; CK(cudaMalloc(&d_blob, blob.size())); CK(cudaMalloc(&d_lut, sizeof(uint16_t) << LMAX));
    CK(cudaMalloc(&d_x, xb.size())); CK(cudaMalloc(&d_y, 4 * (size_t)rows));
    CK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_lut, lut_h, sizeof(uint16_t) << LMAX, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_x, xb.data(), xb.size(), cudaMemcpyHostToDevice));
    if (variant >= 4) {
        std::vector<uint16_t> l5(1 << LMAX); build_lut5(lut_h, l5.data()); CK(cudaMalloc(&d_lut5, 2 << LMAX)); CK(cudaMemcpy(d_lut5, l5.data(), 2 << LMAX, cudaMemcpyHostToDevice));
        std::vector<uint32_t> l4(1 << LMAX); build_lut4(lut_h, l4.data()); CK(cudaMalloc(&d_lut4, 4 << LMAX)); CK(cudaMemcpy(d_lut4, l4.data(), 4 << LMAX, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_smal, (size_t)rows * cols)); CK(cudaMemcpy(d_smal, d_blob + p_sm, (size_t)rows * cols, cudaMemcpyDeviceToDevice));
        std::vector<float> xo((float *)xb.data(), (float *)xb.data() + cols), xq(cols);
        for (uint32_t j = 0; j < cols; j++) xq[(j & ~15u) | ((j & 3u) << 2) | ((j >> 2) & 3u)] = xo[j];       // element j -> 16-block, lane (j&3) gets its 4 values contiguously
        CK(cudaMalloc(&d_xp, 4 * (size_t)cols)); CK(cudaMemcpy(d_xp, xq.data(), 4 * (size_t)cols, cudaMemcpyHostToDevice));
    }
    const uint32_t *d_off = (const uint32_t *)(d_blob + p_off); const uint8_t *d_es = d_blob + p_es, *d_sm = d_blob + p_sm;
    int blocks = (int)((rows + (tpb / 4) - 1) / (tpb / 4));
#define L5(T, R) mv_huff4_v5<T, R><<<(rows + (T / 4) * R - 1) / ((T / 4) * R), T>>>(d_es, d_off, (const uint4 *)d_smal, d_lut5, (const float4 *)d_xp, d_y, (int)rows, (int)cols)
#define LAUNCH() do { if (variant == 8 && tpb == 64) mv_split16<64><<<(rows + 3) / 4, 64>>>(d_es, d_off, d_sm, d_lut, d_x, d_y, (int)rows, (int)cols); \
    else if (variant == 8) mv_split16<128><<<(rows + 7) / 8, 128>>>(d_es, d_off, d_sm, d_lut, d_x, d_y, (int)rows, (int)cols); \
    else if (variant == 5 && tpb == 64) L5(64, 1); else if (variant == 5) L5(128, 1); else if (variant == 6 && tpb == 32) L5(32, 2); else if (variant == 6 && tpb == 64) L5(64, 2); else if (variant == 6) L5(128, 2); \
    else if (variant == 7 && tpb == 32) L5(32, 4); else if (variant == 7 && tpb == 64) L5(64, 4); else if (variant == 7) L5(128, 4); \
    else if (variant == 4 && tpb == 64) mv_huff4_v4<64><<<blocks, 64>>>(d_es, d_off, (const uint4 *)d_smal, d_lut4, (const float4 *)d_xp, d_y, (int)rows, (int)cols); \
    else if (variant == 4) mv_huff4_v4<128><<<blocks, 128>>>(d_es, d_off, (const uint4 *)d_smal, d_lut4, (const float4 *)d_xp, d_y, (int)rows, (int)cols); \
    else if (variant == 3 && tpb == 64) mv_huff4_v3<64><<<blocks, 64>>>(d_es, d_off, d_sm, d_lut3, d_x, d_y, (int)rows, (int)cols); \
    else if (variant == 3 && tpb == 128) mv_huff4_v3<128><<<blocks, 128>>>(d_es, d_off, d_sm, d_lut3, d_x, d_y, (int)rows, (int)cols); \
    else if (variant == 3) mv_huff4_v3<32><<<blocks, 32>>>(d_es, d_off, d_sm, d_lut3, d_x, d_y, (int)rows, (int)cols); \
    else if (variant == 1) mv_huff4<<<blocks, 128>>>(d_es, d_off, d_sm, d_lut, d_x, d_y, (int)rows, (int)cols); \
    else if (tpb == 32) mv_huff4_v2<32><<<blocks, 32>>>(d_es, d_off, d_sm, d_lut, d_x, d_y, (int)rows, (int)cols); \
    else if (tpb == 64) mv_huff4_v2<64><<<blocks, 64>>>(d_es, d_off, d_sm, d_lut, d_x, d_y, (int)rows, (int)cols); \
    else mv_huff4_v2<128><<<blocks, 128>>>(d_es, d_off, d_sm, d_lut, d_x, d_y, (int)rows, (int)cols); } while (0)
    if (variant == 1) tpb = 128;
    LAUNCH(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int i = 0; i < reps; i++) LAUNCH();
    cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); double us = ms * 1e3 / reps;
    std::vector<float> y(rows); CK(cudaMemcpy(y.data(), d_y, 4 * (size_t)rows, cudaMemcpyDeviceToHost));
    FILE *f = fopen(argv[3], "wb"); fwrite(y.data(), 4, rows, f); fclose(f);
    printf("v%d tpb=%d rows=%u cols=%u compressed_read=%.2f MB  time=%.1f us  effective=%.0f GB/s (of 336 theoretical)\n", variant, tpb, rows, cols, read_bytes / 1e6, us, read_bytes / 1e3 / us);
    return 0;
}
