// gpu_experts.cu - GPU compute for MXFP4 (gpt-oss) experts that already sit in VRAM, called by nvme_engine_u.exe through
// LoadLibrary. The expert stream bytes are exactly the vault's (same super-blob the CPU reads): MXF4 gate_up, F32 bias, MXF4 down,
// F32 bias. Math matches the CPU path: FP4 code x 2^(scale-127), interleaved gate/up + bias, clamp(gate, max L), clamp(up, +-L),
// glu = gate * sigmoid(alpha gate), (up + 1) * glu, down + bias. fp32 throughout; only the order of additions differs.
// build: gpu\build_gpu_experts.bat   (nvcc -shared -cudart shared, so it shares the engine's cudart64_12.dll)
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#define EXPORT extern "C" __declspec(dllexport)

__constant__ float FP4C[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// one warp per output row; each lane reads one 32-bit word (8 FP4 codes) at a time, so a warp's loads are coalesced;
// FP4 table in shared memory (constant memory serialises divergent lookups); up to 8 token vectors per pass
__global__ void mxf4_k(const uint8_t *sub, int rows, int cols, const float *__restrict__ X, float *__restrict__ Y, int m) {
    __shared__ float lut[16]; if (threadIdx.x < 16) lut[threadIdx.x] = FP4C[threadIdx.x]; __syncthreads();
    int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (row >= rows) return;
    int nb = cols / 32, nw = cols / 8; const uint8_t *S = sub + 12 + (size_t)row * nb;
    const uint32_t *Wd = (const uint32_t *)(sub + 12 + (size_t)rows * nb + (size_t)row * nb * 16);   /* 4-byte aligned (see packer layout) */
    /* MXF2: scales = base + 2-bit offset per block, or raw bytes from the exception table (header: pb at +12, block offset at +20) */
    int mx2 = sub[3] == '2'; uint32_t desc = 0; const uint8_t *pk = 0, *ex = 0;
    if (mx2) { uint32_t pb = *(const uint32_t *)(sub + 12), boff = *(const uint32_t *)(sub + 20);
        desc = ((const uint32_t *)(sub + 24))[row]; pk = sub + 24 + 4 * (size_t)rows + (size_t)row * pb;
        ex = sub + 24 + 4 * (size_t)rows + (size_t)rows * pb + (size_t)(desc & 0x7fffffffu) * nb;
        Wd = (const uint32_t *)(sub + boff + (size_t)row * nb * 16); }
    for (int t0 = 0; t0 < m; t0 += 8) {
        int tn = m - t0 < 8 ? m - t0 : 8; float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int wi = lane; wi < nw; wi += 32) {
            int bi = wi >> 2; int e8 = !mx2 ? (int)__ldg(S + bi) : (desc >> 31) ? (int)__ldg(ex + bi) : (int)(desc & 255u) + ((__ldg(pk + (bi >> 2)) >> ((bi & 3) * 2)) & 3);
            uint32_t v = __ldg(Wd + wi); float sc = ldexpf(1.0f, e8 - 127); float w[8];
#pragma unroll
            for (int j = 0; j < 4; j++) { uint32_t byte = (v >> (8 * j)) & 255u; w[2 * j] = lut[byte & 15u]; w[2 * j + 1] = lut[byte >> 4]; }
            for (int t = 0; t < tn; t++) {
                const float4 *x = (const float4 *)(X + (size_t)(t0 + t) * cols + (size_t)wi * 8); float4 a = x[0], b = x[1];
                float s = w[0] * a.x; s = fmaf(w[1], a.y, s); s = fmaf(w[2], a.z, s); s = fmaf(w[3], a.w, s);
                s = fmaf(w[4], b.x, s); s = fmaf(w[5], b.y, s); s = fmaf(w[6], b.z, s); s = fmaf(w[7], b.w, s);
                acc[t] = fmaf(sc, s, acc[t]);
            }
        }
        for (int t = 0; t < tn; t++) {
            float v = acc[t];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
            if (lane == 0) Y[(size_t)(t0 + t) * rows + row] = v;
        }
    }
}
__global__ void act_k(const float *GU, const float *bgu, float *A, int m, int I, float lim, float al) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= m * I) return;
    int t = i / I, j = i % I; const float *g = GU + (size_t)t * 2 * I;
    float gt = g[2 * j] + bgu[2 * j], up = g[2 * j + 1] + bgu[2 * j + 1];
    gt = fminf(gt, lim); up = fminf(fmaxf(up, -lim), lim);
    A[i] = (up + 1.0f) * (gt / (1.0f + expf(-al * gt)));
}
__global__ void bias_k(float *D, const float *b, int m, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < m * n) D[i] += b[i % n]; }

static float *dX, *dGU, *dA, *dD; static int g_maxm; static cudaStream_t g_st;

EXPORT int gx_init(int maxm, int hid, int inter) {
    g_maxm = maxm; int c = hid > inter ? hid : inter;
    if (cudaStreamCreateWithFlags(&g_st, cudaStreamNonBlocking)) return 1;
    if (cudaMalloc(&dX, (size_t)maxm * c * 4) || cudaMalloc(&dGU, (size_t)maxm * 2 * inter * 4) || cudaMalloc(&dA, (size_t)maxm * inter * 4) || cudaMalloc(&dD, (size_t)maxm * c * 4)) return 2;
    return 0;
}
// dexp: device address of the expert stream; off_*: byte offsets of its four sub-blobs; hX: m x cols_gu (host); hOut: m x rows_d (host) = down + bias
EXPORT int gx_mxf4_expert(const uint8_t *dexp, uint64_t off_gu, uint64_t off_bgu, uint64_t off_d, uint64_t off_bd, int rows_gu, int cols_gu, int rows_d, int cols_d,
                          const float *hX, int m, float lim, float al, float *hOut) {
    if (m > g_maxm) return 3;
    cudaMemcpyAsync(dX, hX, (size_t)m * cols_gu * 4, cudaMemcpyHostToDevice, g_st);
    int th = 256, w = th / 32;
    mxf4_k<<<(rows_gu + w - 1) / w, th, 0, g_st>>>(dexp + off_gu, rows_gu, cols_gu, dX, dGU, m);
    int I = rows_gu / 2;
    act_k<<<(m * I + 255) / 256, 256, 0, g_st>>>(dGU, (const float *)(dexp + off_bgu + 8), dA, m, I, lim, al);
    mxf4_k<<<(rows_d + w - 1) / w, th, 0, g_st>>>(dexp + off_d, rows_d, cols_d, dA, dD, m);
    bias_k<<<(m * rows_d + 255) / 256, 256, 0, g_st>>>(dD, (const float *)(dexp + off_bd + 8), m, rows_d);
    cudaMemcpyAsync(hOut, dD, (size_t)m * rows_d * 4, cudaMemcpyDeviceToHost, g_st);
    return cudaStreamSynchronize(g_st) != cudaSuccess ? 4 : 0;
}
