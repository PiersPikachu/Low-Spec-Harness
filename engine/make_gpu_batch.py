"""make_gpu_batch.py - batched layers for the split-mode GPU worker (gpu/gpu_engine.cu, in place).
Before: the worker ran its layers once per token, so a VERIFY of k+1 tokens decoded every GPU-resident weight k+1 times.
After: each compressed row is decoded once and multiplied by up to NBMAX token vectors (huffman4 and rans4 kernels);
attention stays per token (rope + KV write for the whole batch first, then causal attention for each token).
Per-token arithmetic is the same as the single-token kernels (same summation order per vector), so results do not change."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "gpu_engine.cu")
s = open(P, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "mvb_k" in s:
    print("already patched"); raise SystemExit(0)

def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt}, got {n}: {old[:80]!r}"; s = s.replace(old, new)

KERN = r'''
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
'''
rep("\n__device__ float block_sum(float v, float *red) {", KERN + "\n__device__ float block_sum(float v, float *red) {")

# activations sized for NBMAX tokens
rep("E->H = (float *)dalloc(E, 4 * hid); E->XN = (float *)dalloc(E, 4 * hid); E->Q = (float *)dalloc(E, 4 * qd); E->K = (float *)dalloc(E, 4 * kvd); E->V = (float *)dalloc(E, 4 * kvd); E->ATT = (float *)dalloc(E, 4 * qd);",
    "E->H = (float *)dalloc(E, 4 * NBMAX * hid); E->XN = (float *)dalloc(E, 4 * NBMAX * hid); E->Q = (float *)dalloc(E, 4 * NBMAX * qd); E->K = (float *)dalloc(E, 4 * NBMAX * kvd); E->V = (float *)dalloc(E, 4 * NBMAX * kvd); E->ATT = (float *)dalloc(E, 4 * NBMAX * qd);")
rep("E->G = (float *)dalloc(E, 4 * c->inter); E->U = (float *)dalloc(E, 4 * c->inter);",
    "E->G = (float *)dalloc(E, 4 * (size_t)NBMAX * c->inter); E->U = (float *)dalloc(E, 4 * (size_t)NBMAX * c->inter);")

HOST = r'''
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

static void forward(Eng *E, int tok, int pos, bool want_logits, bool dump_layers) {'''
rep("\nstatic void forward(Eng *E, int tok, int pos, bool want_logits, bool dump_layers) {", HOST)

# worker: whole batch per layer instead of token by token
rep('''            for (int t = 0; t < n; t++) {
                CK(cudaMemcpy(E.H, &hin[(size_t)t * hid], 4 * (size_t)hid, cudaMemcpyHostToDevice));
                for (int l = 0; l < N; l++) { layer_step(&E, l, pos0 + t); if (dump && t == n - 1) CK(cudaMemcpy(&hl[(size_t)l * hid], E.H, 4 * (size_t)hid, cudaMemcpyDeviceToHost)); }
                CK(cudaMemcpy(&hin[(size_t)t * hid], E.H, 4 * (size_t)hid, cudaMemcpyDeviceToHost));
            }''', '''            for (int t0 = 0; t0 < n; t0 += NBMAX) {
                int m = n - t0 < NBMAX ? n - t0 : NBMAX;
                CK(cudaMemcpy(E.H, &hin[(size_t)t0 * hid], 4 * (size_t)m * hid, cudaMemcpyHostToDevice));
                for (int l = 0; l < N; l++) { layer_step_b(&E, l, pos0 + t0, m); if (dump && t0 + m == n) CK(cudaMemcpy(&hl[(size_t)l * hid], E.H + (size_t)(m - 1) * hid, 4 * (size_t)hid, cudaMemcpyDeviceToHost)); }
                CK(cudaMemcpy(&hin[(size_t)t0 * hid], E.H, 4 * (size_t)m * hid, cudaMemcpyDeviceToHost));
            }''')
open(P, "w", encoding="utf-8", newline="").write(s)
print("batched worker layers added to", P)
