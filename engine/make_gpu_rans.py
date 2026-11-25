"""make_gpu_rans.py - adds rans4 (cube-rANS) weight blobs to gpu/gpu_engine.cu (in place; asserts every edit matched once).
Kernel: one thread per chain (block of R rows x 1 of 4 column streams). The thread decodes its exponents row by row, joins them with the raw
sign/mantissa bytes, multiplies by x and keeps a running dot per row; the 4 chain threads of a row block reduce with warp shuffles."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "gpu_engine.cu")
s = open(P, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "mv_rans_k" in s:
    print("already patched"); raise SystemExit(0)

def rep(old, new):
    global s
    n = s.count(old); assert n == 1, f"expected 1 match, got {n}: {old[:70]!r}"; s = s.replace(old, new)

rep("struct Mat { const uint8_t *es; const uint32_t *off; const uint8_t *sm; const uint16_t *lut; int rows, cols; };",
    "struct Mat { const uint8_t *es; const uint32_t *off; const uint8_t *sm; const uint16_t *lut; int rows, cols; const uint32_t *tab; int R; };   // tab != null -> rans4 blob")
KERN = r'''
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
'''
rep("\n__device__ float block_sum(float v, float *red) {", KERN + "\n__device__ float block_sum(float v, float *red) {")
rep('    if (c->codec != 0) { fprintf(stderr, "this vault uses weight codec %d (rans4); the GPU engine reads huffman4 vaults only so far\\n", c->codec); return -1; }\n',
    '    if (c->codec != 0 && c->codec != 1) { fprintf(stderr, "unknown weight codec %d\\n", c->codec); return -1; }\n')
rep("        std::vector<uint16_t> luts((size_t)nsub << LMAX); uint16_t *dl = (uint16_t *)dalloc(E, sizeof(uint16_t) * luts.size());\n        for (uint32_t i = 0; i < nsub; i++) {\n            uint64_t o; memcpy(&o, host.data() + 8 + 8 * i, 8); const uint8_t *b = host.data() + o; uint32_t rows, cols, es_len; memcpy(&rows, b, 4); memcpy(&cols, b + 4, 4);\n",
    '''        if (c->codec == 1) {
            std::vector<uint32_t> tabs((size_t)nsub * 4096); uint32_t *dt = (uint32_t *)dalloc(E, 4 * tabs.size());
            for (uint32_t i = 0; i < nsub; i++) {
                uint64_t o; memcpy(&o, host.data() + 8 + 8 * i, 8); const uint8_t *b = host.data() + o; uint32_t rows, cols; memcpy(&rows, b, 4); memcpy(&cols, b + 4, 4);
                int R = b[9]; if (b[8] != 12 || R < 1) { fprintf(stderr, "stream %d: unsupported rans4 parameters\\n", si); return -1; }
                size_t nblk = (rows + R - 1) / R, p_off = 524, p_cs = p_off + 4 * (4 * nblk + 1); uint32_t cs_len; memcpy(&cs_len, b + p_off + 4 * (4 * nblk), 4); size_t p_sm = p_cs + cs_len + 8;
                if (!rans_tab_host(b + 12, &tabs[(size_t)i * 4096])) { fprintf(stderr, "stream %d: bad rans4 frequency table\\n", si); return -1; }
                Mat m = {d + o + p_cs, (const uint32_t *)(d + o + p_off), d + o + p_sm, nullptr, (int)rows, (int)cols, dt + (size_t)i * 4096, R};
                if (si < c->nl) E->mats[si].push_back(m); if (si == c->nl) E->emb = m; if (si == lm_s) E->lm = m;
            }
            CK(cudaMemcpy(dt, tabs.data(), 4 * tabs.size(), cudaMemcpyHostToDevice));
            if (si < c->nl && nsub != 7) { fprintf(stderr, "layer %d has %u matrices, engine needs 7\\n", si, nsub); return -1; }
            continue;
        }
        std::vector<uint16_t> luts((size_t)nsub << LMAX); uint16_t *dl = (uint16_t *)dalloc(E, sizeof(uint16_t) * luts.size());
        for (uint32_t i = 0; i < nsub; i++) {
            uint64_t o; memcpy(&o, host.data() + 8 + 8 * i, 8); const uint8_t *b = host.data() + o; uint32_t rows, cols, es_len; memcpy(&rows, b, 4); memcpy(&cols, b + 4, 4);
''')
rep("            Mat m = {d + o + p_es, (const uint32_t *)(d + o + p_off), d + o + p_sm, dl + ((size_t)i << LMAX), (int)rows, (int)cols};",
    "            Mat m = {d + o + p_es, (const uint32_t *)(d + o + p_off), d + o + p_sm, dl + ((size_t)i << LMAX), (int)rows, (int)cols, nullptr, 0};")
rep("template <int TPB = 64> static void mv(const Mat &m, const float *x, float *y, const float *bias, int acc) { mv_k<TPB><<<",
    "template <int TPB = 64> static void mv(const Mat &m, const float *x, float *y, const float *bias, int acc) {\n"
    "    if (m.tab) { int chains = 4 * ((m.rows + m.R - 1) / m.R); mv_rans_k<<<(chains + 127) / 128, 128>>>(m, x, y, bias, acc); return; }\n    mv_k<TPB><<<")
rep("    embed_k<<<1, 4>>>(E->emb, tok, E->H);", "    if (E->emb.tab) embed_rans_k<<<1, 4>>>(E->emb, tok, E->H); else embed_k<<<1, 4>>>(E->emb, tok, E->H);")
open(P, "w", encoding="utf-8", newline="").write(s)
print("rans4 GPU kernels added to", P)
