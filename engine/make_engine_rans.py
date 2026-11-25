"""make_engine_rans.py - adds cube-rANS (rans4) weight blobs to nvme_engine_u.c. Selected per vault by the idx line 'codec 1' (0 = huffman4).
Decode stays inside the matvec: a block of R rows is decoded from its 4 interleaved chains into exponent bytes, joined with the raw
sign/mantissa bytes, and dotted with x in exactly the same summation order as the huffman4 path (so logits are bit-identical across codecs)."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")

def rep(old, new, cnt=1):
    global s
    n = s.count(old)
    assert n == cnt, f"expected {cnt} match(es), got {n}: {old[:70]!r}"
    s = s.replace(old, new)

RANS = r'''/* ---------- cube-rANS (rans4) blobs: exponent rANS with M = 2^12, R rows per chain, 4 interleaved column streams ---------- */
static int G_CODEC = 0;   /* 0 huffman4, 1 rans4 (vault.idx 'codec') */
static int rans_tab(const uint8_t *fq_bytes, uint32_t *tab) {   /* slot -> sym | (freq-1)<<8 | (slot-cum)<<20 */
    uint32_t cum = 0;
    for (int s = 0; s < 256; s++) { uint16_t f; memcpy(&f, fq_bytes + 2 * s, 2); if (cum + f > 4096) return 0;
        for (uint32_t k = 0; k < f; k++) tab[cum + k] = (uint32_t)s | ((uint32_t)(f - 1) << 8) | (k << 20); cum += f; }
    return cum == 4096;
}
#define RRD(x, p) { x = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); p += 4; }
#define RSTEP(x, p, dst) { uint32_t t_ = tab[x & 4095u]; dst = (uint8_t)t_; x = (((t_ >> 8) & 4095u) + 1u) * (x >> 12) + (t_ >> 20); while (x < (1u << 23)) x = (x << 8) | *p++; }
/* decode nrows rows (row-major exponent bytes) of one block from its 4 chains (chain k holds columns j % 4 == k, rows in order) */
static void rans_rows(const uint32_t *tab, const uint8_t *cs, const uint32_t *o4, int cols, int nrows, uint8_t *e) {
    const uint8_t *p0 = cs + o4[0], *p1 = cs + o4[1], *p2 = cs + o4[2], *p3 = cs + o4[3]; uint32_t x0, x1, x2, x3;
    RRD(x0, p0) RRD(x1, p1) RRD(x2, p2) RRD(x3, p3)
    for (int r = 0; r < nrows; r++) {
        uint8_t *er = e + (size_t)r * cols; int j = 0;
        for (; j + 4 <= cols; j += 4) { RSTEP(x0, p0, er[j]) RSTEP(x1, p1, er[j + 1]) RSTEP(x2, p2, er[j + 2]) RSTEP(x3, p3, er[j + 3]) }
        for (; j < cols; j++) { int k = j & 3; if (k == 0) RSTEP(x0, p0, er[j]) else if (k == 1) RSTEP(x1, p1, er[j]) else if (k == 2) RSTEP(x2, p2, er[j]) else RSTEP(x3, p3, er[j]) }
    }
}
#undef RRD
#undef RSTEP
static inline void asm_row(const uint8_t *er, const uint8_t *sm, int cols, float *out) {
    for (int j = 0; j < cols; j++) { uint32_t e = er[j], v = sm[j], bits = ((v & 0x80u) << 24) | (e << 23) | ((v & 0x7Fu) << 16); memcpy(&out[j], &bits, 4); }
}
#if defined(__AVX2__)
static inline float dot_rans_row(const uint8_t *er, const uint8_t *sm, int cols, const float *x) {   /* same order as dot_row / dot_decode3 */
    __m256 acc = _mm256_setzero_ps(); int j = 0;
    for (; j + 8 <= cols; j += 8) acc = _mm256_add_ps(acc, _mm256_mul_ps(asm8(_mm_loadl_epi64((const __m128i *)(er + j)), sm + j), _mm256_loadu_ps(x + j)));
    float a[8]; _mm256_storeu_ps(a, acc);
    for (; j < cols; j++) { uint32_t e = er[j], v = sm[j], bits = ((v & 0x80u) << 24) | (e << 23) | ((v & 0x7Fu) << 16); float w; memcpy(&w, &bits, 4); a[0] += w * x[j]; }
    return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
}
#endif
static void mv_batch_rans(const uint8_t *blob, const float *X, float *Y, int n) {
    uint32_t rows, cols; memcpy(&rows, blob, 4); memcpy(&cols, blob + 4, 4); int mbits = blob[8], R = blob[9];
    if (mbits != 12 || R < 1) die("rans4 blob: unsupported parameters (engine needs M=2^12)");
    int nblk = ((int)rows + R - 1) / R; const uint32_t *off = (const uint32_t *)(blob + 524); const uint8_t *cs = blob + 524 + 4 * (4 * (size_t)nblk + 1);
    const uint8_t *sm = cs + off[4 * (size_t)nblk] + 8; uint32_t *tab = malloc(4 * 4096); if (!rans_tab(blob + 12, tab)) die("rans4 blob: bad frequency table");
#pragma omp parallel
    { uint8_t *e = malloc((size_t)R * cols + 64); float *buf = malloc(4 * (size_t)cols + 64);
#pragma omp for schedule(dynamic, 1)
      for (int b = 0; b < nblk; b++) {
          int r0 = b * R, nr = (int)rows - r0 < R ? (int)rows - r0 : R; rans_rows(tab, cs, off + 4 * (size_t)b, (int)cols, nr, e);
          for (int i = 0; i < nr; i++) {
              int r = r0 + i; const uint8_t *er = e + (size_t)i * cols, *smr = sm + (size_t)r * cols;
#if defined(__AVX2__)
              if (n == 1) { Y[r] = dot_rans_row(er, smr, (int)cols, X); continue; }
#endif
              asm_row(er, smr, (int)cols, buf); for (int t = 0; t < n; t++) Y[(size_t)t * rows + r] = dot_row(buf, X + (size_t)t * cols, (int)cols);
          }
      }
      free(e); free(buf); }
    free(tab);
}
'''
rep("/* Y[t][r] = sum_j W[r][j] * X[t][j] for n vectors; each row is decoded ONCE and reused for all n vectors */",
    RANS + "/* Y[t][r] = sum_j W[r][j] * X[t][j] for n vectors; each row is decoded ONCE and reused for all n vectors */")
rep("static void mv_batch(const uint8_t *blob, const float *X, float *Y, int n) {\n",
    "static void mv_batch(const uint8_t *blob, const float *X, float *Y, int n) {\n    if (G_CODEC == 1) { mv_batch_rans(blob, X, Y, n); return; }\n")
# config key
rep("int lres, r_ln_in, r_ln_post, r_qn, r_kn, r_bq, r_bk, r_bv, r_bo, r_bg, r_bu, r_bd, rl3, supported;",
    "int lres, r_ln_in, r_ln_post, r_qn, r_kn, r_bq, r_bk, r_bv, r_bo, r_bg, r_bu, r_bd, rl3, supported, codec;")
rep('else if (!strcmp(k, "supported")) c->supported = (int)v; else return 0;',
    'else if (!strcmp(k, "supported")) c->supported = (int)v; else if (!strcmp(k, "codec")) c->codec = (int)v; else return 0;')
rep("    E->lm_s = E->c.tie ? E->c.nl : E->c.nl + 1;\n    path_join(p, sizeof p, vdir, \"vault.bin\");",
    "    if (E->c.codec != 0 && E->c.codec != 1) { fprintf(stderr, \"unknown weight codec %d\\n\", E->c.codec); return -1; }\n"
    "    G_CODEC = E->c.codec; lg(\"[vault] weight codec %s\", G_CODEC ? \"rans4 (cube-rANS)\" : \"huffman4\");\n"
    "    E->lm_s = E->c.tie ? E->c.nl : E->c.nl + 1;\n    path_join(p, sizeof p, vdir, \"vault.bin\");")
# embedding rows (read from disk per token)
rep("uint32_t *emb_off, emb_rows, emb_cols; uint16_t *emb_lut;", "uint32_t *emb_off, emb_rows, emb_cols, *emb_tab; int emb_R; uint8_t *emb_e; uint16_t *emb_lut;")
rep("static void embed_init(Eng *E) {\n    uint8_t hdr[264]; uint64_t o0; Stream *s = &E->st[E->c.nl];\n    if (rf_pread(&E->ef, &o0, s->off + 8, 8)) die(\"cannot read embedding header\");\n    uint64_t sub = s->off + o0;",
    '''static void embed_init(Eng *E) {
    uint8_t hdr[264]; uint64_t o0; Stream *s = &E->st[E->c.nl];
    if (rf_pread(&E->ef, &o0, s->off + 8, 8)) die("cannot read embedding header");
    uint64_t sub = s->off + o0;
    if (G_CODEC == 1) {
        uint8_t h[524]; if (rf_pread(&E->ef, h, sub, 524)) die("cannot read embedding sub-header");
        memcpy(&E->emb_rows, h, 4); memcpy(&E->emb_cols, h + 4, 4); E->emb_R = h[9]; if (h[8] != 12 || E->emb_R < 1) die("embedding: unsupported rans4 parameters");
        size_t nblk = (E->emb_rows + E->emb_R - 1) / E->emb_R, no = 4 * nblk + 1; E->emb_off = malloc(4 * no);
        if (rf_pread(&E->ef, E->emb_off, sub + 524, 4 * no)) die("cannot read embedding chain table");
        E->emb_tab = malloc(4 * 4096); if (!rans_tab(h + 12, E->emb_tab)) die("embedding: bad frequency table");
        E->emb_es0 = sub + 524 + 4 * no; E->emb_sm0 = E->emb_es0 + E->emb_off[4 * nblk] + 8; uint32_t mx = 0;
        for (size_t b = 0; b < nblk; b++) { uint32_t l = E->emb_off[4 * b + 4] - E->emb_off[4 * b]; if (l > mx) mx = l; }
        E->emb_es = malloc((size_t)mx + 64); E->emb_sm = malloc(E->emb_cols + 8); E->emb_e = malloc((size_t)E->emb_R * E->emb_cols + 64); return;
    }''')
rep("static void embed_row(Eng *E, int tok, float *out) {\n    if (tok < 0 || (uint32_t)tok >= E->emb_rows) die(\"token id out of range\");\n",
    '''static void embed_row(Eng *E, int tok, float *out) {
    if (tok < 0 || (uint32_t)tok >= E->emb_rows) die("token id out of range");
    if (G_CODEC == 1) {
        int R = E->emb_R, b = tok / R, i = tok - b * R; const uint32_t *o = E->emb_off + 4 * (size_t)b; uint32_t a = o[0], o4[4] = {0, o[1] - a, o[2] - a, o[3] - a};
        if (rf_pread(&E->ef, E->emb_es, E->emb_es0 + a, (size_t)(o[4] - a) + 8)) die("embedding read failed");
        if (rf_pread(&E->ef, E->emb_sm, E->emb_sm0 + (uint64_t)tok * E->emb_cols, E->emb_cols)) die("embedding read failed");
        rans_rows(E->emb_tab, E->emb_es, o4, (int)E->emb_cols, i + 1, E->emb_e); asm_row(E->emb_e + (size_t)i * E->emb_cols, E->emb_sm, (int)E->emb_cols, out); return;
    }
''')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("rans4 codec added to", P)
