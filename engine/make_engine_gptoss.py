"""make_engine_gptoss.py - gpt-oss support for nvme_engine_u.c (WVAULT 7 from pack_gptoss.py), on top of make_engine_moe.py.

  YaRN RoPE      inverse frequencies and attention factor from vault.idx (computed by transformers), float32 like transformers
  sinks          one extra logit per head inside the attention softmax, dropped after (no value)
  sliding window last `sliding_window` positions on layers marked 1 in layer_types, full causal on the others
  router         logits + bias, top-k on the logits, softmax over the k chosen
  experts        MXFP4 matvec straight from the released bytes (FP4 E2M1 code x 2^(scale-127)), interleaved gate/up with bias,
                 clamp(gate, max 7), clamp(up, +-7), glu = gate * sigmoid(1.702 gate), (up + 1) * glu, down with bias
"""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt}, got {n}: {old[:90]!r}"; s = s.replace(old, new)

rep("int moe, n_exp, top_k, norm_topk, exp0, minter; } Cfg;",
    "int moe, n_exp, top_k, norm_topk, exp0, minter; int gptoss, swin, r_sinks, r_rbias; double swlim, swalpha, rscale; unsigned char ltype[256]; float rinv[128]; } Cfg;")
rep('else if (!strcmp(k, "moe_inter")) c->minter = (int)v; else return 0;',
    'else if (!strcmp(k, "moe_inter")) c->minter = (int)v; else if (!strcmp(k, "gptoss")) c->gptoss = (int)v; else if (!strcmp(k, "sliding_window")) c->swin = (int)v;\n'
    '    else if (!strcmp(k, "swiglu_limit")) c->swlim = v; else if (!strcmp(k, "swiglu_alpha")) c->swalpha = v; else if (!strcmp(k, "rope_scale")) c->rscale = v;\n'
    '    else if (!strcmp(k, "r_sinks")) c->r_sinks = (int)v; else if (!strcmp(k, "r_rbias")) c->r_rbias = (int)v; else return 0;')
rep("if (E->ver != 3 && E->ver != 4 && E->ver != 6) {", "if (E->ver != 3 && E->ver != 4 && E->ver != 6 && E->ver != 7) {")
rep('        if (sscanf(line, "stream %d %llu %llu %llu %llu %llx", &idx, &a, &b, &c2, &d, &h) == 6)',
    '        if (!strncmp(line, "layer_types ", 12)) { char *q = line + 12, *e; for (int i = 0; i < 256; i++) { long x = strtol(q, &e, 10); if (e == q) break; E->c.ltype[i] = (unsigned char)x; q = e; } continue; }\n'
    '        if (!strncmp(line, "rope_inv ", 9)) { char *q = line + 9, *e; for (int i = 0; i < 128; i++) { float x = strtof(q, &e); if (e == q) break; E->c.rinv[i] = x; q = e; } continue; }\n'
    '        if (sscanf(line, "stream %d %llu %llu %llu %llu %llx", &idx, &a, &b, &c2, &d, &h) == 6)')
rep("        for (int pos = 0; pos < ctx; pos++) { double a = pos * fr; E->cosT[(size_t)pos * half + i] = (float)cos(a); E->sinT[(size_t)pos * half + i] = (float)sin(a); }",
    "        if (c->gptoss) { float rf = c->rinv[i], sc = (float)c->rscale;   /* transformers: float32 pos * inv_freq, cos/sin x attention factor */\n"
    "            for (int pos = 0; pos < ctx; pos++) { float a = (float)pos * rf; E->cosT[(size_t)pos * half + i] = cosf(a) * sc; E->sinT[(size_t)pos * half + i] = sinf(a) * sc; } continue; }\n"
    "        for (int pos = 0; pos < ctx; pos++) { double a = pos * fr; E->cosT[(size_t)pos * half + i] = (float)cos(a); E->sinT[(size_t)pos * half + i] = (float)sin(a); }")
# attention: sliding window start + sink logit
rep("            float *sc = malloc(4 * (size_t)(p + 1)); const float *qh = q + h * hd; int kvh = h / grp; float mx = -1e30f;\n"
    "            for (int tt = 0; tt <= p; tt++) {",
    "            float *sc = malloc(4 * (size_t)(p + 1)); const float *qh = q + h * hd; int kvh = h / grp; float mx = -1e30f;\n"
    "            int t0 = (c->gptoss && c->ltype[l] && c->swin > 0 && p - c->swin + 1 > 0) ? p - c->swin + 1 : 0; float snk = c->gptoss ? rl[c->r_sinks + h] : 0.f;\n"
    "            if (c->gptoss && snk > mx) mx = snk;\n"
    "            for (int tt = t0; tt <= p; tt++) {")
rep("            float sum = 0; for (int tt = 0; tt <= p; tt++) { sc[tt] = expf(sc[tt] - mx); sum += sc[tt]; }",
    "            float sum = c->gptoss ? expf(snk - mx) : 0; for (int tt = t0; tt <= p; tt++) { sc[tt] = expf(sc[tt] - mx); sum += sc[tt]; }")
rep("            for (int tt = 0; tt <= p; tt++) { const float *vv = vcl", "            for (int tt = t0; tt <= p; tt++) { const float *vv = vcl")

FFN = r'''/* ---------- gpt-oss: MXFP4 experts ---------- */
static const float FP4T[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
/* Y[t][r] = sum_j W[r][j] X[t][j], W stored as released: per row cols/32 E8M0 scales, then cols/2 bytes of FP4 codes (low nibble first) */
static void mxf4_mv(const uint8_t *blob, const float *X, float *Y, int n) {
    uint32_t rows, cols; memcpy(&rows, blob + 4, 4); memcpy(&cols, blob + 8, 4); int nb = (int)cols / 32;
    const uint8_t *S = blob + 12, *Bk = S + (size_t)rows * nb;
#ifdef __AVX2__
    /* 2 x FP4 value as int8 (exact: 0, +-1, +-2, +-3, +-4, +-6, +-8, +-12), looked up with pshufb; result halved at the end */
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12), m4 = _mm_set1_epi8(15);
#pragma omp parallel for schedule(dynamic, 16)
    for (int r = 0; r < (int)rows; r++) {
        __m256 acc[128]; for (int t = 0; t < n; t++) acc[t] = _mm256_setzero_ps();
        const uint8_t *sr = S + (size_t)r * nb, *br = Bk + (size_t)r * nb * 16;
        for (int b = 0; b < nb; b++) {
            __m128i q = _mm_loadu_si128((const __m128i *)(br + (size_t)b * 16));
            __m128i lo = _mm_and_si128(q, m4), hi = _mm_and_si128(_mm_srli_epi16(q, 4), m4);
            __m128i v0 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(lo, hi)), v1 = _mm_shuffle_epi8(lut, _mm_unpackhi_epi8(lo, hi));
            __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v0)), w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v0, 8)));
            __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v1)), w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v1, 8)));
            __m256 scl = _mm256_set1_ps(ldexpf(0.5f, (int)sr[b] - 127));
            for (int t = 0; t < n; t++) {
                const float *x = X + (size_t)t * cols + (size_t)b * 32;
                __m256 d = _mm256_mul_ps(w0, _mm256_loadu_ps(x)); d = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + 8), d);
                d = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + 16), d); d = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + 24), d);
                acc[t] = _mm256_fmadd_ps(scl, d, acc[t]);
            }
        }
        for (int t = 0; t < n; t++) { __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
            s4 = _mm_hadd_ps(s4, s4); s4 = _mm_hadd_ps(s4, s4); Y[(size_t)t * rows + r] = _mm_cvtss_f32(s4); }
    }
    return;
#endif
#pragma omp parallel for schedule(dynamic, 16)
    for (int r = 0; r < (int)rows; r++) {
        float acc[128], w[32]; for (int t = 0; t < n; t++) acc[t] = 0;
        const uint8_t *sr = S + (size_t)r * nb, *br = Bk + (size_t)r * nb * 16;
        for (int b = 0; b < nb; b++) {
            float scl = ldexpf(1.0f, (int)sr[b] - 127); const uint8_t *q = br + (size_t)b * 16;
            for (int j = 0; j < 16; j++) { w[2 * j] = FP4T[q[j] & 15] * scl; w[2 * j + 1] = FP4T[q[j] >> 4] * scl; }
            for (int t = 0; t < n; t++) { const float *x = X + (size_t)t * cols + (size_t)b * 32; float s = 0; for (int j = 0; j < 32; j++) s += w[j] * x[j]; acc[t] += s; }
        }
        for (int t = 0; t < n; t++) Y[(size_t)t * rows + r] = acc[t];
    }
}
static void gptoss_moe(Eng *E, int l, const uint8_t *router, int n) {
    Cfg *c = &E->c; int hid = c->hid, NE = c->n_exp, K = c->top_k, I = c->minter; float lim = (float)c->swlim, al = (float)c->swalpha;
    const float *rb = E->resl + (size_t)l * c->lres + c->r_rbias;
    float *lgt = malloc(4 * (size_t)n * NE), *wt = malloc(4 * (size_t)n * K); int *sel = malloc(sizeof(int) * (size_t)n * K);
    mv_batch(router, E->XN, lgt, n);
    for (int t = 0; t < n; t++) {
        float *z = lgt + (size_t)t * NE; for (int e = 0; e < NE; e++) z[e] += rb[e];
        for (int j = 0; j < K; j++) { int b = -1; for (int e = 0; e < NE; e++) { int used = 0; for (int q = 0; q < j; q++) if (sel[t * K + q] == e) used = 1;
                if (!used && (b < 0 || z[e] > z[b])) b = e; } sel[t * K + j] = b; }
        float mx = z[sel[t * K]], sum = 0; for (int j = 0; j < K; j++) { wt[t * K + j] = expf(z[sel[t * K + j]] - mx); sum += wt[t * K + j]; }
        for (int j = 0; j < K; j++) wt[t * K + j] /= sum;
    }
    float *Xs = malloc(4 * (size_t)n * hid), *GU = malloc(4 * (size_t)n * 2 * I), *A = malloc(4 * (size_t)n * I), *Ds = malloc(4 * (size_t)n * hid), *ww = malloc(4 * (size_t)n);
    int *rows = malloc(sizeof(int) * (size_t)n); memset(E->O, 0, 4 * (size_t)n * hid);
    for (int e = 0; e < NE; e++) {
        int m = 0; for (int t = 0; t < n; t++) for (int j = 0; j < K; j++) if (sel[t * K + j] == e) { rows[m] = t; ww[m] = wt[t * K + j]; m++; }
        if (!m) continue;
        const uint8_t *eb = expert_get(E, l, e); const uint8_t *sub[4]; for (int i = 0; i < 4; i++) { uint64_t o; memcpy(&o, eb + 8 + 8 * i, 8); sub[i] = eb + o; }
        const float *bgu = (const float *)(sub[1] + 8), *bd = (const float *)(sub[3] + 8);
        for (int i = 0; i < m; i++) memcpy(Xs + (size_t)i * hid, E->XN + (size_t)rows[i] * hid, 4 * (size_t)hid);
        mxf4_mv(sub[0], Xs, GU, m);
        for (int i = 0; i < m; i++) { float *g = GU + (size_t)i * 2 * I, *a = A + (size_t)i * I;
            for (int j = 0; j < I; j++) { float gt = g[2 * j] + bgu[2 * j], up = g[2 * j + 1] + bgu[2 * j + 1];
                if (gt > lim) gt = lim; if (up > lim) up = lim; if (up < -lim) up = -lim;
                a[j] = (up + 1.0f) * (gt / (1.0f + expf(-al * gt))); } }
        mxf4_mv(sub[2], A, Ds, m);
        for (int i = 0; i < m; i++) { float *o = E->O + (size_t)rows[i] * hid; const float *d = Ds + (size_t)i * hid; for (int k = 0; k < hid; k++) o[k] += ww[i] * (d[k] + bd[k]); }
    }
    for (size_t i = 0; i < (size_t)n * hid; i++) E->H[i] += E->O[i];
    free(lgt); free(wt); free(sel); free(Xs); free(GU); free(A); free(Ds); free(ww); free(rows);
}
static void layer_compute('''
rep("static void layer_compute(", FFN)
rep("    if (c->moe) { moe_ffn(E, l, sub[4], n); return; }", "    if (c->gptoss) { gptoss_moe(E, l, sub[4], n); return; }\n    if (c->moe) { moe_ffn(E, l, sub[4], n); return; }")
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("gpt-oss support added to", P)
