"""make_engine_kv.py - KV cache per layer: sliding-window layers keep a ring, optional fp16 storage (--kv16), RAM reserved for it.

  ring    gpt-oss 'sliding_attention' layers only attend to the last `sliding_window` (128) positions, so they keep a ring of
          sliding_window + KV_MARGIN slots instead of ctx (exact: the same values are read). SETPOS may roll back at most
          KV_MARGIN positions (speculative / prefix reuse); a deeper rollback answers 'ERR ring' and the harness re-reads the prompt.
          NVME_KV_MARGIN overrides the margin (tests use a small one to force wrap-around).
  --kv16  K and V stored as IEEE fp16 (F16C), half the memory and half the bytes attention reads; rounding ~1e-3 relative, so
          logits move slightly (not bit-exact). Dot products and the V sum vectorised (AVX2) in this mode. fp32 mode unchanged.
  reserve the full KV size is subtracted from the RAM budget for pinned weights / experts (pages are touched as context grows)."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:80], s.count(old)); s = s.replace(old, new)

rep("    float *kc, *vc, *cosT, *sinT, *resl, *fnrm;", "    float *kc, *vc, *cosT, *sinT, *resl, *fnrm; float **k32, **v32; uint16_t **k16, **v16; int *kcap; int kv16, ctx_hint, has_ring, kv_margin;")
KVF = r'''
static double kv_total_bytes(Eng *E, int ctx) {          /* what eng_init_model will allocate for K and V */
    Cfg *c = &E->c; double kvd = (double)c->nkv * c->hd, tot = 0; int mg = E->kv_margin;
    for (int l = 0; l < c->nl; l++) { int cap = (c->gptoss && c->ltype[l] && c->swin > 0 && c->swin + mg < ctx) ? c->swin + mg : ctx; tot += 2.0 * cap * kvd * (E->kv16 ? 2 : 4); }
    return tot;
}
/* pin streams in RAM'''
rep("\n/* pin streams in RAM", KVF, 1)
rep('''        double mt, ma, mw, mp; mem_info(&mt, &ma, &mw, &mp); double lim = (ma - 1500.0) * 1048576.0; if (lim < 0) lim = 0;''',
    '''        double mt, ma, mw, mp; mem_info(&mt, &ma, &mw, &mp); double lim = (ma - 1500.0) * 1048576.0 - kv_total_bytes(E, E->ctx_hint > 0 ? E->ctx_hint : 0); if (lim < 0) lim = 0;''')
rep('''    E->kc = calloc((size_t)c->nl * ctx * kvd, 4); E->vc = calloc((size_t)c->nl * ctx * kvd, 4);''',
'''    E->kcap = malloc(sizeof(int) * c->nl); E->k32 = calloc(c->nl, sizeof(float *)); E->v32 = calloc(c->nl, sizeof(float *));
    E->k16 = calloc(c->nl, sizeof(uint16_t *)); E->v16 = calloc(c->nl, sizeof(uint16_t *)); E->has_ring = 0;
    for (int l = 0; l < c->nl; l++) {
        int cap = (c->gptoss && c->ltype[l] && c->swin > 0 && c->swin + E->kv_margin < ctx) ? c->swin + E->kv_margin : ctx; E->kcap[l] = cap;
        if (cap < ctx) E->has_ring = 1;
        if (E->kv16) { E->k16[l] = calloc((size_t)cap * kvd, 2); E->v16[l] = calloc((size_t)cap * kvd, 2); if (!E->k16[l] || !E->v16[l]) die("cannot allocate the KV cache"); }
        else { E->k32[l] = calloc((size_t)cap * kvd, 4); E->v32[l] = calloc((size_t)cap * kvd, 4); if (!E->k32[l] || !E->v32[l]) die("cannot allocate the KV cache"); }
    }
    lg("[kv] %s, %.0f MB for ctx %d%s", E->kv16 ? "fp16" : "fp32", kv_total_bytes(E, ctx) / 1e6, ctx, E->has_ring ? " (sliding-window layers use a ring)" : "");''')
# attention
rep('''    double ta = now(); float *kcl = E->kc + (size_t)l * E->ctx * kvd, *vcl = E->vc + (size_t)l * E->ctx * kvd; int grp = nh / nkv; float scale = 1.0f / sqrtf((float)hd);''',
    '''    double ta = now(); float *kcl = E->k32[l], *vcl = E->v32[l]; uint16_t *kh16 = E->k16[l], *vh16 = E->v16[l]; int cap = E->kcap[l], ring = cap < E->ctx;
    int grp = nh / nkv; float scale = 1.0f / sqrtf((float)hd);''')
rep('''        memcpy(kcl + (size_t)p * kvd, k, 4 * (size_t)kvd); memcpy(vcl + (size_t)p * kvd, v, 4 * (size_t)kvd);''',
'''        { size_t sl = (size_t)(ring ? p % cap : p) * kvd;
          if (kh16) { for (int d = 0; d < kvd; d += 8) { _mm_storeu_si128((__m128i *)(kh16 + sl + d), _mm256_cvtps_ph(_mm256_loadu_ps(k + d), 0));
                                                      _mm_storeu_si128((__m128i *)(vh16 + sl + d), _mm256_cvtps_ph(_mm256_loadu_ps(v + d), 0)); } }
          else { memcpy(kcl + sl, k, 4 * (size_t)kvd); memcpy(vcl + sl, v, 4 * (size_t)kvd); } }''')
rep('''            for (int tt = t0; tt <= p; tt++) { const float *kk = kcl + (size_t)tt * kvd + kvh * hd; float s = 0; for (int d = 0; d < hd; d++) s += qh[d] * kk[d]; s *= scale; sc[tt] = s; if (s > mx) mx = s; }''',
'''            if (kh16) for (int tt = t0; tt <= p; tt++) {
                const uint16_t *kk = kh16 + (size_t)(ring ? tt % cap : tt) * kvd + kvh * hd; __m256 a = _mm256_setzero_ps();
                for (int d = 0; d < hd; d += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(qh + d), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kk + d))), a);
                __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1)); s4 = _mm_hadd_ps(s4, s4); s4 = _mm_hadd_ps(s4, s4);
                float s = _mm_cvtss_f32(s4) * scale; sc[tt] = s; if (s > mx) mx = s; }
            else for (int tt = t0; tt <= p; tt++) { const float *kk = kcl + (size_t)(ring ? tt % cap : tt) * kvd + kvh * hd; float s = 0; for (int d = 0; d < hd; d++) s += qh[d] * kk[d]; s *= scale; sc[tt] = s; if (s > mx) mx = s; }''')
rep('''            for (int tt = t0; tt <= p; tt++) { const float *vv = vcl + (size_t)tt * kvd + kvh * hd; float w = sc[tt] * inv; for (int d = 0; d < hd; d++) o[d] += w * vv[d]; }''',
'''            if (vh16) for (int tt = t0; tt <= p; tt++) {
                const uint16_t *vv = vh16 + (size_t)(ring ? tt % cap : tt) * kvd + kvh * hd; __m256 w = _mm256_set1_ps(sc[tt] * inv);
                for (int d = 0; d < hd; d += 8) _mm256_storeu_ps(o + d, _mm256_fmadd_ps(w, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vv + d))), _mm256_loadu_ps(o + d))); }
            else for (int tt = t0; tt <= p; tt++) { const float *vv = vcl + (size_t)(ring ? tt % cap : tt) * kvd + kvh * hd; float w = sc[tt] * inv; for (int d = 0; d < hd; d++) o[d] += w * vv[d]; }''')
# SETPOS: a sliding ring cannot roll back further than its margin
rep('''            if (p < 0 || p > E.pos) printf("ERR setpos %d (pos %d)\\n", p, E.pos); else { E.pos = p; printf("OK %d\\n", p); } fflush(stdout); }''',
    '''            if (p < 0 || p > E.pos) printf("ERR setpos %d (pos %d)\\n", p, E.pos);
            else if (E.has_ring && E.pos - p > E.kv_margin) printf("ERR ring %d (pos %d, rollback limit %d)\\n", p, E.pos, E.kv_margin);
            else { E.pos = p; printf("OK %d\\n", p); } fflush(stdout); }''')
# options and the RAM reserve: ctx known before the buffers are sized
rep('''        else if (!strcmp(argv[i], "--vram-mb") && i + 1 < argc) E->vram_mb = atof(argv[++i]);''',
    '''        else if (!strcmp(argv[i], "--vram-mb") && i + 1 < argc) E->vram_mb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--kv16")) E->kv16 = 1;''')
rep("if (eng_open(&E, argv[2])) return 1; eng_buffers(&E, E.resident_mb);",
    "if (eng_open(&E, argv[2])) return 1; { const char *km = getenv(\"NVME_KV_MARGIN\"); E.kv_margin = km ? atoi(km) : 1024; } E.ctx_hint = ctx; eng_buffers(&E, E.resident_mb);", cnt=2)
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("engine: per-layer KV (ring for sliding layers), --kv16, RAM reserve")
