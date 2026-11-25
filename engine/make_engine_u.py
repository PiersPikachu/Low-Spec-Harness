"""make_engine_u.py - derive nvme_engine_u.c (metadata-driven layer layout) from the proven nvme_engine.c.
Every edit asserts it matched exactly once, so a changed source fails loudly instead of silently."""
import sys, os
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_base.c")       # base engine source, kept with the harness
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(SRC, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")

def rep(old, new):
    global s
    n = s.count(old)
    assert n == 1, f"expected 1 match, got {n}: {old[:70]!r}"
    s = s.replace(old, new)

def cut(start, end, new):
    global s
    a = s.index(start); b = s.index(end, a)
    assert s.count(start) == 1
    s = s[:a] + new + s[b:]

# ---- config: layout offsets, rope scaling, support flag
rep("typedef struct { int hid, inter, nl, nh, nkv, hd, vocab, tie, eos, eos2; double eps, theta; } Cfg;",
    "typedef struct { int hid, inter, nl, nh, nkv, hd, vocab, tie, eos, eos2; double eps, theta;\n"
    "    int lres, r_ln_in, r_ln_post, r_qn, r_kn, r_bq, r_bk, r_bv, r_bo, r_bg, r_bu, r_bd, rl3, supported; double rfactor, rlow, rhigh, rorig; } Cfg;")
rep('else if (!strcmp(k, "theta")) c->theta = v; else return 0;',
    'else if (!strcmp(k, "theta")) c->theta = v;\n'
    '    else if (!strcmp(k, "lres")) c->lres = (int)v; else if (!strcmp(k, "r_ln_in")) c->r_ln_in = (int)v; else if (!strcmp(k, "r_ln_post")) c->r_ln_post = (int)v;\n'
    '    else if (!strcmp(k, "r_qn")) c->r_qn = (int)v; else if (!strcmp(k, "r_kn")) c->r_kn = (int)v; else if (!strcmp(k, "r_bq")) c->r_bq = (int)v;\n'
    '    else if (!strcmp(k, "r_bk")) c->r_bk = (int)v; else if (!strcmp(k, "r_bv")) c->r_bv = (int)v; else if (!strcmp(k, "r_bo")) c->r_bo = (int)v;\n'
    '    else if (!strcmp(k, "r_bg")) c->r_bg = (int)v; else if (!strcmp(k, "r_bu")) c->r_bu = (int)v; else if (!strcmp(k, "r_bd")) c->r_bd = (int)v;\n'
    '    else if (!strcmp(k, "rope_llama3")) c->rl3 = (int)v; else if (!strcmp(k, "rope_factor")) c->rfactor = v; else if (!strcmp(k, "rope_low")) c->rlow = v;\n'
    '    else if (!strcmp(k, "rope_high")) c->rhigh = v; else if (!strcmp(k, "rope_orig")) c->rorig = v; else if (!strcmp(k, "supported")) c->supported = (int)v; else return 0;')
rep("Cfg c; int ctx, B,", "Cfg c; int ver, ctx, B,")
rep("*cosT, *sinT, *ln_in, *ln_post, *qn, *kn, *fnrm;", "*cosT, *sinT, *resl, *fnrm;")

# ---- idx: accept v3 (Qwen3 default layout) and v4 (explicit layout)
rep("char p[600], line[600]; memset(&E->c, 0, sizeof E->c);",
    "char p[600], line[600]; memset(&E->c, 0, sizeof E->c); E->ver = 0;\n"
    "    E->c.r_ln_in = E->c.r_ln_post = E->c.r_qn = E->c.r_kn = E->c.r_bq = E->c.r_bk = E->c.r_bv = E->c.r_bo = E->c.r_bg = E->c.r_bu = E->c.r_bd = -1;")
rep('if (!strncmp(line, "WVAULT", 6)) { if (atoi(line + 7) != 3) { fprintf(stderr, "vault format %d is old: pack it again into a new folder (upgrade_v3.bat)\\n", atoi(line + 7)); return -1; } continue; }',
    'if (!strncmp(line, "WVAULT", 6)) { E->ver = atoi(line + 7); if (E->ver != 3 && E->ver != 4) { fprintf(stderr, "vault format %d unknown: repack with pack_universal_v3.py\\n", E->ver); return -1; } continue; }')
rep("fclose(f); E->lm_s = E->c.tie ? E->c.nl : E->c.nl + 1;",
    "fclose(f);\n"
    "    if (E->ver == 3) { Cfg *c = &E->c; c->r_ln_in = 0; c->r_ln_post = c->hid; c->r_qn = 2 * c->hid; c->r_kn = 2 * c->hid + c->hd; c->lres = 2 * c->hid + 2 * c->hd; c->supported = 1; }\n"
    '    if (!E->c.supported || E->c.r_ln_in < 0 || E->c.r_ln_post < 0 || E->c.lres <= 0) { fprintf(stderr, "vault layout not supported by this engine (see engine_u.reasons in vault.json)\\n"); return -1; }\n'
    "    E->lm_s = E->c.tie ? E->c.nl : E->c.nl + 1;")

# ---- model init: per-layer resident block described by the idx, optional llama3 rope scaling
cut("static void eng_init_model(Eng *E, int ctx) {", "static void rmsnorm(",
r'''static void eng_init_model(Eng *E, int ctx) {
    Cfg *c = &E->c; E->ctx = ctx; char p[600]; int hid = c->hid, qd = c->nh * c->hd, kvd = c->nkv * c->hd, B = E->B;
    size_t nres = (size_t)c->nl * c->lres + hid; float *res = malloc(4 * nres);
    path_join(p, sizeof p, E->vdir, "resident.bin"); FILE *f = fopen(p, "rb"); if (!f || fread(res, 4, nres, f) != nres) die("resident.bin missing or wrong size"); fclose(f);
    E->resl = res; E->fnrm = res + (size_t)c->nl * c->lres;
    E->kc = calloc((size_t)c->nl * ctx * kvd, 4); E->vc = calloc((size_t)c->nl * ctx * kvd, 4);
    int half = c->hd / 2; E->cosT = malloc(4 * (size_t)ctx * half); E->sinT = malloc(4 * (size_t)ctx * half);
    for (int i = 0; i < half; i++) {
        double fr = pow(c->theta, -2.0 * i / c->hd);
        if (c->rl3) { double wl = 2.0 * 3.14159265358979323846 / fr, lo = c->rorig / c->rlow, hi = c->rorig / c->rhigh;
            if (wl > lo) fr /= c->rfactor; else if (wl >= hi) { double sm = (c->rorig / wl - c->rlow) / (c->rhigh - c->rlow); fr = (1.0 - sm) * fr / c->rfactor + sm * fr; } }
        for (int pos = 0; pos < ctx; pos++) { double a = pos * fr; E->cosT[(size_t)pos * half + i] = (float)cos(a); E->sinT[(size_t)pos * half + i] = (float)sin(a); }
    }
    E->H = malloc(4 * (size_t)B * hid); E->XN = malloc(4 * (size_t)B * hid); E->Q = malloc(4 * (size_t)B * qd); E->K = malloc(4 * (size_t)B * kvd); E->V = malloc(4 * (size_t)B * kvd);
    E->ATT = malloc(4 * (size_t)B * qd); E->O = malloc(4 * (size_t)B * hid); E->G = malloc(4 * (size_t)B * c->inter); float *U = malloc(4 * (size_t)B * c->inter); E->logits = malloc(4 * (size_t)c->vocab);
    E->t_attn = 0; E->pos = 0; (void)U; free(U);
}
''')

# ---- layer compute: optional q/k norm and optional biases
rep("static void layer_compute(Eng *E, int l, const uint8_t *blob, int n, int pos0) {",
    "static FILE *G_DUMPH = NULL;\n"
    "static void add_bias(float *y, int n, int d, const float *b) { if (!b) return; for (int t = 0; t < n; t++) for (int i = 0; i < d; i++) y[(size_t)t * d + i] += b[i]; }\n"
    "static void layer_compute(Eng *E, int l, const uint8_t *blob, int n, int pos0) {")
rep("const float *lni = E->ln_in + (size_t)l * hid, *lnp = E->ln_post + (size_t)l * hid, *qn = E->qn + (size_t)l * hd, *kn = E->kn + (size_t)l * hd;",
    "const float *rl = E->resl + (size_t)l * c->lres; const float *lni = rl + c->r_ln_in, *lnp = rl + c->r_ln_post;\n"
    "    const float *qn = c->r_qn >= 0 ? rl + c->r_qn : NULL, *kn = c->r_kn >= 0 ? rl + c->r_kn : NULL;\n"
    "    const float *bq = c->r_bq >= 0 ? rl + c->r_bq : NULL, *bk = c->r_bk >= 0 ? rl + c->r_bk : NULL, *bv = c->r_bv >= 0 ? rl + c->r_bv : NULL, *bo = c->r_bo >= 0 ? rl + c->r_bo : NULL;\n"
    "    const float *bg = c->r_bg >= 0 ? rl + c->r_bg : NULL, *bu = c->r_bu >= 0 ? rl + c->r_bu : NULL, *bd = c->r_bd >= 0 ? rl + c->r_bd : NULL;")
rep("mv_batch(sub[0], E->XN, E->Q, n); mv_batch(sub[1], E->XN, E->K, n); mv_batch(sub[2], E->XN, E->V, n);",
    "mv_batch(sub[0], E->XN, E->Q, n); mv_batch(sub[1], E->XN, E->K, n); mv_batch(sub[2], E->XN, E->V, n);\n"
    "    add_bias(E->Q, n, qd, bq); add_bias(E->K, n, kvd, bk); add_bias(E->V, n, kvd, bv);")
rep("for (int h = 0; h < nh; h++) rmsnorm(q + h * hd, q + h * hd, qn, hd, eps);", "if (qn) for (int h = 0; h < nh; h++) rmsnorm(q + h * hd, q + h * hd, qn, hd, eps);")
rep("for (int h = 0; h < nkv; h++) rmsnorm(k + h * hd, k + h * hd, kn, hd, eps);", "if (kn) for (int h = 0; h < nkv; h++) rmsnorm(k + h * hd, k + h * hd, kn, hd, eps);")
rep("mv_batch(sub[3], E->ATT, E->O, n); for (size_t i", "mv_batch(sub[3], E->ATT, E->O, n); add_bias(E->O, n, hid, bo); for (size_t i")
rep("mv_batch(sub[4], E->XN, E->G, n); mv_batch(sub[5], E->XN, U, n);", "mv_batch(sub[4], E->XN, E->G, n); mv_batch(sub[5], E->XN, U, n); add_bias(E->G, n, inter, bg); add_bias(U, n, inter, bu);")
rep("free(U); mv_batch(sub[6], E->G, E->O, n); for (size_t i", "free(U); mv_batch(sub[6], E->G, E->O, n); add_bias(E->O, n, hid, bd); for (size_t i")

# ---- debug: last-token hidden state after every layer, full-vocab LOGITS
rep("if (i < c->nl) layer_compute(E, i, blob, n, pos0);",
    "if (i < c->nl) { layer_compute(E, i, blob, n, pos0); if (G_DUMPH) fwrite(E->H + (size_t)(n - 1) * c->hid, 4, (size_t)c->hid, G_DUMPH); }")
rep('printf("L"); for (int i = 0; i < E.c.vocab && i < 4096; i++) printf(" %.7g", E.logits[i]);',
    'if (G_DUMPH) fflush(G_DUMPH); printf("L"); for (int i = 0; i < E.c.vocab; i++) printf(" %.9g", E.logits[i]);')
rep('G_LOG = fopen(logp, "a"); double t0 = now(); if (eng_open(&E, argv[2])) return 1;',
    'G_LOG = fopen(logp, "a"); { const char *dh = getenv("NVME_DUMP_H"); if (dh) G_DUMPH = fopen(dh, "wb"); } double t0 = now(); if (eng_open(&E, argv[2])) return 1;')
open(DST, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("wrote", DST, len(s.splitlines()), "lines")

import subprocess
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_split.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_rans.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_patch.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_spec.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_specp.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_moe.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_gptoss.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_moe2.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_vram.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_gpux.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_pred.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_prefetch.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_loader.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_trace2.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_copyq.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_mxf2.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_stop.py')])
subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_engine_kv.py')])
