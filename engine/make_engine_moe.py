"""make_engine_moe.py - mixture-of-experts support for nvme_engine_u.c (Qwen3-MoE layout, WVAULT 6 from pack_universal_v3.py).

Per layer the stream holds q,k,v,o + router; every expert is its own stream (gate, up, down). After attention the router picks
top_k experts per token (softmax over all experts, optional renormalisation of the chosen weights - Qwen3-MoE); each needed expert
is fetched from an LRU expert cache in RAM (read from the vault with direct I/O on a miss) and applied to just the tokens that
chose it, experts in ascending order as in Hugging Face's Qwen3MoeSparseMoeBlock. Pinned RAM goes first to the attention streams
and lm_head; whatever --resident-mb is left becomes the expert cache. MOESTATS reports cache hits, misses and read time."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt}, got {n}: {old[:90]!r}"; s = s.replace(old, new)

rep("rl3, supported, codec; double rfactor, rlow, rhigh, rorig; } Cfg;", "rl3, supported, codec; double rfactor, rlow, rhigh, rorig; int moe, n_exp, top_k, norm_topk, exp0, minter; } Cfg;")
rep('else if (!strcmp(k, "codec")) c->codec = (int)v; else return 0;',
    'else if (!strcmp(k, "codec")) c->codec = (int)v; else if (!strcmp(k, "moe")) c->moe = (int)v; else if (!strcmp(k, "n_experts")) c->n_exp = (int)v;\n'
    '    else if (!strcmp(k, "top_k")) c->top_k = (int)v; else if (!strcmp(k, "norm_topk")) c->norm_topk = (int)v; else if (!strcmp(k, "expert_stream0")) c->exp0 = (int)v;\n'
    '    else if (!strcmp(k, "moe_inter")) c->minter = (int)v; else return 0;')
rep("    uint8_t *resident[128]; int first_streamed;", "    uint8_t **resident; int first_streamed;")
rep("    void (*gfwd)(Eng *, const int *, int, int);\n};", "    void (*gfwd)(Eng *, const int *, int, int);\n"
    "    uint8_t **xslot; int *xowner, *xmap; int64_t *xtick, tick; int nslots; size_t xsz; long long xhit, xmiss; double xread_ms;\n};")
rep("E->nst = 0; E->st = calloc(128, sizeof(Stream));", "E->nst = 0; E->st = calloc(70000, sizeof(Stream));")
rep("if (idx < 128) { E->st[idx].off", "if (idx < 70000) { E->st[idx].off")
rep("if (E->ver != 3 && E->ver != 4) {", "if (E->ver != 3 && E->ver != 4 && E->ver != 6) {")
rep("    E->lm_s = E->c.tie ? E->c.nl : E->c.nl + 1;\n", "    E->lm_s = E->c.tie ? E->c.nl : E->c.nl + 1; E->resident = calloc((size_t)E->nst + 1, sizeof(uint8_t *));\n"
    "    if (E->c.moe) lg(\"[vault] mixture of experts: %d experts per layer, top %d, expert size %d, expert streams from %d\", E->c.n_exp, E->c.top_k, E->c.minter, E->c.exp0);\n")

CACHE = r'''/* ---------- MoE expert cache: LRU over fixed slots, filled with direct I/O ---------- */
static void moe_cache_init(Eng *E, double budget) {
    Cfg *c = &E->c; int nx = c->nl * c->n_exp; E->xsz = 0;
    for (int i = 0; i < nx; i++) if (E->st[c->exp0 + i].padded > E->xsz) E->xsz = (size_t)E->st[c->exp0 + i].padded;
    int ns = (int)(budget / (double)E->xsz); if (ns < c->top_k) ns = c->top_k; if (ns > nx) ns = nx;
    E->nslots = ns; E->xslot = malloc(sizeof(uint8_t *) * ns); E->xowner = malloc(sizeof(int) * ns); E->xtick = calloc(ns, sizeof(int64_t));
    E->xmap = malloc(sizeof(int) * nx); for (int i = 0; i < nx; i++) E->xmap[i] = -1;
    for (int i = 0; i < ns; i++) { E->xslot[i] = aalloc(E->xsz); if (!E->xslot[i]) die("cannot allocate expert cache"); E->xowner[i] = -1; }
    lg("[moe] expert cache: %d slots x %.1f MB = %.0f MB (%.1f%% of %d experts)", ns, E->xsz / 1e6, ns * (double)E->xsz / 1e6, 100.0 * ns / nx, nx);
}
static const uint8_t *expert_get(Eng *E, int l, int e) {
    int id = l * E->c.n_exp + e, s = E->xmap[id];
    if (s >= 0) { E->xhit++; E->xtick[s] = ++E->tick; return E->xslot[s]; }
    int v = 0; for (int i = 1; i < E->nslots; i++) if (E->xtick[i] < E->xtick[v]) v = i;
    if (E->xowner[v] >= 0) E->xmap[E->xowner[v]] = -1;
    Stream *st = &E->st[E->c.exp0 + id]; double t0 = now();
    if (read_stream(&E->vf, E->xslot[v], st->off, (size_t)st->padded, E->io_threads, E->io_chunk)) die("expert read failed");
    E->xread_ms += (now() - t0) * 1e3; E->xmiss++; E->xowner[v] = id; E->xmap[id] = v; E->xtick[v] = ++E->tick; return E->xslot[v];
}
/* pin streams in RAM'''
rep("/* pin streams in RAM", CACHE)
rep('    if (E->nst > 120) die("too many streams");', '    if ((E->c.moe ? E->c.exp0 : E->nst) > 120) die("too many streams");\n'
    '    if (E->c.moe && E->gpu_layers) die("MoE vaults run on the CPU engine only (no --gpu-layers yet)");')
rep("E->bufsz = 0; for (int i = 0; i < E->nst; i++)", "E->bufsz = 0; for (int i = 0; i < (E->c.moe ? E->c.exp0 : E->nst); i++)")
rep('E->bufs[0] = aalloc(E->bufsz); E->bufs[1] = aalloc(E->bufsz); if (!E->bufs[0] || !E->bufs[1]) die("cannot allocate stream buffers");',
    'E->bufs[0] = aalloc(E->bufsz); E->bufs[1] = aalloc(E->bufsz); if (!E->bufs[0] || !E->bufs[1]) die("cannot allocate stream buffers");\n'
    '    if (E->c.moe) moe_cache_init(E, budget - E->resident_used);')

FFN = r'''/* MoE feed-forward for n tokens whose post-attention normed states are in E->XN; adds the result to E->H */
static void moe_ffn(Eng *E, int l, const uint8_t *router, int n) {
    Cfg *c = &E->c; int hid = c->hid, NE = c->n_exp, K = c->top_k, I = c->minter;
    float *lgt = malloc(4 * (size_t)n * NE), *wt = malloc(4 * (size_t)n * K); int *sel = malloc(sizeof(int) * (size_t)n * K);
    mv_batch(router, E->XN, lgt, n);
    for (int t = 0; t < n; t++) {
        float *z = lgt + (size_t)t * NE, mx = z[0], sum = 0; for (int e = 1; e < NE; e++) if (z[e] > mx) mx = z[e];
        for (int e = 0; e < NE; e++) { z[e] = expf(z[e] - mx); sum += z[e]; }
        for (int e = 0; e < NE; e++) z[e] /= sum;                                     /* softmax over all experts (fp32) */
        float ws = 0;
        for (int j = 0; j < K; j++) { int b = -1; for (int e = 0; e < NE; e++) { int used = 0; for (int q = 0; q < j; q++) if (sel[t * K + q] == e) used = 1;
                if (!used && (b < 0 || z[e] > z[b])) b = e; } sel[t * K + j] = b; wt[t * K + j] = z[b]; ws += z[b]; }
        if (c->norm_topk) for (int j = 0; j < K; j++) wt[t * K + j] /= ws;
    }
    { static FILE *tr = NULL; static int ti = 0; if (!ti) { const char *p = getenv("NVME_MOE_TRACE"); if (p) tr = fopen(p, "ab"); ti = 1; }
      if (tr) for (int t = 0; t < n; t++) { int16_t rec[2 + 16]; rec[0] = (int16_t)l; rec[1] = (int16_t)(E->pos + t);
          for (int j = 0; j < K && j < 16; j++) rec[2 + j] = (int16_t)sel[t * K + j]; fwrite(rec, 2, 2 + (K < 16 ? K : 16), tr); fflush(tr); } }
    float *Xs = malloc(4 * (size_t)n * hid), *Gs = malloc(4 * (size_t)n * I), *Us = malloc(4 * (size_t)n * I), *Ds = malloc(4 * (size_t)n * hid), *ww = malloc(4 * (size_t)n);
    int *rows = malloc(sizeof(int) * (size_t)n); memset(E->O, 0, 4 * (size_t)n * hid);
    for (int e = 0; e < NE; e++) {
        int m = 0; for (int t = 0; t < n; t++) for (int j = 0; j < K; j++) if (sel[t * K + j] == e) { rows[m] = t; ww[m] = wt[t * K + j]; m++; }
        if (!m) continue;
        const uint8_t *eb = expert_get(E, l, e); const uint8_t *sub[3]; for (int i = 0; i < 3; i++) { uint64_t o; memcpy(&o, eb + 8 + 8 * i, 8); sub[i] = eb + o; }
        for (int i = 0; i < m; i++) memcpy(Xs + (size_t)i * hid, E->XN + (size_t)rows[i] * hid, 4 * (size_t)hid);
        mv_batch(sub[0], Xs, Gs, m); mv_batch(sub[1], Xs, Us, m);
        for (size_t i = 0; i < (size_t)m * I; i++) { float g = Gs[i]; Gs[i] = g / (1.0f + expf(-g)) * Us[i]; }
        mv_batch(sub[2], Gs, Ds, m);
        for (int i = 0; i < m; i++) { float *o = E->O + (size_t)rows[i] * hid; const float *d = Ds + (size_t)i * hid; for (int k = 0; k < hid; k++) o[k] += ww[i] * d[k]; }
    }
    for (size_t i = 0; i < (size_t)n * hid; i++) E->H[i] += E->O[i];
    free(lgt); free(wt); free(sel); free(Xs); free(Gs); free(Us); free(Ds); free(ww); free(rows);
}
static void layer_compute('''
rep("static void layer_compute(", FFN)
rep("    float *U = malloc(4 * (size_t)n * inter);\n", "    if (c->moe) { moe_ffn(E, l, sub[4], n); return; }\n    float *U = malloc(4 * (size_t)n * inter);\n")
rep('''        else if (!strncmp(line, "SETPOS", 6)) { int p = -1; sscanf(line, "SETPOS %d", &p);
            if (p < 0 || p > E.pos) printf("ERR setpos %d (pos %d)\\n", p, E.pos); else { E.pos = p; printf("OK %d\\n", p); } fflush(stdout); }''',
    '''        else if (!strncmp(line, "SETPOS", 6)) { int p = -1; sscanf(line, "SETPOS %d", &p);
            if (p < 0 || p > E.pos) printf("ERR setpos %d (pos %d)\\n", p, E.pos); else { E.pos = p; printf("OK %d\\n", p); } fflush(stdout); }
        else if (!strncmp(line, "MOESTATS", 8)) { printf("M %lld %lld %.1f %d\\n", E.xhit, E.xmiss, E.xread_ms, E.nslots); fflush(stdout); }''')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("MoE support added to", P)
