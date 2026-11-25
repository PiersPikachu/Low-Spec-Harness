"""make_engine_moe2.py - overlap expert reads with expert compute (applies after make_engine_moe.py and make_engine_gptoss.py).

Per layer, once routing is known: experts already in the cache are computed first while a loader thread reads the missing ones into
slots reserved up front (LRU among slots not needed by this layer); each missing expert is computed as soon as its read lands.
Every expert writes its outputs to its own buffer and the results are summed into the hidden state in ascending expert order
afterwards, so the arithmetic (and therefore the output) is identical to the sequential version. Falls back to the sequential path
when a batch needs more experts than the cache can hold at once."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt}, got {n}: {old[:90]!r}"; s = s.replace(old, new)

PLAN = r'''/* ---------- expert plan: hits computed while misses load ---------- */
typedef struct { Eng *E; int n, nmiss, async; int id[1024], slot[1024], order[1024], cnt[1024], base[1024]; volatile long ready[1024];
                 int *rows; float *ww; Thr th; } XPlan;
static void xplan_load(void *a) {
    XPlan *P = a; Eng *E = P->E; SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (int k = P->n - P->nmiss; k < P->n; k++) { int i = P->order[k]; Stream *st = &E->st[E->c.exp0 + P->id[i]]; double t0 = now();
        if (read_stream(&E->vf, E->xslot[P->slot[i]], st->off, (size_t)st->padded, E->io_threads, E->io_chunk)) die("expert read failed");
        E->xread_ms += (now() - t0) * 1e3; __atomic_store_n(&P->ready[i], 1, __ATOMIC_RELEASE); }
}
/* sel/wt: top-k choice and weight per token; fills the plan (experts ascending, rows and weights grouped per expert) and starts the loader */
static void xplan_begin(Eng *E, int l, const int *sel, const float *wt, int n, int K, XPlan *P) {
    int NE = E->c.n_exp; P->E = E; P->n = 0; P->nmiss = 0; P->async = 0; P->rows = malloc(sizeof(int) * (size_t)n * K); P->ww = malloc(4 * (size_t)n * K);
    int tot = 0;
    for (int e = 0; e < NE; e++) {
        int m = 0, b = tot; for (int t = 0; t < n; t++) for (int j = 0; j < K; j++) if (sel[t * K + j] == e) { P->rows[tot] = t; P->ww[tot] = wt[t * K + j]; tot++; m++; }
        if (m && P->n < 1024) { P->id[P->n] = l * NE + e; P->cnt[P->n] = m; P->base[P->n] = b; P->n++; }
    }
    int64_t stamp = ++E->tick; int nh = 0, *miss = malloc(sizeof(int) * P->n), nm = 0;
    for (int i = 0; i < P->n; i++) { int s = E->xmap[P->id[i]]; if (s >= 0) { E->xtick[s] = stamp; P->slot[i] = s; P->ready[i] = 1; P->order[nh++] = i; E->xhit++; } else { miss[nm++] = i; P->ready[i] = 0; } }
    int ok = (nm <= E->nslots - nh);
    for (int k = 0; k < nm && ok; k++) {
        int v = -1; for (int q = 0; q < E->nslots; q++) if (E->xtick[q] < stamp && (v < 0 || E->xtick[q] < E->xtick[v])) v = q;
        if (v < 0) { ok = 0; break; }
        if (E->xowner[v] >= 0) E->xmap[E->xowner[v]] = -1;
        E->xowner[v] = P->id[miss[k]]; E->xmap[P->id[miss[k]]] = v; E->xtick[v] = stamp; P->slot[miss[k]] = v; P->order[nh + k] = miss[k]; E->xmiss++;
    }
    if (!ok) { /* too many at once: undo nothing harmful (reserved slots stay empty-owned), load sequentially */
        for (int k = 0; k < nm; k++) { int i = miss[k]; int s = E->xmap[P->id[i]]; if (s >= 0 && E->xowner[s] == P->id[i] && !P->ready[i]) { E->xowner[s] = -1; E->xmap[P->id[i]] = -1; E->xmiss--; } }
        for (int k = 0; k < nm; k++) P->order[nh + k] = miss[k];
        P->nmiss = nm; P->async = -1; free(miss); return;
    }
    P->nmiss = nm; free(miss);
    if (nm) { P->async = 1; thr_start(&P->th, xplan_load, P); }
}
static const uint8_t *xplan_get(XPlan *P, int i) {
    Eng *E = P->E;
    if (P->async == -1 && !P->ready[i]) { const uint8_t *b = expert_get(E, P->id[i] / E->c.n_exp, P->id[i] % E->c.n_exp); return b; }  /* expert_get counts it */
    while (!__atomic_load_n(&P->ready[i], __ATOMIC_ACQUIRE)) _mm_pause();
    return E->xslot[P->slot[i]];
}
static void xplan_end(XPlan *P) { if (P->async == 1) thr_join(&P->th); free(P->rows); free(P->ww); }
/* pin streams in RAM'''
rep("/* pin streams in RAM", PLAN)

# --- Qwen3-MoE expert loop
rep('''    float *Xs = malloc(4 * (size_t)n * hid), *Gs = malloc(4 * (size_t)n * I), *Us = malloc(4 * (size_t)n * I), *Ds = malloc(4 * (size_t)n * hid), *ww = malloc(4 * (size_t)n);
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
    free(lgt); free(wt); free(sel); free(Xs); free(Gs); free(Us); free(Ds); free(ww); free(rows);''',
'''    float *Xs = malloc(4 * (size_t)n * hid), *Gs = malloc(4 * (size_t)n * I), *Us = malloc(4 * (size_t)n * I), *Dall = malloc(4 * (size_t)n * K * hid);
    XPlan *PL = malloc(sizeof(XPlan)); xplan_begin(E, l, sel, wt, n, K, PL); memset(E->O, 0, 4 * (size_t)n * hid);
    for (int oi = 0; oi < PL->n; oi++) {
        int ix = PL->order[oi], m = PL->cnt[ix]; const int *rows = PL->rows + PL->base[ix];
        const uint8_t *eb = xplan_get(PL, ix); const uint8_t *sub[3]; for (int i = 0; i < 3; i++) { uint64_t o; memcpy(&o, eb + 8 + 8 * i, 8); sub[i] = eb + o; }
        for (int i = 0; i < m; i++) memcpy(Xs + (size_t)i * hid, E->XN + (size_t)rows[i] * hid, 4 * (size_t)hid);
        mv_batch(sub[0], Xs, Gs, m); mv_batch(sub[1], Xs, Us, m);
        for (size_t i = 0; i < (size_t)m * I; i++) { float g = Gs[i]; Gs[i] = g / (1.0f + expf(-g)) * Us[i]; }
        mv_batch(sub[2], Gs, Dall + (size_t)PL->base[ix] * hid, m);
    }
    for (int ix = 0; ix < PL->n; ix++) for (int i = 0; i < PL->cnt[ix]; i++) {       /* sum in ascending expert order: same arithmetic as before */
        int q = PL->base[ix] + i; float *o = E->O + (size_t)PL->rows[q] * hid; const float *d = Dall + (size_t)q * hid; float w = PL->ww[q];
        for (int k = 0; k < hid; k++) o[k] += w * d[k]; }
    xplan_end(PL); free(PL);
    for (size_t i = 0; i < (size_t)n * hid; i++) E->H[i] += E->O[i];
    free(lgt); free(wt); free(sel); free(Xs); free(Gs); free(Us); free(Dall);''')

# --- gpt-oss expert loop
rep('''    float *Xs = malloc(4 * (size_t)n * hid), *GU = malloc(4 * (size_t)n * 2 * I), *A = malloc(4 * (size_t)n * I), *Ds = malloc(4 * (size_t)n * hid), *ww = malloc(4 * (size_t)n);
    int *rows = malloc(sizeof(int) * (size_t)n); memset(E->O, 0, 4 * (size_t)n * hid);
    for (int e = 0; e < NE; e++) {
        int m = 0; for (int t = 0; t < n; t++) for (int j = 0; j < K; j++) if (sel[t * K + j] == e) { rows[m] = t; ww[m] = wt[t * K + j]; m++; }
        if (!m) continue;
        const uint8_t *eb = expert_get(E, l, e); const uint8_t *sub[4];''',
'''    float *Xs = malloc(4 * (size_t)n * hid), *GU = malloc(4 * (size_t)n * 2 * I), *A = malloc(4 * (size_t)n * I), *Dall = malloc(4 * (size_t)n * K * hid);
    XPlan *PL = malloc(sizeof(XPlan)); xplan_begin(E, l, sel, wt, n, K, PL); memset(E->O, 0, 4 * (size_t)n * hid); const float *bdv[1024];
    for (int oi = 0; oi < PL->n; oi++) {
        int ix = PL->order[oi], m = PL->cnt[ix]; const int *rows = PL->rows + PL->base[ix]; float *Ds = Dall + (size_t)PL->base[ix] * hid;
        const uint8_t *eb = xplan_get(PL, ix); const uint8_t *sub[4];''')
rep('''        mxf4_mv(sub[2], A, Ds, m);
        for (int i = 0; i < m; i++) { float *o = E->O + (size_t)rows[i] * hid; const float *d = Ds + (size_t)i * hid; for (int k = 0; k < hid; k++) o[k] += ww[i] * (d[k] + bd[k]); }
    }
    for (size_t i = 0; i < (size_t)n * hid; i++) E->H[i] += E->O[i];
    free(lgt); free(wt); free(sel); free(Xs); free(GU); free(A); free(Ds); free(ww); free(rows);''',
'''        mxf4_mv(sub[2], A, Ds, m); bdv[ix] = bd;
    }
    for (int ix = 0; ix < PL->n; ix++) for (int i = 0; i < PL->cnt[ix]; i++) {       /* ascending expert order, as before */
        int q = PL->base[ix] + i; float *o = E->O + (size_t)PL->rows[q] * hid; const float *d = Dall + (size_t)q * hid, *bd = bdv[ix]; float w = PL->ww[q];
        for (int k = 0; k < hid; k++) o[k] += w * (d[k] + bd[k]); }
    xplan_end(PL); free(PL);
    for (size_t i = 0; i < (size_t)n * hid; i++) E->H[i] += E->O[i];
    free(lgt); free(wt); free(sel); free(Xs); free(GU); free(A); free(Dall);''')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("expert read/compute overlap added to", P)
