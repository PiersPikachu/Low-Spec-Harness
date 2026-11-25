"""make_engine_loader.py - one persistent expert loader for the whole run (replaces the per-layer loader thread; applied last).

A single thread drains a FIFO of jobs {RAM slot, expert, optional VRAM save of the slot's previous expert, optional VRAM source}.
Every RAM slot and VRAM slot carries a ready flag: a job clears it when queued and sets it when its bytes have landed; whoever
needs a slot (CPU expert compute, GPU thread, a later VRAM copy) waits on that flag. Layers no longer wait for the loader at
their end, so reads - including next-layer prefetches (--prefetch [N]: the top N predicted experts per token, default 2) - keep
flowing through attention and routing. Victim slots are never ones still loading. Bytes are the vault's: outputs unchanged."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)

a = s.index("typedef struct { Eng *E; int n, nmiss, async;"); b = s.index("static void xplan_end(XPlan *P)"); b = s.index("\n", b) + 1
NEW = r'''/* ---------- persistent expert loader: one FIFO, runs across layers ---------- */
#define LQN 8192
typedef struct { int slot, id, vsave, vsrc; } LJob;
static LJob g_lq[LQN]; static volatile long g_lq_head, g_lq_tail; static HANDLE g_lq_ev; static Thr g_lq_th; static int g_lq_on;
static volatile long *g_xready, *g_vready;          /* 1 = slot contents valid */
static void wait_flag(volatile long *f) { int k = 0; while (!__atomic_load_n(f, __ATOMIC_ACQUIRE)) { if (++k < 2000) _mm_pause(); else SwitchToThread(); } }
static void lq_thread(void *arg) {
    Eng *E = arg; SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        long h = __atomic_load_n(&g_lq_head, __ATOMIC_ACQUIRE);
        if (h == __atomic_load_n(&g_lq_tail, __ATOMIC_ACQUIRE)) { WaitForSingleObject(g_lq_ev, 2); continue; }
        LJob j = g_lq[h % LQN]; Stream *st = &E->st[E->c.exp0 + j.id]; double t0 = now();
        if (j.vsave >= 0) { if (g_cuMemcpy(E->vbase + (size_t)j.vsave * E->xsz, E->xslot[j.slot], E->xsz, 1)) die("VRAM save failed");
            __atomic_store_n(&g_vready[j.vsave], 1, __ATOMIC_RELEASE); }
        if (j.vsrc >= 0) { wait_flag(&g_vready[j.vsrc]); if (g_cuMemcpy(E->xslot[j.slot], E->vbase + (size_t)j.vsrc * E->xsz, (size_t)st->padded, 2)) die("VRAM copy failed");
            E->vcopy_ms += (now() - t0) * 1e3; }
        else { if (read_stream(&E->vf, E->xslot[j.slot], st->off, (size_t)st->padded, E->io_threads, E->io_chunk)) die("expert read failed");
            E->xread_ms += (now() - t0) * 1e3; }
        __atomic_store_n(&g_xready[j.slot], 1, __ATOMIC_RELEASE); __atomic_store_n(&g_lq_head, h + 1, __ATOMIC_RELEASE);
    }
}
static void lq_start(Eng *E) {
    g_xready = malloc(sizeof(long) * (E->nslots + 1)); for (int i = 0; i < E->nslots; i++) g_xready[i] = 1;
    g_vready = malloc(sizeof(long) * (E->nv + 1)); for (int i = 0; i < E->nv; i++) g_vready[i] = 1;
    g_lq_ev = CreateEventA(NULL, FALSE, FALSE, NULL); thr_start(&g_lq_th, lq_thread, E); g_lq_on = 1;
}
static void lq_push(int slot, int id, int vsave, int vsrc) {
    while (__atomic_load_n(&g_lq_tail, __ATOMIC_ACQUIRE) - __atomic_load_n(&g_lq_head, __ATOMIC_ACQUIRE) >= LQN - 1) SwitchToThread();
    __atomic_store_n(&g_xready[slot], 0, __ATOMIC_RELEASE); if (vsave >= 0) __atomic_store_n(&g_vready[vsave], 0, __ATOMIC_RELEASE);
    long t = g_lq_tail; g_lq[t % LQN] = (LJob){slot, id, vsave, vsrc}; __atomic_store_n(&g_lq_tail, t + 1, __ATOMIC_RELEASE); SetEvent(g_lq_ev);
}
static void lq_drain(void) { while (__atomic_load_n(&g_lq_head, __ATOMIC_ACQUIRE) != __atomic_load_n(&g_lq_tail, __ATOMIC_ACQUIRE)) SwitchToThread(); }

typedef struct { Eng *E; int n, async, ng, ncpu; int id[1024], slot[1024], order[1024], cnt[1024], base[1024], gpu[1024], g[1024]; int *rows; float *ww; } XPlan;
/* least recently used RAM slot that is not needed by this layer and not still loading */
static int slot_victim(Eng *E, int64_t stamp) { int v = -1; for (int q = 0; q < E->nslots; q++) if (E->xtick[q] < stamp && g_xready[q] && (v < 0 || E->xtick[q] < E->xtick[v])) v = q; return v; }
/* free RAM slot v for a new expert; returns the VRAM slot its previous expert is saved to (or -1) */
static int reclaim(Eng *E, int v, int64_t stamp) {
    int old = E->xowner[v]; if (old < 0) return -1; E->xmap[old] = -1;
    if (!E->nv || E->v2map[old] >= 0) return -1;
    int w = vram_victim(E, stamp);
    if (w >= 0 && E->v2owner[w] >= 0 && E->xmap[E->v2owner[w]] < 0 && E->xcnt[E->v2owner[w]] > E->xcnt[old]) w = -1;   /* not worth displacing */
    if (w < 0) return -1;
    if (E->v2owner[w] >= 0) E->v2map[E->v2owner[w]] = -1; E->v2owner[w] = old; E->v2map[old] = w; E->v2tick[w] = stamp; E->vsave++; return w;
}
/* sel/wt: top-k choice and weight per token; groups rows per expert (ascending), classifies RAM / GPU / miss, queues the reads */
static void xplan_begin(Eng *E, int l, const int *sel, const float *wt, int n, int K, XPlan *P) {
    int NE = E->c.n_exp; P->E = E; P->n = 0; P->async = 1; P->ng = 0; P->rows = malloc(sizeof(int) * (size_t)n * K); P->ww = malloc(4 * (size_t)n * K);
    int tot = 0;
    for (int e = 0; e < NE; e++) {
        int m = 0, b = tot; for (int t = 0; t < n; t++) for (int j = 0; j < K; j++) if (sel[t * K + j] == e) { P->rows[tot] = t; P->ww[tot] = wt[t * K + j]; tot++; m++; }
        if (m && P->n < 1024) { P->id[P->n] = l * NE + e; P->cnt[P->n] = m; P->base[P->n] = b; P->n++; }
    }
    int64_t stamp = ++E->tick; int nh = 0, *miss = malloc(sizeof(int) * P->n), nm = 0, gpuok = E->gx && E->gxlay && E->nv;
    for (int i = 0; i < P->n; i++) { P->gpu[i] = 0; if (E->nv) { E->xcnt[P->id[i]]++; int v = E->v2map[P->id[i]]; if (v >= 0) E->v2tick[v] = stamp; } }
    for (int i = 0; i < P->n; i++) { int s = E->xmap[P->id[i]];
        if (s >= 0 && !(gpuok && E->v2map[P->id[i]] >= 0)) { E->xtick[s] = stamp; P->slot[i] = s; P->order[nh++] = i; E->xhit++; }
        else if (gpuok && E->v2map[P->id[i]] >= 0) { P->gpu[i] = 1; P->slot[i] = E->v2map[P->id[i]]; P->g[P->ng++] = i; E->vgpu++; }
        else miss[nm++] = i; }
    P->ncpu = nh + nm;
    for (int k = 0; k < nm; k++) {
        int mi = miss[k], id = P->id[mi], v = slot_victim(E, stamp); P->order[nh + k] = mi;
        if (v < 0) { P->slot[mi] = -1; continue; }                     /* no free slot: fetched synchronously when needed */
        int sv = reclaim(E, v, stamp), vs = (E->nv && E->v2map[id] >= 0) ? E->v2map[id] : -1;
        if (vs >= 0) E->vhit++;
        E->xowner[v] = id; E->xmap[id] = v; E->xtick[v] = stamp; P->slot[mi] = v; E->xmiss++; lq_push(v, id, sv, vs);
    }
    free(miss);
    for (int k = 0; k < E->pf_n; k++) {                                  /* next layer's predicted experts */
        int id = E->pf_ids[k]; if (E->xmap[id] >= 0 || (E->nv && E->v2map[id] >= 0)) continue;
        int v = slot_victim(E, stamp); if (v < 0) break;
        int sv = reclaim(E, v, stamp); E->xowner[v] = id; E->xmap[id] = v; E->xtick[v] = stamp; E->pf_issued++; lq_push(v, id, sv, -1);
    }
}
static const uint8_t *xplan_get(XPlan *P, int i) {
    Eng *E = P->E; int s = P->slot[i];
    if (s < 0) { lq_drain(); return expert_get(E, P->id[i] / E->c.n_exp, P->id[i] % E->c.n_exp); }
    wait_flag(&g_xready[s]); return E->xslot[s];
}
static void xplan_end(XPlan *P) { free(P->rows); free(P->ww); }
'''
s = s[:a] + NEW + s[b:]
rep("    if (E->c.moe) { moe_cache_init(E, budget - E->resident_used); vram_init(E); gpux_init(E); }",
    "    if (E->c.moe) { moe_cache_init(E, budget - E->resident_used); vram_init(E); gpux_init(E); lq_start(E); }")
# synchronous fallback must not take a slot that is loading or reserved for this layer
rep("    int v = 0; for (int i = 1; i < E->nslots; i++) if (E->xtick[i] < E->xtick[v]) v = i;\n    if (E->xowner[v] >= 0) E->xmap[E->xowner[v]] = -1;",
    "    int v = -1; for (int i = 0; i < E->nslots; i++) if ((!g_xready || g_xready[i]) && (v < 0 || E->xtick[i] < E->xtick[v])) v = i;\n"
    "    if (v < 0) v = 0;\n    if (E->xowner[v] >= 0) E->xmap[E->xowner[v]] = -1;")
# GPU thread waits until a VRAM save it depends on has landed
rep("    for (int k = 0; k < P->ng; k++) { int ix = P->g[k], m = P->cnt[ix]; const int *rows = P->rows + P->base[ix]; double t0 = now();",
    "    for (int k = 0; k < P->ng; k++) { int ix = P->g[k], m = P->cnt[ix]; const int *rows = P->rows + P->base[ix]; double t0 = now();\n"
    "        wait_flag(&g_vready[P->slot[ix]]);")
# prefetch count: --prefetch [N] predicted experts per token (default 2 = the two strongest)
rep('        else if (!strcmp(argv[i], "--prefetch")) E->pf_on = 1;',
    '        else if (!strcmp(argv[i], "--prefetch")) { E->pf_on = 2; if (i + 1 < argc && argv[i + 1][0] >= \'1\' && argv[i + 1][0] <= \'9\') E->pf_on = atoi(argv[++i]); }')
rep("            if (E->pf_on && n <= 4) for (int t = 0; t < n; t++) for (int j = 0; j < K && E->pf_n < 64; j++) {",
    "            if (E->pf_on && n <= 4) for (int t = 0; t < n; t++) for (int j = 0; j < K && j < E->pf_on && E->pf_n < 64; j++) {")
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("persistent expert loader added")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
s = s.replace("static const uint8_t *expert_get(Eng *E, int l, int e) {", "static volatile long *g_xready;     /* defined with the loader below */\nstatic const uint8_t *expert_get(Eng *E, int l, int e) {", 1)
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
