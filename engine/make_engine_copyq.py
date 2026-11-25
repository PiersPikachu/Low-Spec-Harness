"""make_engine_copyq.py - VRAM saves on their own copy thread (applies after make_engine_loader.py / make_engine_trace2.py).

Before: the SSD loader itself copied a reclaimed RAM expert into VRAM (~1.1 ms over PCIe) before reading the new expert into that
slot, so every save delayed the bottleneck thread. Now a save is queued to a separate copy thread the moment the slot is
reclaimed; the slot carries a 'saving' flag, and the loader waits on it only when it reaches that slot's read (by then the save
has normally finished, since the loader runs behind). Saving slots are never picked as victims. Bytes unchanged."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)

rep("static volatile long *g_xready, *g_vready;          /* 1 = slot contents valid */",
    "static volatile long *g_xready, *g_vready;          /* 1 = slot contents valid */\n"
    "typedef struct { int slot, vsave; } CJob;\n"
    "static CJob g_cq[LQN]; static volatile long g_cq_head, g_cq_tail; static HANDLE g_cq_ev; static Thr g_cq_th; static volatile long *g_xsaving; static double g_save_ms;")
# loader: wait for a pending save of the slot before overwriting it; saves no longer done here
rep("        LJob j = g_lq[h % LQN]; Stream *st = &E->st[E->c.exp0 + j.id]; double t0 = now();\n",
    "        LJob j = g_lq[h % LQN]; Stream *st = &E->st[E->c.exp0 + j.id];\n"
    "        { int k = 0; while (__atomic_load_n(&g_xsaving[j.slot], __ATOMIC_ACQUIRE)) { if (++k < 2000) _mm_pause(); else SwitchToThread(); } }\n"
    "        double t0 = now();\n")
CQ = r'''static void cq_thread(void *arg) {
    Eng *E = arg; SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        long h = __atomic_load_n(&g_cq_head, __ATOMIC_ACQUIRE);
        if (h == __atomic_load_n(&g_cq_tail, __ATOMIC_ACQUIRE)) { WaitForSingleObject(g_cq_ev, 2); continue; }
        CJob j = g_cq[h % LQN]; double t0 = now();
        if (g_cuMemcpy(E->vbase + (size_t)j.vsave * E->xsz, E->xslot[j.slot], E->xsz, 1)) die("VRAM save failed");
        g_save_ms += (now() - t0) * 1e3;
        __atomic_store_n(&g_vready[j.vsave], 1, __ATOMIC_RELEASE); __atomic_store_n(&g_xsaving[j.slot], 0, __ATOMIC_RELEASE);
        __atomic_store_n(&g_cq_head, h + 1, __ATOMIC_RELEASE);
    }
}
static void lq_start(Eng *E) {'''
rep("static void lq_start(Eng *E) {", CQ)
rep("    g_lq_ev = CreateEventA(NULL, FALSE, FALSE, NULL); thr_start(&g_lq_th, lq_thread, E); g_lq_on = 1;",
    "    g_xsaving = calloc(E->nslots + 1, sizeof(long)); g_cq_ev = CreateEventA(NULL, FALSE, FALSE, NULL); thr_start(&g_cq_th, cq_thread, E);\n"
    "    g_lq_ev = CreateEventA(NULL, FALSE, FALSE, NULL); thr_start(&g_lq_th, lq_thread, E); g_lq_on = 1;")
# push: the save goes to the copy queue; the read job carries no save
rep("    __atomic_store_n(&g_xready[slot], 0, __ATOMIC_RELEASE); if (vsave >= 0) __atomic_store_n(&g_vready[vsave], 0, __ATOMIC_RELEASE);\n"
    "    long t = g_lq_tail; g_lq[t % LQN] = (LJob){slot, id, vsave, vsrc};",
    "    __atomic_store_n(&g_xready[slot], 0, __ATOMIC_RELEASE);\n"
    "    if (vsave >= 0) { __atomic_store_n(&g_vready[vsave], 0, __ATOMIC_RELEASE); __atomic_store_n(&g_xsaving[slot], 1, __ATOMIC_RELEASE);\n"
    "        while (__atomic_load_n(&g_cq_tail, __ATOMIC_ACQUIRE) - __atomic_load_n(&g_cq_head, __ATOMIC_ACQUIRE) >= LQN - 1) SwitchToThread();\n"
    "        long c = g_cq_tail; g_cq[c % LQN] = (CJob){slot, vsave}; __atomic_store_n(&g_cq_tail, c + 1, __ATOMIC_RELEASE); SetEvent(g_cq_ev); }\n"
    "    long t = g_lq_tail; g_lq[t % LQN] = (LJob){slot, id, -1, vsrc};")
rep("static void lq_drain(void) { while (__atomic_load_n(&g_lq_head, __ATOMIC_ACQUIRE) != __atomic_load_n(&g_lq_tail, __ATOMIC_ACQUIRE)) SwitchToThread(); }",
    "static void lq_drain(void) { while (__atomic_load_n(&g_lq_head, __ATOMIC_ACQUIRE) != __atomic_load_n(&g_lq_tail, __ATOMIC_ACQUIRE)\n"
    "                                 || __atomic_load_n(&g_cq_head, __ATOMIC_ACQUIRE) != __atomic_load_n(&g_cq_tail, __ATOMIC_ACQUIRE)) SwitchToThread(); }")
rep("if (E->xtick[q] < stamp && g_xready[q] && (v < 0 || E->xtick[q] < E->xtick[v])) v = q; return v; }",
    "if (E->xtick[q] < stamp && g_xready[q] && !g_xsaving[q] && (v < 0 || E->xtick[q] < E->xtick[v])) v = q; return v; }")
# the synchronous fallback may not take a slot that is being saved either
rep("if ((!g_xready || g_xready[i]) && (v < 0 || E->xtick[i] < E->xtick[v])) v = i;",
    "if ((!g_xready || (g_xready[i] && !g_xsaving[i])) && (v < 0 || E->xtick[i] < E->xtick[v])) v = i;")
rep("static volatile long *g_xready;     /* defined with the loader below */", "static volatile long *g_xready, *g_xsaving;     /* defined with the loader below */")
# report save time in MOESTATS as the vcopy field when the copy thread is in use
rep('E.vsave, E.vcopy_ms, E.vgpu', 'E.vsave, E.vcopy_ms + g_save_ms, E.vgpu')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("VRAM copy queue added")
