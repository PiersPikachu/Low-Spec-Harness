"""make_engine_gpux.py - compute VRAM-resident gpt-oss experts ON the GPU (gpu_experts.dll) instead of copying them back to RAM.
Applies after make_engine_vram.py. Per layer: experts in RAM run on the CPU, experts only in VRAM run on the GPU in a parallel thread,
missing ones are read from the SSD into RAM; all outputs are summed in ascending expert order. Falls back to the copy path when the
DLL is missing (or NVME_NO_GPUX is set), and for Qwen-style experts (no GPU kernel for them yet)."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt}, got {n}: {old[:90]!r}"; s = s.replace(old, new)

rep("    uint8_t *vbase; int nv, *v2owner, *v2map; int64_t *v2tick; long long vhit, vsave; double vram_mb, vcopy_ms;\n};",
    "    uint8_t *vbase; int nv, *v2owner, *v2map; int64_t *v2tick; long long vhit, vsave; double vram_mb, vcopy_ms;\n"
    "    int gx, gxlay, gx_rgu, gx_cgu, gx_rd, gx_cd; uint64_t gxoff[4]; long long vgpu; double gpu_ms;\n};")
GX = r'''typedef int (*gxinit_t)(int, int, int);
typedef int (*gxmxf4_t)(const uint8_t *, uint64_t, uint64_t, uint64_t, uint64_t, int, int, int, int, const float *, int, float, float, float *);
static gxinit_t g_gxinit; static gxmxf4_t g_gxmxf4;
static void gpux_init(Eng *E) {
    E->gx = 0; if (!E->c.gptoss || !E->nv || getenv("NVME_NO_GPUX")) return;
    char p[MAX_PATH]; DWORD n = GetModuleFileNameA(NULL, p, MAX_PATH); while (n && p[n - 1] != '\\' && p[n - 1] != '/') n--; p[n] = 0;
    strcat(p, "gpu_experts.dll"); HMODULE h = LoadLibraryA(p); if (!h) h = LoadLibraryA("gpu_experts.dll");
    if (!h) { lg("[moe] gpu_experts.dll not found: VRAM experts are copied back instead of computed on the GPU"); return; }
    g_gxinit = (gxinit_t)GetProcAddress(h, "gx_init"); g_gxmxf4 = (gxmxf4_t)GetProcAddress(h, "gx_mxf4_expert");
    if (!g_gxinit || !g_gxmxf4 || g_gxinit(128, E->c.hid, E->c.minter)) { lg("[moe] gpu_experts.dll failed to initialise"); return; }
    E->gx = 1; lg("[moe] GPU computes VRAM-resident experts (gpu_experts.dll)");
}
/* ---------- expert plan: hits computed while misses load ---------- */'''
rep("/* ---------- expert plan: hits computed while misses load ---------- */", GX)
rep("    if (E->c.moe) { moe_cache_init(E, budget - E->resident_used); vram_init(E); }", "    if (E->c.moe) { moe_cache_init(E, budget - E->resident_used); vram_init(E); gpux_init(E); }")
rep("vsrc[1024], vsave[1024]; volatile long ready[1024];", "vsrc[1024], vsave[1024], gpu[1024], g[1024], ng, ncpu; volatile long ready[1024];")
rep("    for (int k = P->n - P->nmiss; k < P->n; k++) { int i = P->order[k];", "    for (int k = P->ncpu - P->nmiss; k < P->ncpu; k++) { int i = P->order[k];")
rep("    for (int i = 0; i < P->n; i++) { P->vsrc[i] = -1; P->vsave[i] = -1;", "    P->ng = 0; for (int i = 0; i < P->n; i++) { P->vsrc[i] = -1; P->vsave[i] = -1; P->gpu[i] = 0;")
rep("E->xhit++; } else { miss[nm++] = i; P->ready[i] = 0; } }",
    "E->xhit++; }\n"
    "        else if (E->gx && E->gxlay && E->nv && E->v2map[P->id[i]] >= 0) { P->gpu[i] = 1; P->slot[i] = E->v2map[P->id[i]]; P->g[P->ng++] = i; P->ready[i] = 1; E->vgpu++; }\n"
    "        else { miss[nm++] = i; P->ready[i] = 0; } }\n    P->ncpu = nh + nm;")
# gpt-oss: GPU thread + CPU loop over ncpu
JOB = r'''typedef struct { Eng *E; XPlan *P; float *Dall; } GxJob;
static void gx_run(void *a) {
    GxJob *J = a; Eng *E = J->E; XPlan *P = J->P; int hid = E->c.hid; float *Xg = malloc(4 * (size_t)128 * hid);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);   /* the GPU waits on this thread: do not let the spinning OpenMP workers starve it */
    for (int k = 0; k < P->ng; k++) { int ix = P->g[k], m = P->cnt[ix]; const int *rows = P->rows + P->base[ix]; double t0 = now();
        for (int i = 0; i < m; i++) memcpy(Xg + (size_t)i * hid, E->XN + (size_t)rows[i] * hid, 4 * (size_t)hid);
        if (g_gxmxf4(E->vbase + (size_t)P->slot[ix] * E->xsz, E->gxoff[0], E->gxoff[1], E->gxoff[2], E->gxoff[3], E->gx_rgu, E->gx_cgu, E->gx_rd, E->gx_cd,
                     Xg, m, (float)E->c.swlim, (float)E->c.swalpha, J->Dall + (size_t)P->base[ix] * hid)) die("GPU expert failed");
        E->gpu_ms += (now() - t0) * 1e3; }
    free(Xg);
}
static void gptoss_moe('''
rep("static void gptoss_moe(", JOB)
rep('''    XPlan *PL = malloc(sizeof(XPlan)); xplan_begin(E, l, sel, wt, n, K, PL); memset(E->O, 0, 4 * (size_t)n * hid); const float *bdv[1024];
    for (int oi = 0; oi < PL->n; oi++) {''',
'''    XPlan *PL = malloc(sizeof(XPlan)); xplan_begin(E, l, sel, wt, n, K, PL); memset(E->O, 0, 4 * (size_t)n * hid); const float *bdv[1024];
    for (int i = 0; i < PL->n; i++) bdv[i] = NULL;
    GxJob gj = {E, PL, Dall}; Thr gth; if (PL->ng) thr_start(&gth, gx_run, &gj);
    for (int oi = 0; oi < PL->ncpu; oi++) {''')
rep('''        const float *bgu = (const float *)(sub[1] + 8), *bd = (const float *)(sub[3] + 8);''',
'''        const float *bgu = (const float *)(sub[1] + 8), *bd = (const float *)(sub[3] + 8);
        if (!E->gxlay) { for (int q = 0; q < 4; q++) E->gxoff[q] = (uint64_t)(sub[q] - eb);
            uint32_t r0, c0, r2, c2; memcpy(&r0, sub[0] + 4, 4); memcpy(&c0, sub[0] + 8, 4); memcpy(&r2, sub[2] + 4, 4); memcpy(&c2, sub[2] + 8, 4);
            E->gx_rgu = (int)r0; E->gx_cgu = (int)c0; E->gx_rd = (int)r2; E->gx_cd = (int)c2; E->gxlay = 1; }''')
rep('''        mxf4_mv(sub[2], A, Ds, m); bdv[ix] = bd;
    }''', '''        mxf4_mv(sub[2], A, Ds, m);   /* add the bias now: the slot may be reused by a later expert of this layer (small-cache path) */
        for (int i = 0; i < m; i++) { float *d = Ds + (size_t)i * hid; for (int k = 0; k < hid; k++) d[k] += bd[k]; }
    }
    if (PL->ng) thr_join(&gth);''')
rep('''const float *d = Dall + (size_t)q * hid, *bd = bdv[ix]; float w = PL->ww[q];
        for (int k = 0; k < hid; k++) o[k] += w * (d[k] + bd[k]); }''',
'''const float *d = Dall + (size_t)q * hid, *bd = bdv[ix]; float w = PL->ww[q];
        if (bd) for (int k = 0; k < hid; k++) o[k] += w * (d[k] + bd[k]); else for (int k = 0; k < hid; k++) o[k] += w * d[k]; }''')
# Qwen path loops over the CPU list too (no GPU experts there, so ncpu == n)
rep("    for (int oi = 0; oi < PL->n; oi++) {\n        int ix = PL->order[oi], m = PL->cnt[ix]; const int *rows = PL->rows + PL->base[ix];\n        const uint8_t *eb = xplan_get(PL, ix); const uint8_t *sub[3];",
    "    for (int oi = 0; oi < PL->ncpu; oi++) {\n        int ix = PL->order[oi], m = PL->cnt[ix]; const int *rows = PL->rows + PL->base[ix];\n        const uint8_t *eb = xplan_get(PL, ix); const uint8_t *sub[3];")
rep('E.xhit, E.xmiss, E.xread_ms, E.nslots, E.vhit, E.nv, E.vsave, E.vcopy_ms);', 'E.xhit, E.xmiss, E.xread_ms, E.nslots, E.vhit, E.nv, E.vsave, E.vcopy_ms, E.vgpu, E.gpu_ms);')
rep('printf("M %lld %lld %.1f %d %lld %d %lld %.1f\\n"', 'printf("M %lld %lld %.1f %d %lld %d %lld %.1f %lld %.1f\\n"')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("GPU expert compute added to", P)

# GPU first: an expert held in VRAM is computed on the GPU even when RAM also has it (the GPU is ~2-20x faster per expert),
# so the hottest experts gravitate to VRAM and the CPU takes the rest
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
old = "for (int i = 0; i < P->n; i++) { int s = E->xmap[P->id[i]]; if (s >= 0) { E->xtick[s] = stamp;"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "for (int i = 0; i < P->n; i++) { int s = E->xmap[P->id[i]]; if (s >= 0 && !(E->gx && E->gxlay && E->nv && E->v2map[P->id[i]] >= 0)) { E->xtick[s] = stamp;")
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("GPU-first placement")

# frequency-aware VRAM tier: count uses per expert; VRAM eviction takes empty slots, then duplicates of RAM, then the least-used
# expert (not the least recent); a reclaimed RAM expert is saved into VRAM only if it has been used at least as often as what it
# would displace - stops one-off experts from churning VRAM (29k saves per 6 prompts on gpt-oss-120b)
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
def rp(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)
rp("long long vhit, vsave; double vram_mb, vcopy_ms;", "long long vhit, vsave; double vram_mb, vcopy_ms; int *xcnt;")
rp("    for (int i = 0; i < nv; i++) E->v2owner[i] = -1; for (int i = 0; i < nx; i++) E->v2map[i] = -1;",
   "    for (int i = 0; i < nv; i++) E->v2owner[i] = -1; for (int i = 0; i < nx; i++) E->v2map[i] = -1; E->xcnt = calloc(nx, sizeof(int));")
rp("        if (E->xmap[o] >= 0 && (d < 0 || E->v2tick[q] < E->v2tick[d])) d = q;\n        if (v < 0 || E->v2tick[q] < E->v2tick[v]) v = q; }",
   "        if (E->xmap[o] >= 0 && (d < 0 || E->v2tick[q] < E->v2tick[d])) d = q;\n"
   "        if (v < 0 || E->xcnt[o] < E->xcnt[E->v2owner[v]] || (E->xcnt[o] == E->xcnt[E->v2owner[v]] && E->v2tick[q] < E->v2tick[v])) v = q; }")
rp("if (E->nv) { int v = E->v2map[P->id[i]]; if (v >= 0) E->v2tick[v] = stamp; } }",
   "if (E->nv) { E->xcnt[P->id[i]]++; int v = E->v2map[P->id[i]]; if (v >= 0) E->v2tick[v] = stamp; } }")
rp("                if (w >= 0) { if (E->v2owner[w] >= 0) E->v2map[E->v2owner[w]] = -1;",
   "                if (w >= 0 && E->v2owner[w] >= 0 && E->xmap[E->v2owner[w]] < 0 && E->xcnt[E->v2owner[w]] > E->xcnt[old]) w = -1;   /* not worth displacing */\n"
   "                if (w >= 0) { if (E->v2owner[w] >= 0) E->v2map[E->v2owner[w]] = -1;")
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("frequency-aware VRAM tier")
