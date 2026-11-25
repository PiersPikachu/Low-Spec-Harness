"""make_engine_vram.py - the GPU's memory as a second expert cache tier (--vram-mb N), applied after make_engine_moe2.py.

cudart is loaded at run time (LoadLibrary), so the gcc-built CPU engine needs no CUDA build. RAM slots are page-locked
(cudaHostRegister) for fast copies. When a RAM slot is reclaimed, the expert it held is first copied into VRAM (if not there yet,
evicting the least recently used VRAM expert not needed by the current layer); a RAM miss whose expert sits in VRAM is copied
back over PCIe (~0.5 ms for 6.9 MB) instead of read from the SSD (~2.8 ms). All copies run on the loader thread, overlapped with
compute. Bytes are moved, never changed, so outputs are identical. MOESTATS adds: VRAM hits, VRAM slots, saves."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt}, got {n}: {old[:90]!r}"; s = s.replace(old, new)

rep("    uint8_t **xslot; int *xowner, *xmap; int64_t *xtick, tick; int nslots; size_t xsz; long long xhit, xmiss; double xread_ms;\n};",
    "    uint8_t **xslot; int *xowner, *xmap; int64_t *xtick, tick; int nslots; size_t xsz; long long xhit, xmiss; double xread_ms;\n"
    "    uint8_t *vbase; int nv, *v2owner, *v2map; int64_t *v2tick; long long vhit, vsave; double vram_mb, vcopy_ms;\n};")
rep('else if (!strcmp(argv[i], "--resident-mb") && i + 1 < argc) E->resident_mb = atof(argv[++i]);',
    'else if (!strcmp(argv[i], "--resident-mb") && i + 1 < argc) E->resident_mb = atof(argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--vram-mb") && i + 1 < argc) E->vram_mb = atof(argv[++i]);')

CU = r'''/* ---------- VRAM expert tier: cudart loaded at run time ---------- */
typedef int (*cuMalloc_t)(void **, size_t); typedef int (*cuMemcpy_t)(void *, const void *, size_t, int);
typedef int (*cuHostReg_t)(void *, size_t, unsigned); typedef int (*cuMemInfo_t)(size_t *, size_t *);
static cuMalloc_t g_cuMalloc; static cuMemcpy_t g_cuMemcpy; static cuHostReg_t g_cuHostReg; static cuMemInfo_t g_cuMemInfo;
static int cuda_load(void) {
    const char *env = getenv("NVME_CUDART"); HMODULE h = env ? LoadLibraryA(env) : NULL;
    const char *cand[] = {"cudart64_12.dll", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\cudart64_12.dll", "cudart64_110.dll", NULL};
    for (int i = 0; !h && cand[i]; i++) h = LoadLibraryA(cand[i]);
    if (!h) return 0;
    g_cuMalloc = (cuMalloc_t)GetProcAddress(h, "cudaMalloc"); g_cuMemcpy = (cuMemcpy_t)GetProcAddress(h, "cudaMemcpy");
    g_cuHostReg = (cuHostReg_t)GetProcAddress(h, "cudaHostRegister"); g_cuMemInfo = (cuMemInfo_t)GetProcAddress(h, "cudaMemGetInfo");
    return g_cuMalloc && g_cuMemcpy && g_cuHostReg && g_cuMemInfo;
}
static void vram_init(Eng *E) {
    E->nv = 0; if (E->vram_mb <= 0) return;
    if (!cuda_load()) { lg("[moe] --vram-mb ignored: cudart64 not found (set NVME_CUDART to its path)"); return; }
    size_t fr = 0, tot = 0; if (g_cuMemInfo(&fr, &tot)) { lg("[moe] --vram-mb ignored: no CUDA device"); return; }
    double want = E->vram_mb * 1e6, lim = (double)fr - 400e6; if (want > lim) want = lim;
    int nx = E->c.nl * E->c.n_exp, nv = (int)(want / (double)E->xsz); if (nv > nx) nv = nx;
    while (nv > 0 && g_cuMalloc((void **)&E->vbase, (size_t)nv * E->xsz)) nv = nv * 9 / 10;
    if (nv <= 0) { lg("[moe] VRAM tier: allocation failed"); return; }
    E->nv = nv; E->v2owner = malloc(sizeof(int) * nv); E->v2tick = calloc(nv, sizeof(int64_t)); E->v2map = malloc(sizeof(int) * nx);
    for (int i = 0; i < nv; i++) E->v2owner[i] = -1; for (int i = 0; i < nx; i++) E->v2map[i] = -1;
    int reg = 0; for (int i = 0; i < E->nslots; i++) reg += g_cuHostReg(E->xslot[i], E->xsz, 0) == 0;
    lg("[moe] VRAM tier: %d slots = %.0f MB (%.1f%% of experts), %d/%d RAM slots page-locked", nv, nv * (double)E->xsz / 1e6, 100.0 * nv / nx, reg, E->nslots);
}
/* pick the VRAM slot to receive an expert: least recently used whose tick is older than `stamp` (not needed by this layer) */
/* preference: an empty slot, then the LRU slot whose expert is ALSO in RAM (dropping a duplicate costs nothing), then plain LRU */
static int vram_victim(Eng *E, int64_t stamp) {
    int v = -1, d = -1;
    for (int q = 0; q < E->nv; q++) { if (E->v2tick[q] >= stamp) continue; int o = E->v2owner[q];
        if (o < 0) return q;
        if (E->xmap[o] >= 0 && (d < 0 || E->v2tick[q] < E->v2tick[d])) d = q;
        if (v < 0 || E->v2tick[q] < E->v2tick[v]) v = q; }
    return d >= 0 ? d : v;
}
'''
rep("/* ---------- expert plan: hits computed while misses load ---------- */", CU + "/* ---------- expert plan: hits computed while misses load ---------- */")
rep("typedef struct { Eng *E; int n, nmiss, async; int id[1024], slot[1024], order[1024], cnt[1024], base[1024]; volatile long ready[1024];",
    "typedef struct { Eng *E; int n, nmiss, async; int id[1024], slot[1024], order[1024], cnt[1024], base[1024], vsrc[1024], vsave[1024]; volatile long ready[1024];")
# loader: save the slot's previous expert to VRAM, then fill the slot from VRAM or the SSD
rep('''    for (int k = P->n - P->nmiss; k < P->n; k++) { int i = P->order[k]; Stream *st = &E->st[E->c.exp0 + P->id[i]]; double t0 = now();
        if (read_stream(&E->vf, E->xslot[P->slot[i]], st->off, (size_t)st->padded, E->io_threads, E->io_chunk)) die("expert read failed");
        E->xread_ms += (now() - t0) * 1e3; __atomic_store_n(&P->ready[i], 1, __ATOMIC_RELEASE); }''',
'''    for (int k = P->n - P->nmiss; k < P->n; k++) { int i = P->order[k]; Stream *st = &E->st[E->c.exp0 + P->id[i]]; double t0 = now();
        if (P->vsave[i] >= 0) { if (g_cuMemcpy(E->vbase + (size_t)P->vsave[i] * E->xsz, E->xslot[P->slot[i]], E->xsz, 1)) die("VRAM save failed"); }
        if (P->vsrc[i] >= 0) { if (g_cuMemcpy(E->xslot[P->slot[i]], E->vbase + (size_t)P->vsrc[i] * E->xsz, (size_t)st->padded, 2)) die("VRAM copy failed");
            E->vcopy_ms += (now() - t0) * 1e3; }
        else { if (read_stream(&E->vf, E->xslot[P->slot[i]], st->off, (size_t)st->padded, E->io_threads, E->io_chunk)) die("expert read failed");
            E->xread_ms += (now() - t0) * 1e3; }
        __atomic_store_n(&P->ready[i], 1, __ATOMIC_RELEASE); }''')
# plan: protect VRAM copies this layer needs, decide source and save target per miss
rep('''    int64_t stamp = ++E->tick; int nh = 0, *miss = malloc(sizeof(int) * P->n), nm = 0;''',
'''    int64_t stamp = ++E->tick; int nh = 0, *miss = malloc(sizeof(int) * P->n), nm = 0;
    for (int i = 0; i < P->n; i++) { P->vsrc[i] = -1; P->vsave[i] = -1; if (E->nv) { int v = E->v2map[P->id[i]]; if (v >= 0) E->v2tick[v] = stamp; } }''')
rep('''        if (E->xowner[v] >= 0) E->xmap[E->xowner[v]] = -1;
        E->xowner[v] = P->id[miss[k]]; E->xmap[P->id[miss[k]]] = v; E->xtick[v] = stamp; P->slot[miss[k]] = v; P->order[nh + k] = miss[k]; E->xmiss++;''',
'''        int old = E->xowner[v], mi = miss[k];
        if (old >= 0) { E->xmap[old] = -1;
            if (E->nv && E->v2map[old] < 0) { int w = vram_victim(E, stamp);       /* keep the reclaimed expert in VRAM */
                if (w >= 0) { if (E->v2owner[w] >= 0) E->v2map[E->v2owner[w]] = -1; E->v2owner[w] = old; E->v2map[old] = w; E->v2tick[w] = stamp; P->vsave[mi] = w; E->vsave++; } } }
        if (E->nv && E->v2map[P->id[mi]] >= 0) { P->vsrc[mi] = E->v2map[P->id[mi]]; E->vhit++; }
        E->xowner[v] = P->id[mi]; E->xmap[P->id[mi]] = v; E->xtick[v] = stamp; P->slot[mi] = v; P->order[nh + k] = mi; E->xmiss++;''')
rep('    if (E->c.moe) moe_cache_init(E, budget - E->resident_used);', '    if (E->c.moe) { moe_cache_init(E, budget - E->resident_used); vram_init(E); }')
rep('printf("M %lld %lld %.1f %d\\n", E.xhit, E.xmiss, E.xread_ms, E.nslots);', 'printf("M %lld %lld %.1f %d %lld %d %lld %.1f\\n", E.xhit, E.xmiss, E.xread_ms, E.nslots, E.vhit, E.nv, E.vsave, E.vcopy_ms);')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("VRAM expert tier added to", P)
