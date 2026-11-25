"""make_engine_prefetch.py - prefetch layer l+1's predicted experts while layer l computes (gpt-oss; --prefetch or NVME_MOE_PREFETCH=1).
Prediction as in make_engine_pred.py (layer l+1's norm + router on the residual after layer l's attention; measured 82-84% of
used experts predicted). Predicted experts that are in neither RAM nor VRAM get RAM slots reserved after this layer's own misses
and are read by the same loader thread right after them, so the disk keeps working through attention and routing. Decode only
(n <= 4 tokens). The bytes read are the vault's; only timing changes, so outputs are identical."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)
rep("int pred_l, pred_n, *pred; long long pred_hit, pred_tot;", "int pred_l, pred_n, *pred; long long pred_hit, pred_tot; int pf_on, pf_n, pf_ids[64]; long long pf_issued;")
rep('    if (getenv("NVME_MOE_PREDICT")) {', '    E->pf_n = 0;\n    if (getenv("NVME_MOE_PREDICT") || E->pf_on) {')
# after a prediction is made, queue the predicted experts of layer l+1 (decode only)
rep("            E->pred_l = l + 1; E->pred_n = n; free(xp); free(pl);",
    "            E->pred_l = l + 1; E->pred_n = n; free(xp); free(pl);\n"
    "            if (E->pf_on && n <= 4) for (int t = 0; t < n; t++) for (int j = 0; j < K && E->pf_n < 64; j++) {\n"
    "                int id = (l + 1) * NE + E->pred[t * K + j], dup = 0; for (int q = 0; q < E->pf_n; q++) if (E->pf_ids[q] == id) dup = 1;\n"
    "                if (!dup && E->xmap[id] < 0 && !(E->nv && E->v2map[id] >= 0)) E->pf_ids[E->pf_n++] = id; }")
# plan: reserve slots for the prefetch list after the layer's own misses; loader reads them last
rep("vsrc[1024], vsave[1024], gpu[1024], g[1024], ng, ncpu; volatile long ready[1024];",
    "vsrc[1024], vsave[1024], gpu[1024], g[1024], ng, ncpu, npf, pfid[64], pfslot[64], pfsave[64]; volatile long ready[1024];")
rep("    P->nmiss = nm; free(miss);\n    if (nm) { P->async = 1; thr_start(&P->th, xplan_load, P); }",
'''    P->nmiss = nm; free(miss); P->npf = 0;
    for (int k = 0; k < E->pf_n; k++) {                       /* next layer's predicted experts, into unneeded slots */
        int id = E->pf_ids[k]; if (E->xmap[id] >= 0) continue;
        int v = -1; for (int q = 0; q < E->nslots; q++) if (E->xtick[q] < stamp && (v < 0 || E->xtick[q] < E->xtick[v])) v = q;
        if (v < 0) break;
        int old = E->xowner[v], sv = -1;
        if (old >= 0) { E->xmap[old] = -1;
            if (E->nv && E->v2map[old] < 0) { int w = vram_victim(E, stamp);
                if (w >= 0 && E->v2owner[w] >= 0 && E->xmap[E->v2owner[w]] < 0 && E->xcnt[E->v2owner[w]] > E->xcnt[old]) w = -1;
                if (w >= 0) { if (E->v2owner[w] >= 0) E->v2map[E->v2owner[w]] = -1; E->v2owner[w] = old; E->v2map[old] = w; E->v2tick[w] = stamp; sv = w; E->vsave++; } } }
        E->xowner[v] = id; E->xmap[id] = v; E->xtick[v] = stamp; P->pfid[P->npf] = id; P->pfslot[P->npf] = v; P->pfsave[P->npf] = sv; P->npf++; E->pf_issued++;
    }
    if (nm || P->npf) { P->async = 1; thr_start(&P->th, xplan_load, P); }''')
rep("        __atomic_store_n(&P->ready[i], 1, __ATOMIC_RELEASE); }",
'''        __atomic_store_n(&P->ready[i], 1, __ATOMIC_RELEASE); }
    for (int k = 0; k < P->npf; k++) { Stream *st = &E->st[E->c.exp0 + P->pfid[k]]; double t0 = now();
        if (P->pfsave[k] >= 0 && g_cuMemcpy(E->vbase + (size_t)P->pfsave[k] * E->xsz, E->xslot[P->pfslot[k]], E->xsz, 1)) die("VRAM save failed");
        if (read_stream(&E->vf, E->xslot[P->pfslot[k]], st->off, (size_t)st->padded, E->io_threads, E->io_chunk)) die("expert prefetch failed");
        E->xread_ms += (now() - t0) * 1e3; }''')
rep('        else if (!strcmp(argv[i], "--vram-mb") && i + 1 < argc) E->vram_mb = atof(argv[++i]);',
    '        else if (!strcmp(argv[i], "--vram-mb") && i + 1 < argc) E->vram_mb = atof(argv[++i]);\n'
    '        else if (!strcmp(argv[i], "--prefetch")) E->pf_on = 1;')
rep('E.pred_hit, E.pred_tot);', 'E.pred_hit, E.pred_tot, E.pf_issued);')
rep('printf("M %lld %lld %.1f %d %lld %d %lld %.1f %lld %.1f %lld %lld\\n"', 'printf("M %lld %lld %.1f %d %lld %d %lld %.1f %lld %.1f %lld %lld %lld\\n"')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("next-layer expert prefetch added")
