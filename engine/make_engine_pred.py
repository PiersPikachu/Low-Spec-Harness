"""make_engine_pred.py - measure how well layer l can predict layer l+1's experts (gpt-oss), before building a prefetcher.
With NVME_MOE_PREDICT=1, while routing layer l the engine also applies layer l+1's post-attention norm and router to the current
residual stream (after layer l's attention, before its MoE) and keeps the top-k; at layer l+1 it counts how many of the
actually chosen experts were predicted. MOESTATS gains two fields: predicted-and-used, total used. Read-only: outputs unchanged."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)
rep("long long vhit, vsave; double vram_mb, vcopy_ms; int *xcnt;", "long long vhit, vsave; double vram_mb, vcopy_ms; int *xcnt; int pred_l, pred_n, *pred; long long pred_hit, pred_tot;")
PRED = r'''
    if (getenv("NVME_MOE_PREDICT")) {      /* score last layer's prediction, then predict layer l+1 from the current residual */
        if (E->pred && E->pred_l == l && E->pred_n == n)
            for (int t = 0; t < n; t++) for (int j = 0; j < K; j++) { int a = sel[t * K + j]; E->pred_tot++;
                for (int q = 0; q < K; q++) if (E->pred[t * K + q] == a) { E->pred_hit++; break; } }
        E->pred_l = -1;
        if (l + 1 < c->nl && E->resident[l + 1]) {
            const uint8_t *nb = E->resident[l + 1]; uint64_t o4; memcpy(&o4, nb + 8 + 8 * 4, 8);
            const float *rl1 = E->resl + (size_t)(l + 1) * c->lres; float *xp = malloc(4 * (size_t)n * hid), *pl = malloc(4 * (size_t)n * NE);
            for (int t = 0; t < n; t++) rmsnorm(xp + (size_t)t * hid, E->H + (size_t)t * hid, rl1 + c->r_ln_post, hid, c->eps);
            mv_batch(nb + o4, xp, pl, n);
            if (!E->pred) E->pred = malloc(sizeof(int) * 128 * 16);
            for (int t = 0; t < n && t < 128; t++) { float *z = pl + (size_t)t * NE; for (int e = 0; e < NE; e++) z[e] += rl1[c->r_rbias + e];
                for (int j = 0; j < K; j++) { int b = -1; for (int e = 0; e < NE; e++) { int u = 0; for (int q = 0; q < j; q++) if (E->pred[t * K + q] == e) u = 1;
                        if (!u && (b < 0 || z[e] > z[b])) b = e; } E->pred[t * K + j] = b; } }
            E->pred_l = l + 1; E->pred_n = n; free(xp); free(pl);
        }
    }
    float *Xs = malloc(4 * (size_t)n * hid), *GU = malloc'''
rep("    float *Xs = malloc(4 * (size_t)n * hid), *GU = malloc", PRED)
rep('E.vgpu, E.gpu_ms);', 'E.vgpu, E.gpu_ms, E.pred_hit, E.pred_tot);')
rep('printf("M %lld %lld %.1f %d %lld %d %lld %.1f %lld %.1f\\n"', 'printf("M %lld %lld %.1f %d %lld %d %lld %.1f %lld %.1f %lld %lld\\n"')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("next-layer expert prediction (measurement) added")
