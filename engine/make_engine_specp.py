"""make_engine_specp.py - exact speculative SAMPLING support for nvme_engine_u.c (target side):
  VERIFYP n temp topp topk\n<tokens>   one weight pass over n tokens; for each position prints the target's sampling distribution
                                      "P cnt id prob id prob ..." (the exact distribution GEN's sampler draws from: top-k, temperature,
                                      top-p cut, renormalised), then "END verifyp n pos ms". temp <= 0 gives the one-hot argmax."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")

def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt} match(es), got {n}: {old[:80]!r}"; s = s.replace(old, new)

DIST = r'''static int cmpd(const void *a, const void *b) {'''
rep(DIST, r'''/* the distribution sample_tok draws from, written out: ids[0..n) with probabilities pr (sum 1) */
static int dist_tok(const float *lgt, int V, float temp, float topp, int topk, int *ids, float *pr) {
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (lgt[i] > lgt[b]) b = i; ids[0] = b; pr[0] = 1.0f; return 1; }
    if (topk < 1 || topk > 128) topk = 128; if (topk > V) topk = V;
    int idx[128]; float val[128]; int k = 0;
    for (int i = 0; i < V; i++) { float v = lgt[i]; if (k == topk && v <= val[k - 1]) continue; int j = (k < topk) ? k++ : k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; } val[j] = v; idx[j] = i; }
    float p[128], sum = 0, mx = val[0]; for (int j = 0; j < k; j++) { p[j] = expf((val[j] - mx) / temp); sum += p[j]; }
    for (int j = 0; j < k; j++) p[j] /= sum;
    float cum = 0; int kk = k; for (int j = 0; j < k; j++) { cum += p[j]; if (cum >= topp) { kk = j + 1; break; } }
    float s2 = 0; for (int j = 0; j < kk; j++) s2 += p[j];
    for (int j = 0; j < kk; j++) { ids[j] = idx[j]; pr[j] = p[j] / s2; }
    return kk;
}
static int cmpd(const void *a, const void *b) {''')
rep('''        else if (!strncmp(line, "VERIFY", 6)) {''', '''        else if (!strncmp(line, "VERIFYP", 7)) {
            int ntok = 0, topk = 20; float temp = 0.7f, topp = 0.8f;
            if (sscanf(line, "VERIFYP %d %f %f %d", &ntok, &temp, &topp, &topk) != 4) { printf("ERR bad VERIFYP\\n"); fflush(stdout); if (!fgets(line, 1 << 22, stdin)) break; continue; }
            if (!fgets(line, 1 << 22, stdin)) break;
            if (ntok < 1 || ntok > E.B || E.pos + ntok > ctx) { printf("ERR verify n=%d pos=%d batch=%d ctx=%d\\n", ntok, E.pos, E.B, ctx); fflush(stdout); continue; }
            if (!G_LALL) G_LALL = malloc(4 * (size_t)E.B * E.c.vocab);
            int *pt = malloc(sizeof(int) * (size_t)ntok); char *p = line; for (int i = 0; i < ntok; i++) { char *e; pt[i] = (int)strtol(p, &e, 10); p = e; }
            double tv = now(); forward(&E, pt, ntok, 2); free(pt);
            for (int t = 0; t < ntok; t++) { int ids[128]; float pr[128]; int n = dist_tok(G_LALL + (size_t)t * E.c.vocab, E.c.vocab, temp, topp, topk, ids, pr);
                printf("P %d", n); for (int j = 0; j < n; j++) printf(" %d %.9g", ids[j], pr[j]); printf("\\n"); }
            printf("END verifyp %d %d %.0f\\n", ntok, E.pos, (now() - tv) * 1e3); fflush(stdout); }
        else if (!strncmp(line, "VERIFY", 6)) {''')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("VERIFYP added to", P)
