"""make_gpu_specp.py - draft side of exact speculative sampling for gpu/gpu_engine.cu (in place):
  DRAFTP k temp topp topk seed ntok\n<tokens>   feeds the tokens, then samples k tokens; for each prints the distribution it sampled from
                                               "D tok cnt id prob ..." (same top-k / temperature / top-p rule as GEN), forwards every drafted
                                               token, then "END draftp pos". temp <= 0 = greedy (one-hot)."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "gpu_engine.cu")
s = open(P, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "DRAFTP" in s:
    print("already patched"); raise SystemExit(0)

def rep(old, new):
    global s
    n = s.count(old); assert n == 1, f"expected 1, got {n}: {old[:80]!r}"; s = s.replace(old, new)

rep("int main(int argc, char **argv) {", r'''static int dist_tok(const float *lgt, int V, float temp, float topp, int topk, int *ids, float *pr) {
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

int main(int argc, char **argv) {''')
rep('''        else if (!strncmp(line, "SETPOS", 6)) {''', '''        else if (!strncmp(line, "DRAFTP", 6)) {
            int k = 0, topk = 20, ntok = 0; float temp = 0.7f, topp = 0.8f; unsigned long long seed = 1;
            if (sscanf(line, "DRAFTP %d %f %f %d %llu %d", &k, &temp, &topp, &topk, &seed, &ntok) != 6) { printf("ERR bad DRAFTP\\n"); fflush(stdout); continue; }
            std::vector<int> pt; if (!read_tokens(ntok, pt)) break;
            if (ntok < 1 || E.pos + ntok + k > ctx) { printf("ERR ctx %d %d\\n", E.pos, ctx); fflush(stdout); continue; }
            uint64_t rng = seed ? seed : 88172645463325252ull;
            for (int i = 0; i < ntok; i++) forward(&E, pt[i], E.pos++, i == ntok - 1, false);
            for (int st = 0; st < k; st++) {
                int ids[128]; float pr[128]; int n = dist_tok(E.hlog, E.c.vocab, temp, topp, topk, ids, pr);
                float r = rng_f(&rng), c = 0; int tok = ids[n - 1];
                for (int j = 0; j < n; j++) { c += pr[j]; if (r < c) { tok = ids[j]; break; } }
                printf("D %d %d", tok, n); for (int j = 0; j < n; j++) printf(" %d %.9g", ids[j], pr[j]); printf("\\n");
                forward(&E, tok, E.pos++, st < k - 1, false);
            }
            printf("END draftp %d\\n", E.pos); fflush(stdout); }
        else if (!strncmp(line, "SETPOS", 6)) {''')
open(P, "w", encoding="utf-8", newline="").write(s); print("DRAFTP added to", P)
