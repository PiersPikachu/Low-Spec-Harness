"""make_engine_spec.py - speculative-decoding support for nvme_engine_u.c (target side):
  VERIFY n\n<tokens>   run n tokens through the model in ONE pass over the weights and return the greedy (argmax) next token after
                       each of them:  "A a0 a1 ... a(n-1)" + "END verify n pos ms"
  SETPOS p             roll the KV cache back to p tokens (p <= current position), e.g. after drafted tokens are rejected
One VERIFY of k+1 tokens costs about one normal token when the engine is disk- or decode-bound: every weight row is decoded once and
multiplied by all n vectors (mv_batch), which is what makes checking k drafted tokens nearly free."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read().replace("\r\n", "\n")

def rep(old, new, cnt=1):
    global s
    n = s.count(old); assert n == cnt, f"expected {cnt} match(es), got {n}: {old[:80]!r}"; s = s.replace(old, new)

rep("static void forward(Eng *E, const int *tok, int n, int want_logits) {",
    "static float *G_LALL = NULL;   /* VERIFY: logits of every position, n x vocab */\nstatic void forward(Eng *E, const int *tok, int n, int want_logits) {")
rep("        else { uint64_t o; memcpy(&o, blob + 8, 8); rmsnorm(E->XN, E->H + (size_t)(n - 1) * c->hid, E->fnrm, c->hid, (float)c->eps); mv_batch(blob + o, E->XN, E->logits, 1); }",
    "        else { uint64_t o; memcpy(&o, blob + 8, 8);\n"
    "            if (want_logits == 2) { for (int t = 0; t < n; t++) rmsnorm(E->XN + (size_t)t * c->hid, E->H + (size_t)t * c->hid, E->fnrm, c->hid, (float)c->eps); mv_batch(blob + o, E->XN, G_LALL, n); }\n"
    "            else { rmsnorm(E->XN, E->H + (size_t)(n - 1) * c->hid, E->fnrm, c->hid, (float)c->eps); mv_batch(blob + o, E->XN, E->logits, 1); } }")
rep('''        else if (!strncmp(line, "RESET", 5)) { E.pos = 0; lg("[reset] kv cache cleared"); printf("OK\\n"); fflush(stdout); }''',
    '''        else if (!strncmp(line, "RESET", 5)) { E.pos = 0; lg("[reset] kv cache cleared"); printf("OK\\n"); fflush(stdout); }
        else if (!strncmp(line, "SETPOS", 6)) { int p = -1; sscanf(line, "SETPOS %d", &p);
            if (p < 0 || p > E.pos) printf("ERR setpos %d (pos %d)\\n", p, E.pos); else { E.pos = p; printf("OK %d\\n", p); } fflush(stdout); }
        else if (!strncmp(line, "VERIFY", 6)) {
            int ntok = 0; sscanf(line, "VERIFY %d", &ntok); if (!fgets(line, 1 << 22, stdin)) break;
            if (ntok < 1 || ntok > E.B || E.pos + ntok > ctx) { printf("ERR verify n=%d pos=%d batch=%d ctx=%d\\n", ntok, E.pos, E.B, ctx); fflush(stdout); continue; }
            if (!G_LALL) G_LALL = malloc(4 * (size_t)E.B * E.c.vocab);
            int *pt = malloc(sizeof(int) * (size_t)ntok); char *p = line; for (int i = 0; i < ntok; i++) { char *e; pt[i] = (int)strtol(p, &e, 10); p = e; }
            double tv = now(); forward(&E, pt, ntok, 2); free(pt);
            printf("A"); for (int t = 0; t < ntok; t++) { const float *lgt = G_LALL + (size_t)t * E.c.vocab; int b = 0; for (int i = 1; i < E.c.vocab; i++) if (lgt[i] > lgt[b]) b = i; printf(" %d", b); }
            printf("\\nEND verify %d %d %.0f\\n", ntok, E.pos, (now() - tv) * 1e3); fflush(stdout); }''')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s)
print("VERIFY/SETPOS added to", P)
