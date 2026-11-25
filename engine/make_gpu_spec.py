"""make_gpu_spec.py - SETPOS p for gpu/gpu_engine.cu serve mode (draft side of speculative decoding): roll the KV cache back to p tokens."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "gpu_engine.cu")
s = open(P, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "SETPOS" in s:
    print("already patched"); raise SystemExit(0)
old = '''        else if (!strncmp(line, "RESET", 5)) { E.pos = 0; printf("OK\\n"); fflush(stdout); }'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, old + '''
        else if (!strncmp(line, "SETPOS", 6)) { int p = -1; sscanf(line, "SETPOS %d", &p);
            if (p < 0 || p > E.pos) printf("ERR setpos %d (pos %d)\\n", p, E.pos); else { E.pos = p; printf("OK %d\\n", p); } fflush(stdout); }''')
open(P, "w", encoding="utf-8", newline="").write(s); print("SETPOS added to", P)
