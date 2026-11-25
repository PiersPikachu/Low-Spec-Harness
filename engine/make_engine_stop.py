"""make_engine_stop.py - GEN can be interrupted: a 'STOP' line on stdin ends generation after the current token (END stop).
Polls the stdin pipe without blocking (PeekNamedPipe) once per generated token. The last token is forwarded like 'max', so the
KV cache holds everything that was printed."""
import os
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvme_engine_u.c")
s = open(P, encoding="utf-8", errors="surrogateescape", newline="").read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old)); s = s.replace(old, new)
rep("static volatile long *g_xready, *g_xsaving;     /* defined with the loader below */",
    "static volatile long *g_xready, *g_xsaving;     /* defined with the loader below */\n"
    "static int stdin_stop(void) {          /* a pending 'STOP' line on the stdin pipe? (non-blocking) */\n"
    "    DWORD av = 0; HANDLE h = GetStdHandle(STD_INPUT_HANDLE);\n"
    "    if (!PeekNamedPipe(h, NULL, 0, NULL, &av, NULL) || av == 0) return 0;\n"
    "    char ln[64]; if (!fgets(ln, sizeof ln, stdin)) return 0; return !strncmp(ln, \"STOP\", 4);\n}")
rep('''        if (new_tok >= max_new) { forward(E, &tok, 1, 0); reason = "max"; break; }''',
    '''        if (new_tok >= max_new) { forward(E, &tok, 1, 0); reason = "max"; break; }
        if (stdin_stop()) { forward(E, &tok, 1, 0); reason = "stop"; break; }''')
open(P, "w", encoding="utf-8", errors="surrogateescape", newline="\n").write(s); print("engine: STOP during GEN")
