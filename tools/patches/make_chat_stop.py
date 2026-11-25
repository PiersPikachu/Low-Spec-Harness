"""make_chat_stop.py - chat.Engine.request_stop(): ask the engine to end the current GEN after the next token (STOP line);
reason 'stop' keeps the KV cache like 'max'. SpecPair stops through its should_stop check, so request_stop sets a flag there."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "chat.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
rep('''        self.fed = []                                      # tokens currently in the engine's KV cache

    def reset(self):
        self.p.stdin.write("RESET\\n"); self.p.stdin.flush(); self.p.stdout.readline(); self.fed = []''',
'''        self.fed = []                                      # tokens currently in the engine's KV cache
        self._stop_sent = False

    def request_stop(self):
        """end the running GEN after the next token (the engine polls stdin for STOP between tokens)"""
        if not self._stop_sent:
            self._stop_sent = True
            try: self.p.stdin.write("STOP\\n"); self.p.stdin.flush()
            except Exception: pass

    def reset(self):
        self.p.stdin.write("RESET\\n"); self.p.stdin.flush(); self.p.stdout.readline(); self.fed = []''')
rep('''        new = ids[lcp:]
        self.p.stdin.write(f"GEN {max_new}''', '''        new = ids[lcp:]; self._stop_sent = False
        self.p.stdin.write(f"GEN {max_new}''')
rep('''        self.fed = ids + (out if reason == "max" else out[:-1])''', '''        self.fed = ids + (out if reason in ("max", "stop") else out[:-1])''')
rep('''        self.k = k; self.eos = eos; self.fed = []; self.batch = 128     # prompt tokens per weight pass''',
    '''        self.k = k; self.eos = eos; self.fed = []; self.batch = 128     # prompt tokens per weight pass
        self.stop_flag = False

    def request_stop(self):
        self.stop_flag = True''')
rep('''            if should_stop and should_stop(): reason = "stopped"; break''',
    '''            if (should_stop and should_stop()) or self.stop_flag: reason = "stopped" if not self.stop_flag else "stop"; break''')
rep('''        rng = random.Random(seed); sp = (temp, top_p, top_k); t0 = time.time(); rows = None''',
    '''        rng = random.Random(seed); sp = (temp, top_p, top_k); t0 = time.time(); rows = None; self.stop_flag = False''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("chat.py: request_stop for Engine and SpecPair")
