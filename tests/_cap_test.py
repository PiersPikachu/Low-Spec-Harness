"""per-reply file growth cap (default 4 x max reply, or the Config setting) and the outline + last-40-lines continuity (scripted model)"""
import json, shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)
P = core.load_plugins(); A = P["artifact"]; real = A.ROOT; A.ROOT = Path(tempfile.mkdtemp()) / "artifacts"
LINES = [f"function step{i:03d}() {{ return {i}; }}   // padded to about sixty characters\n" for i in range(160)]
FULL = "".join(LINES)
def j(s): return json.dumps(s)[1:-1]
class Tok:
    chat_template = "<|channel|>"
    def __init__(self): self.v = []
    def ids(self, text, n=12):
        out = []
        for i in range(0, len(text), n): self.v.append(text[i:i + n]); out.append(len(self.v) - 1)
        return out
    def decode(self, ids, skip_special_tokens=False): return "".join(self.v[i] for i in ids)
tok = Tok()
class Eng:
    def __init__(self, rounds): self.rounds = rounds; self.stop = False
    def request_stop(self): self.stop = True
    def generate(self, ids, max_new, temp, top_p, top_k, seed, on_token):
        r = self.rounds.pop(0); script = r() if callable(r) else r; out = []; self.stop = False
        for t in script:
            out.append(t); on_token(out)
            if self.stop: return out + [tok.ids(" ")[0]], "stop", "tok_s=9"
        return out, "eos", "tok_s=9"
def call(action, content, name="big"):
    return tok.ids(f'<|channel|>analysis<|message|>w<|end|><|start|>assistant<|channel|>commentary to=functions.artifact <|constrain|>json<|message|>'
                   f'{{"action": "{action}", "name": "{name}", "path": "app.js", "content": "{j(content)}"}}')
final = lambda: tok.ids("<|channel|>final<|message|>The file is incomplete; say continue to resume.")
def run(cfg_extra, name):
    A.run("create", name=name, type="code", path="app.js", content="")
    rest = lambda: call("append", FULL[len((A.ROOT / name / "app.js").read_text()):], name)
    eng = Eng([call("write", FULL, name), rest, rest, rest, rest, final, final])
    S = object.__new__(core.Session); S.cfg = dict(hp.load_config(), ctx=10**9, greedy=True, **cfg_extra); S.tok = tok; S.eos = set(); S.eng = eng
    S.spec = False; S.chat_mode = True; S._ids = lambda msgs, tools: [1]
    new, st = S.reply([{"role": "user", "content": "write it"}], "", core.tool_schemas(P, {"artifact"}), P, lambda k, t: None, lambda: False, 4)
    return new, (A.ROOT / name / "app.js").read_text()
# A: default cap = 4 x max reply (max_new 1500 -> 6000 characters)
new, f = run({"max_new": 1500, "max_file_growth": 0}, "a1")
res = [m["content"] for m in new if m["role"] == "tool"]
chk(f"default cap: {len(f)} characters saved (cap 6000), whole lines", len(f) <= 6000 and f.endswith("\n") and FULL.startswith(f) and len(f) > 6000 - 70)
chk("model told the limit is reached and to tell the user", any("growth limit (6,000 characters) is reached" in r for r in res))
chk("reply ends with the model's message to the user", new[-1]["content"].startswith("The file is incomplete"))
# B: the Config setting overrides the default
new, f = run({"max_new": 1500, "max_file_growth": 9000}, "b1")
chk(f"setting max_file_growth=9000: {len(f)} characters saved", 9000 - 70 < len(f) <= 9000)
# C: continuity - the continuation includes an outline and the last 40 lines
new, f = run({"max_new": 100000, "max_file_growth": 0}, "c1")
cont = next(m["content"] for m in new if m["role"] == "tool" and "Continue the same file" in m["content"])
chk("continuation lists the outline (functions with line numbers)", "Outline:" in cont and "L1: function step000()" in cont)
chk("continuation shows the last 40 lines", "Last 40 lines:" in cont and cont.count("| function step") == 40)
chk("no cap at 4 x 100000: whole file written", f == FULL)
# refused after the cap: a further write in the same reply returns an error
A.run("create", name="d1", type="code", path="app.js", content="")
eng = Eng([call("write", FULL[:5000], "d1"), call("append", "more\n", "d1"), final])
S = object.__new__(core.Session); S.cfg = dict(hp.load_config(), ctx=10**9, greedy=True, max_new=500, max_file_growth=0); S.tok = tok; S.eos = set(); S.eng = eng
S.spec = False; S.chat_mode = True; S._ids = lambda msgs, tools: [1]
new, st = S.reply([{"role": "user", "content": "w"}], "", core.tool_schemas(P, {"artifact"}), P, lambda k, t: None, lambda: False, 4)
res = [m["content"] for m in new if m["role"] == "tool"]
chk("a write after the cap is refused", any("already reached" in r for r in res) and "more" not in (A.ROOT / "d1" / "app.js").read_text())
shutil.rmtree(A.ROOT.parent); A.ROOT = real
print("CAP TESTS", "PASSED" if ok else "FAILED")
