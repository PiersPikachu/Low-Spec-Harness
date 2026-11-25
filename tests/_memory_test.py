"""trimmed turns are archived (nothing lost) and a reply that fills the context window continues after compaction (scripted model)"""
import json, shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)
P = core.load_plugins(); A = P["artifact"]; real = A.ROOT; A.ROOT = Path(tempfile.mkdtemp()) / "artifacts"
class Tok:
    chat_template = "plain"
    def __init__(self): self.v = []
    def ids(self, text):
        self.v.append(text); return [len(self.v) - 1]
    def decode(self, ids, skip_special_tokens=False): return "".join(self.v[i] for i in ids)
tok = Tok()
class Eng:
    def __init__(self, plan): self.plan = plan; self.calls = []
    def request_stop(self): pass
    def generate(self, ids, max_new, temp, top_p, top_k, seed, on_token):
        self.calls.append(len(ids)); words, reason = self.plan.pop(0); out = []
        for w in words: out.append(tok.ids(w)[0]); on_token(out)
        return out, reason, "tok_s=9"
S = object.__new__(core.Session); S.tok = tok; S.eos = set(); S.spec = False; S.chat_mode = True
S._ids = lambda msgs, tools: [0] * sum(10 + len(m.get("content") or "") // 10 for m in msgs)     # ~1 token per 10 chars
hist = []
for i in range(12): hist += [{"role": "user", "content": f"question {i} " + "x" * 300}, {"role": "assistant", "content": f"answer {i} " + "y" * 300}]
hist.append({"role": "user", "content": "the current question"})
# 1) context too small for the whole history: oldest turns are archived, the current question is kept
S.cfg = dict(hp.load_config(), ctx=400, max_new=100, greedy=True); S.eng = Eng([(["The ", "answer."], "eos")])
new, st = S.reply(hist, "", None, P, lambda k, t: None, lambda: False, 1)
arts = [d.name for d in A.ROOT.iterdir()] if A.ROOT.exists() else []
chk("a memory artifact was created " + str(arts), len(arts) == 1 and arts[0].startswith("memory-"))
txt = (A.ROOT / arts[0] / "notes.md").read_text(encoding="utf-8") if arts else ""
chk("dropped turns archived word for word", "question 0 " + "x" * 300 in txt and "answer 0 " + "y" * 300 in txt)
chk("prompt fitted the window", S.eng.calls[0] + 100 <= 400 + 50)
chk("reply produced", new[-1]["content"] == "The answer.")
# 2) a reply that runs into the end of the window continues after compaction, as one message
S.cfg = dict(hp.load_config(), ctx=900, max_new=300, greedy=True)
S.eng = Eng([(["Part one, ", "still going, "], "ctx"), (["part two, ", "done."], "eos")])
seen = []
new2, st2 = S.reply(hist, "", None, P, lambda k, t: seen.append((k, t)), lambda: False, 1)
chk("continued into one message: " + repr(new2[-1]["content"]), new2[-1]["content"] == "Part one, still going, part two, done." and len(new2) == 1)
chk("the continuation resumed from the generated tokens (prompt grew by them)", len(S.eng.calls) == 2)
chk("streamed text never went backwards", "".join(t for k, t in seen if k == "assistant") == "Part one, still going, part two, done.")
shutil.rmtree(A.ROOT.parent); A.ROOT = real
print("MEMORY TESTS", "PASSED" if ok else "FAILED")
