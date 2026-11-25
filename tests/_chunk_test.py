"""chunked artifact writing end to end with a scripted model (deterministic), plus the real engine's STOP command"""
import json, shutil, sys, tempfile, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp, chat
ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)

P = core.load_plugins(); A = P["artifact"]; real = A.ROOT; A.ROOT = Path(tempfile.mkdtemp()) / "artifacts"
LINES = [f"<p>line {i:03d} of the page, padded to make it about sixty chars.</p>\n" for i in range(160)]   # ~10.6k chars
FULL = "".join(LINES)
A.run("create", name="big", type="page", title="Big")

def j(s): return json.dumps(s)[1:-1]                      # JSON-escape a string body
class Tok:                                                 # token = a short piece of text
    chat_template = "<|channel|>"
    def __init__(self): self.v = []
    def ids(self, text, n=12):
        out = []
        for i in range(0, len(text), n): self.v.append(text[i:i + n]); out.append(len(self.v) - 1)
        return out
    def decode(self, ids, skip_special_tokens=False): return "".join(self.v[i] for i in ids)
tok = Tok()
class Eng:                                                 # replays one scripted reply per round; honours STOP like the engine
    def __init__(self): self.rounds = []; self.stop = False; self.seen = []
    def request_stop(self): self.stop = True
    def generate(self, ids, max_new, temp, top_p, top_k, seed, on_token):
        script = self.rounds.pop(0); out = []; self.stop = False
        for t in script:
            out.append(t); on_token(out)
            if self.stop: return out + [tok.ids(" ")[0]], "stop", "tok_s=99"    # engine emits one more token, then END stop
            if len(out) >= max_new: return out, "max", "tok_s=99"
        return out, "eos", "tok_s=99"
eng = Eng()
def call(action, content, extra=""):
    return tok.ids(f'<|channel|>analysis<|message|>writing<|end|><|start|>assistant<|channel|>commentary to=functions.artifact <|constrain|>json<|message|>'
                   f'{{"action": "{action}", "name": "big", "path": "index.html"{extra}, "content": "{j(content)}"}}')
# the model tries to write the whole file in one call; then (as instructed) appends the rest; then answers
eng.rounds = [call("write", FULL), None, None, None, tok.ids("<|channel|>final<|message|>The page is complete.")]
S = object.__new__(core.Session); S.cfg = dict(hp.load_config(), max_new=100000, ctx=10**9, greedy=True); S.tok = tok; S.eos = set(); S.eng = eng
S.spec = False; S.chat_mode = True; seen_prompts = []
S._ids = lambda msgs, tools: (seen_prompts.append(json.dumps(msgs)), [1, 2, 3])[1]
live_seen = []
def on_text(k, t):
    if k == "act" and t.get("ev") == "call_stream" and (A.ROOT / "big" / "index.html").exists():
        live_seen.append(len((A.ROOT / "big" / "index.html").read_text()))
def fill_next():                                           # the 'model' continues from where the tool result says
    pass
orig = eng.generate
def gen(ids, *a):
    if eng.rounds and eng.rounds[0] is None:
        saved = (A.ROOT / "big" / "index.html").read_text(); rest = FULL[len(saved):]
        eng.rounds[0] = call("append", rest)
    return orig(ids, *a)
eng.generate = gen
tools = core.tool_schemas(P, {"artifact"}); t0 = time.time()
new, stats = S.reply([{"role": "user", "content": "Write the big page."}], "", tools, P, on_text, lambda: False, 2)
final = (A.ROOT / "big" / "index.html").read_text()
calls = [m for m in new if m.get("tool_calls")]; results = [m["content"] for m in new if m["role"] == "tool"]
print(f"   {len(calls)} calls, sizes {[len(c['tool_calls'][0]['function']['arguments']['content']) for c in calls]}, file {len(final)} chars (expected {len(FULL)})")
chk("file assembled exactly from the parts", final == FULL)
chk("each harness-cut call is <= one line past 3000 chars and ends on a line", all(len(c['tool_calls'][0]['function']['arguments']['content']) <= 3000 + 80 and
     c['tool_calls'][0]['function']['arguments']['content'].endswith("\n") for c in calls))
chk("continuation rounds beyond max_tool_rounds=2 allowed", len(calls) >= 4)
chk("tool result tells the model where to continue", "Continue the same file now" in results[0] and "line 0" in results[0])
chk("final answer reached", new[-1]["content"] == "The page is complete.")
chk("history shows one clean change per call", [c["action"] for c in A._meta(A.ROOT / "big")["changes"]] == ["write"] + ["append"] * (len(calls) - 1))
chk(f"live saves were on disk while streaming (sizes seen {sorted(set(live_seen))[:4]}...)", any(0 < x < 3000 for x in live_seen))
chk("older calls compacted in later prompts", "characters, saved by the tool]" in seen_prompts[-1])
# Stop button mid-generation (plain engine): stop is honoured, no call made
A.run("create", name="s", type="note"); eng.rounds = [tok.ids("<|channel|>final<|message|>" + "word " * 200)]; n = [0]
def should_stop(): n[0] += 1; return n[0] > 20
new2, st2 = S.reply([{"role": "user", "content": "talk"}], "", tools, P, lambda k, t: None, should_stop, 2)
chk("Stop ends the reply mid-generation (" + st2 + ")", "stop: stop" in st2 and len(new2) == 1)
shutil.rmtree(A.ROOT.parent); A.ROOT = real
# the real engine honours STOP during GEN
v = hp.resolve_vault("vault_06b_huff"); e = chat.Engine([str(hp.ENGINE), "serve", str(v), "--ctx", "512", "--log", str(hp.LOGS / "stoptest.txt")], str(hp.LOGS / "stoptest.err"))
got = []
def ot(out):
    got.append(len(out))
    if len(out) == 5: e.request_stop()
out, reason, _ = e.generate([1, 2, 3, 4, 5], 200, 0.0, 1.0, 1, 1, ot); e.close()
chk(f"real engine: STOP after token 5 -> {len(out)} tokens, reason {reason}", reason == "stop" and 5 <= len(out) <= 7)
print("CHUNK TESTS", "PASSED" if ok else "FAILED")
