"""tool-calls skill injection per format; lenient JSON; cut-off calls get a clear, actionable error"""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "app")]
import harness_core as c
P = c.load_plugins(); tools = c.tool_schemas(P, {"artifact", "harness_dev"}); ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)
h = c.tool_skill_text(tools, P, True); q = c.tool_skill_text(tools, P, False)
chk("harmony text has the harmony example only", "to=functions.artifact" in h and "<tool_call>" not in h)
chk("tool_call text has the <tool_call> example only", "<tool_call>" in q and "to=functions" not in q)
chk("{tools} filled with names and actions", "- artifact (actions: list, create" in h and "- harness_dev (actions:" in h)
chk("shared rules in both", all("write the file in parts" in x.lower() or "written in parts" in x.lower() for x in (h, q)))
print(f"   prompt size: harmony {len(h)} chars, tool_call {len(q)} chars")
# the exact failure reported: a harmony call whose JSON stops mid-string (reply hit max new tokens)
raw = ('<|channel|>analysis<|message|>Write it.<|end|><|start|>assistant<|channel|>commentary to=functions.artifact <|constrain|>json<|message|>'
       '{"action":"write","name":"rubiks-cube","path":"index.html","content":"<!DOCTYPE html>\\n<html lang=\\"en\\">\\n<head>\\n<meta charset=\\"UTF-8\\">')
r = c.parse_reply(raw); n, a = r["calls"][0]
chk("cut-off harmony call -> clear error " + repr(a.get("_error", ""))[:110], a.get("_cut") and "cut off" in a["_error"] and "append" in a["_error"])
# a complete call with a raw newline inside the JSON string (strict JSON would reject it) is now accepted
raw2 = raw.split('"content"')[0] + '"content":"<p>line one\nline two</p>"}'
r2 = c.parse_reply(raw2); chk("raw newline inside JSON string accepted", r2["calls"][0][1].get("content") == "<p>line one\nline two</p>")
# Qwen / Hermes: <tool_call> opened but never closed
r3 = c.parse_reply('<tool_call>\n{"name": "artifact", "arguments": {"action": "write", "name": "x", "content": "<html>')
chk("cut-off <tool_call> -> clear error", any(a.get("_cut") for _, a in r3["calls"]))
r4 = c.parse_reply('<tool_call>\n{"name": "artifact", "arguments": {"action": "list"}}\n</tool_call>')
chk("normal <tool_call> unaffected", r4["calls"] == [("artifact", {"action": "list"})])
sk = c.load_skills().get("tool-calls"); chk("skill detected: applies_when=tools, not always_on", sk and sk["applies_when"] == "tools" and not sk["always_on"])
print("TOOL SKILL TESTS", "PASSED" if ok else "FAILED")
