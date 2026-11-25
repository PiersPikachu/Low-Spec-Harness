"""tool-name resolution: the reported failing call and the other ways models mangle tool names"""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "app")]
import harness_core as c
P = c.load_plugins(); en = {"artifact", "calculator", "clock"}; ok = True
cases = [("create", {"action": "create", "name": "rubiks-cube", "type": "page"}, "artifact", "create"),
         ("create", {"name": "x"}, "artifact", "create"),                          # action missing: taken from the name
         ("functions.artifact", {"action": "list"}, "artifact", "list"),
         ("artifact.write", {"name": "x", "content": "y"}, "artifact", "write"),
         ("artifact_edit", {"name": "x"}, "artifact", "edit"),
         ("functions.create", {"action": "create"}, "artifact", "create"),
         ("artifact", {"action": "read", "name": "x"}, "artifact", "read"),
         ("calculator", {"expression": "1+1"}, "calculator", None),
         ("read_file", {"path": "."}, "read_file", None)]                           # not enabled: left alone -> error
for name, args, want, act in cases:
    n, a = c.resolve_tool(P, name, args, en); good = n == want and (act is None or a.get("action") == act); ok &= good
    print("PASS" if good else "FAIL", f"{name}({args}) -> {n}({a})")
print(c.run_tool(P, "read_file", {"path": "."})[:20] == c.run_tool(P, "read_file", {"path": "."})[:20])
print("unknown:", c.run_tool(P, "teleport", {}))
# the exact reported call, as gpt-oss emits it, through the parser and resolver
raw = '<|channel|>analysis<|message|>Create the page.<|end|><|start|>assistant<|channel|>commentary to=functions.create <|constrain|>json<|message|>{"action": "create", "name": "rubiks-cube", "type": "page", "title": "Interactive 3D Rubik\'s Cube"}'
r = c.parse_reply(raw); n, a = c.resolve_tool(P, *r["calls"][0], en); good = n == "artifact" and a["action"] == "create"; ok &= good
print("PASS" if good else "FAIL", "reported call:", r["calls"][0][0], "->", n, a)
print("ALIAS TESTS", "PASSED" if ok else "FAILED")
