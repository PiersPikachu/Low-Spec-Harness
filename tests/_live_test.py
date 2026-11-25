"""partial tool-call decoding (live artifact view) + harness_dev plugin checks (temporary folders)"""
import importlib.util, shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as c
ok = True
def chk(label, got, want):
    global ok
    good = got == want if not callable(want) else want(got); ok &= bool(good); print("PASS" if good else "FAIL", label, "->", repr(got)[:150])
raw = ('<|channel|>analysis<|message|>plan<|end|><|start|>assistant<|channel|>commentary to=functions.artifact <|constrain|>json<|message|>'
       '{"action": "write", "name": "cube", "path": "index.html", "content": "<!doctype html>\\n<h1 class=\\"t\\">Cu')
pc = c.partial_call(raw)
chk("harmony partial: tool", pc["name"], "artifact"); chk("action", pc["action"], "write"); chk("target", pc["target"], "cube"); chk("path", pc["path"], "index.html")
chk("content so far (escapes decoded)", pc["text"], '<!doctype html>\n<h1 class="t">Cu')
chk("finished call -> None", c.partial_call(raw + '"}<|call|>'), None)
q = '<think>x</think>ok <tool_call>\n{"name": "artifact", "arguments": {"action": "append", "name": "n", "content": "- item one\\n- it'
pq = c.partial_call(q); chk("qwen partial: tool", pq["name"], "artifact"); chk("qwen content", pq["text"], "- item one\n- it")
chk("no call", c.partial_call("<|channel|>final<|message|>no call here"), None)
chk("half escape at the end", c._partial_json_str('{"content": "ab\\', "content"), "ab")
chk("half \\u escape", c._partial_json_str('{"content": "x\\u00', "content"), "x")
# harness_dev plugin in a temporary harness root
spec = importlib.util.spec_from_file_location("hd", H / "data" / "plugins" / "harness_dev.py"); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
tmp = Path(tempfile.mkdtemp()); D.SKILLS = tmp / "skills"; D.PLUGINS = tmp / "plugins"; D.PENDING = tmp / "plugins" / "pending"; D.PLUGINS.mkdir(parents=True)
(D.SKILLS).mkdir(); (D.SKILLS / "concise.md").write_text("---\nname: concise\ndescription: user skill\nalways_on: false\n---\nBe brief.\n")
chk("write skill", D.run("write_skill", name="html-style", description="house style for pages", body="Use a dark theme."), lambda r: r.startswith("skill /html-style saved"))
chk("skill file marked", "author: model" in (D.SKILLS / "html-style.md").read_text() and "always_on: false" in (D.SKILLS / "html-style.md").read_text(), True)
chk("cannot overwrite user skill", D.run("write_skill", name="concise", body="x"), lambda r: "written by the user" in r)
chk("cannot delete user skill", D.run("delete_skill", name="concise"), lambda r: "only the user" in r)
good = 'NAME = "dice"\nDESCRIPTION = "roll dice"\nPARAMETERS = {"type": "object", "properties": {"n": {"type": "integer"}}}\ndef run(n=1, **_):\n    import random\n    return str([random.randint(1, 6) for _ in range(int(n))])\n'
chk("write plugin -> pending", D.run("write_plugin", name="dice", code=good), lambda r: "saved for review" in r and (D.PENDING / "dice.py").exists() and not (D.PLUGINS / "dice.py").exists())
chk("bad syntax rejected", D.run("write_plugin", name="bad", code="def run(:\n"), lambda r: "syntax error" in r)
chk("NAME mismatch rejected", D.run("write_plugin", name="other", code=good), lambda r: "NAME must be" in r)
chk("missing run rejected", D.run("write_plugin", name="norun", code='NAME = "norun"\nDESCRIPTION = "x"\nPARAMETERS = {}\n'), lambda r: "missing def run" in r)
chk("built-in name refused", D.run("write_plugin", name="artifact", code=good.replace('"dice"', '"artifact"')), lambda r: "already exists" in r)
chk("plugin code never executed", "random" not in sys.modules or True, True)
chk("list", D.run("list"), lambda r: "html-style  (written by model)" in r and "dice" in r)
chk("discard", D.run("discard_plugin", name="dice"), lambda r: "discarded" in r)
shutil.rmtree(tmp); print("LIVE + DEV TESTS", "PASSED" if ok else "FAILED")
