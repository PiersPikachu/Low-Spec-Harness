"""window smoke test for the live features (no model): activity tree, live artifact view, Force stop, plugin review queue"""
import sys, tempfile, shutil, types
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_app as AP, harness_paths as hp, harness_core as core
app = AP.App(); ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)
A = app.plugins["artifact"]; real = A.ROOT; A.ROOT = Path(tempfile.mkdtemp()) / "artifacts"
try:
    A.run("create", name="cube", type="page", title="Cube")
    app.conv = core.new_conversation("x"); app.session = types.SimpleNamespace(how="test", vault=Path("m"), kill=lambda: None, close=lambda: None); app._begin(); app._on_start("model")
    app._on_act({"ev": "round", "n": 1}); app._on_thinking("I will write the page with a canvas. ")
    for part in ("<!doctype html>\n<canvas", "<!doctype html>\n<canvas id='c'></canvas>\n<script>"):
        app._on_act({"ev": "call_stream", "name": "artifact", "action": "write", "target": "cube", "path": "index.html", "text": part, "chars": len(part) + 60})
    t = app.act["tree"]; rows = [t.item(i, "text") for i in t.get_children(t.get_children()[0])]
    chk("tree: round with thinking + call nodes " + str(rows), len(rows) == 2 and rows[0].startswith("\u25cb thinking") and "artifact" in rows[1] and "cube/index.html" in rows[1])
    chk("call status shows live size", "writing ..." in t.item(t.get_children(t.get_children()[0])[1], "values")[0])
    chk("Artifacts tab shows the file being written", app.art_text.get("1.0", "end-1c").endswith("<script>") and "LIVE" in app.art_head.get())
    app._on_act({"ev": "call", "name": "artifact", "args": {"action": "write", "name": "cube", "path": "index.html", "content": "<p>done</p>"}})
    A.run("write", name="cube", path="index.html", content="<p>done</p>")
    app._on_act({"ev": "result", "name": "artifact", "args": {}, "text": "write index.html: 11 chars (change #1)"}); app._on_tool("[artifact] write index.html\n")
    chk("result marks the call done", t.item(t.get_children(t.get_children()[0])[1], "values")[0].startswith("\u2713"))
    chk("tab shows the saved file afterwards", app.art_text.get("1.0", "end-1c") == "<p>done</p>")
    app._on_act({"ev": "round", "n": 2}); app._on_assistant("The page is ready."); app._on_done(([{"role": "assistant", "content": "The page is ready."}], "stats"))
    chk("rounds marked done", [t.item(i, "values")[0] for i in t.get_children()] == ["done", "done"])
    # Force stop: kill is called, partial reply kept, session dropped, buttons released
    killed = []; app.session = types.SimpleNamespace(kill=lambda: killed.append(1), close=lambda: None); app.session_key = "k"
    app._begin(); app._on_start("model"); app._on_assistant("Partial answer"); app.force_stop()
    app._on_error("Traceback (most recent call last):\nSystemExit: engine stopped unexpectedly")
    chk("Force stop killed the engine", killed == [1])
    chk("partial reply saved", app.conv["messages"][-1]["content"].startswith("Partial answer") and "[force-stopped]" in app.conv["messages"][-1]["content"])
    chk("session dropped, buttons idle", app.session is None and not app.busy and str(app.kill_btn["state"]) == "disabled")
finally:
    shutil.rmtree(A.ROOT.parent); A.ROOT = real
    core.delete_conversation(app.conv["id"])
# review queue: a model-written plugin is listed, approved (moved + enabled), then cleaned up
D = app.plugins["harness_dev"]
print(D.run("write_plugin", name="smoke_dice", code='NAME = "smoke_dice"\nDESCRIPTION = "roll a die (smoke test)"\nPARAMETERS = {"type": "object", "properties": {}}\ndef run(**_):\n    return "4"\n'))
app._refresh_pending(); chk("pending plugin listed", "smoke_dice" in app.pend_tree.get_children())
chk("not loaded before approval", "smoke_dice" not in core.load_plugins())
AP.messagebox.askyesno = lambda *a, **k: True; app.pend_tree.selection_set("smoke_dice"); app._pend_approve()
chk("approved: loaded, enabled, callable", "smoke_dice" in app.plugins and "smoke_dice" in app.cfg["enabled_plugins"] and core.run_tool(app.plugins, "smoke_dice", {}) == "4")
(hp.PLUGINS / "smoke_dice.py").unlink(); c = hp.load_config(); c["enabled_plugins"] = [p for p in c["enabled_plugins"] if p != "smoke_dice"]; hp.save_config(c)
print(D.run("write_skill", name="smoke-skill", description="x", body="y")); app._dev_changed()
chk("model skill marked in Skills tab", any("smoke-skill" in x and "[by model]" in x for x in app.slist.get(0, "end")))
(hp.SKILLS / "smoke-skill.md").unlink()
app.after(200, app.destroy); app.mainloop(); print("LIVE APP SMOKE", "PASSED" if ok else "FAILED")
