"""Artifacts tab smoke test (temporary artifacts root): list, select, edit + Save (versioned), revert, model-side refresh hook"""
import sys, tempfile, shutil
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_app as AP
app = AP.App(); A = app.plugins["artifact"]; real = A.ROOT; A.ROOT = Path(tempfile.mkdtemp()) / "artifacts"; ok = True
try:
    A.run("create", name="demo-page", type="page", title="Demo page", content="<!doctype html><h1>Hello</h1>\n")
    A.run("create", name="ideas", type="note", title="Ideas", content="# Ideas\n- one\n")
    app._refresh_artifacts(); ok &= app.art_names[:2] == ["ideas", "demo-page"] or set(app.art_names) == {"ideas", "demo-page"}
    print("list:", app.art_list.get(0, "end"))
    app._art_show("demo-page"); ok &= app.art_path == "index.html" and "<h1>Hello</h1>" in app.art_text.get("1.0", "end")
    print("shown:", app.art_head.get(), "|", app.art_file_var.get()[:40])
    app.art_text.delete("1.0", "end"); app.art_text.insert("1.0", "<!doctype html><h1>Hello again</h1>\n"); app._art_save()
    body = (A.ROOT / "demo-page" / "index.html").read_text(); ok &= "Hello again" in body; print("after Save:", repr(body), "| status:", app.status.get())
    hist = app.art_hist.get(0, "end"); print("history:", hist); ok &= hist and hist[0].startswith("#1")
    A.run("revert", name="demo-page", version=1); app._refresh_artifacts(); body = (A.ROOT / "demo-page" / "index.html").read_text(); ok &= "Hello again" not in body
    print("after revert:", repr(body)); ok &= "Hello again" not in app.art_text.get("1.0", "end")
    A.run("write", name="ideas", path="extra/more.md", content="more"); app._on_tool("[artifact] write extra/more.md\n")
    app._art_show("ideas"); print("files now:", app.art_files.get(0, "end")); ok &= "extra/more.md" in app.art_files.get(0, "end")
finally:
    shutil.rmtree(A.ROOT.parent); A.ROOT = real
app.after(200, app.destroy); app.mainloop(); print("ARTIFACTS TAB", "PASSED" if ok else "FAILED")
