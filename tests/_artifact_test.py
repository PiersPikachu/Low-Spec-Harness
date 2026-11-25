"""artifact plugin unit test in a temporary root (does not touch the real artifacts/ folder)"""
import importlib.util, shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("artifact", H / "data" / "plugins" / "artifact.py"); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
A.ROOT = Path(tempfile.mkdtemp()) / "artifacts"; ok = True
def chk(label, out, want):
    global ok
    good = want in out; ok &= good; print("PASS" if good else "FAIL", label, "->", out.splitlines()[0][:110] if out else out)
chk("list empty", A.run("list"), "no artifacts yet")
chk("create note", A.run("create", name="Trip Notes", type="note", title="Trip notes"), "created artifact 'trip-notes' (note) with notes.md")
chk("append", A.run("append", name="trip-notes", content="- book ferry"), "append notes.md")
chk("read", A.run("read", name="trip-notes"), "notes.md (3 lines)")
chk("edit unique", A.run("edit", name="trip-notes", old="book ferry", new="book ferry (Friday)"), "edited notes.md (change #2)")
chk("edit missing", A.run("edit", name="trip-notes", old="nothing here", new="x"), "found 0 times")
chk("create site", A.run("create", name="demo-site", type="site", title="Demo"), "created artifact 'demo-site' (site) with index.html")
chk("write css", A.run("write", name="demo-site", path="css/style.css", content="body{font-family:sans-serif}"), "write css/style.css")
chk("files", A.run("files", name="demo-site"), "css/style.css")
chk("escape blocked", A.run("write", name="demo-site", path="../../evil.txt", content="x"), "must stay inside")
chk("versions dir blocked", A.run("read", name="demo-site", path=".versions/00001.prev"), "give a file path inside")
chk("history", A.run("history", name="trip-notes"), "#1")
chk("revert edit", A.run("revert", name="trip-notes", version=2), "restored to how it was before change #2")
chk("reverted text", A.run("read", name="trip-notes"), "notes.md")
txt = (A.ROOT / "trip-notes" / "notes.md").read_text(); good = "book ferry" in txt and "(Friday)" not in txt; ok &= good; print("PASS" if good else "FAIL", "content after revert:", repr(txt))
chk("revert new file", A.run("revert", name="demo-site", version=1), "removed (it did not exist before change #1)")
good = not (A.ROOT / "demo-site" / "css" / "style.css").exists(); ok &= good; print("PASS" if good else "FAIL", "file created by change #1 removed by revert")
chk("unknown artifact", A.run("read", name="nope"), "no artifact named 'nope'")
chk("list", A.run("list"), "demo-site  [site]")
shutil.rmtree(A.ROOT.parent); print("ARTIFACT TESTS", "PASSED" if ok else "FAILED")
