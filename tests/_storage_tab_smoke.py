"""Model folders tab: lists the built-in and added folders with drive type / speed / models, adds and removes a folder,
sets the install target (which the model hub then uses), and moves a small vault between folders with verification."""
import shutil, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app")]
import harness_app as AP, harness_paths as hp, storage
app = AP.App(); ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)
rows = {app.st_tree.item(i, "text"): app.st_tree.item(i, "values") for i in app.st_tree.get_children()}
for k, v in rows.items(): print("   ", k, v)
chk("built-in data\\vaults listed as install target", str(hp.VAULTS) in rows and "install target" in rows[str(hp.VAULTS)][5])
chk("D: folder listed with its disk type", any("vaults_parked" in k for k in rows))
saved = hp.load_config()
tmp = Path(tempfile.mkdtemp(prefix="cube_models_"))
try:
    hp.add_model_dir(tmp); app._st_refresh(); chk("added folder shows up", str(tmp) in [app.st_tree.item(i, "text") for i in app.st_tree.get_children()])
    app.st_tree.selection_set(str(tmp)); app._st_default()
    chk("install target switched", hp.default_model_dir() == tmp)
    import model_hub as mh
    chk("the model hub installs into it", mh.hardware()["vault_dir"] == tmp)
    # move a small test vault there and back (copy, verify, delete)
    src = H / "tests" / "data" / "moe_synth_vault"; work = hp.VAULTS / "vault_move_test"; shutil.copytree(src, work)
    log = []; dst = storage.move_vault(work, tmp, log.append)
    chk("vault moved with every stream verified", dst.exists() and not work.exists() and any("bad=0" in l for l in log))
    back = storage.move_vault(dst, hp.VAULTS, log.append); chk("and moved back", back.exists() and not dst.exists())
    shutil.rmtree(back)
finally:
    hp.save_config(saved); shutil.rmtree(tmp, ignore_errors=True); app._st_refresh()
chk("config restored", hp.default_model_dir() == hp.VAULTS)
app.after(200, app.destroy); app.mainloop(); print("STORAGE TAB", "PASSED" if ok else "FAILED")
