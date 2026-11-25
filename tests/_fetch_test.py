"""fetch_models.py end to end on a small vault: parts 'released' into a local folder (file:// URLs), downloaded, verified,
reassembled into the install folder, engine-checked; an interrupted download resumes; a corrupted part is rejected."""
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app")]
import harness_paths as hp
ok = True
def chk(label, cond):
    global ok
    ok &= bool(cond); print("PASS" if cond else "FAIL", label)
T = Path(tempfile.mkdtemp(prefix="fetchtest_")); src = H / "tests" / "data" / "moe_synth_vault"; name = "vault_fetch_test"
pkg = T / "pkg"; rel = T / "release"; inst = T / "install"; pkg.mkdir(); rel.mkdir(); inst.mkdir()
shutil.copy2(H / "models" / "gpt-oss" / "fetch_models.py", pkg / "fetch_models.py")
(pkg / name).mkdir(); small = [f.name for f in src.iterdir() if f.is_file() and f.name != "vault.bin"]
for f in small: shutil.copy2(src / f, pkg / name / f)
data = (src / "vault.bin").read_bytes(); PART = 1 << 16; parts = []
for i, off in enumerate(range(0, len(data), PART)):
    b = data[off:off + PART]; a = f"{name}.vault.bin.part{i:03d}"; (rel / a).write_bytes(b)
    parts.append({"asset": a, "offset": off, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()})
json.dump({"release_tag": "t", "part_bytes": PART, "vaults": {name: {"vault_bin_size": len(data), "vault_bin_sha256": hashlib.sha256(data).hexdigest(),
          "parts": parts, "files": small}}}, open(pkg / "manifest.json", "w"))
# point the script at this package and make it install into the temp folder
s = (pkg / "fetch_models.py").read_text(encoding="utf-8")
s = s.replace('ROOT = HERE.parent.parent; sys.path.insert(0, str(ROOT / "app"))', f'ROOT = Path(r"{H}"); sys.path.insert(0, str(ROOT / "app"))')
s = s.replace('NAMES = {"20b": "vault_gptoss_20b_mxf2", "120b": "vault_gptoss_120b_mxf2"}', f'NAMES = {{"20b": "{name}"}}')
(pkg / "fetch_models.py").write_text(s, encoding="utf-8")
saved = hp.load_config(); hp.add_model_dir(inst, default=True)
try:
    base = rel.as_uri()
    r = subprocess.run([sys.executable, str(pkg / "fetch_models.py"), "20b", "--base", base], capture_output=True, text=True); print("   ", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
    out = inst / name / "vault.bin"
    chk(f"{len(parts)} parts downloaded, reassembled byte-exact", out.exists() and out.read_bytes() == data)
    chk("engine check passed on the rebuilt vault", "bad=0" in r.stdout)
    # resume: cut the file in the middle of part 2, run again
    with open(out, "r+b") as f: f.truncate(PART * 2 + 12345)
    r = subprocess.run([sys.executable, str(pkg / "fetch_models.py"), "20b", "--base", base], capture_output=True, text=True)
    chk("interrupted download resumed from the last whole part: " + (r.stdout.strip().splitlines() or [""])[0], f"2/{len(parts)} parts already here" in r.stdout and out.read_bytes() == data)
    # a corrupted part on the server is rejected
    out.unlink(); bad = rel / parts[1]["asset"]; bb = bytearray(bad.read_bytes()); bb[100] ^= 1; bad.write_bytes(bytes(bb))
    r = subprocess.run([sys.executable, str(pkg / "fetch_models.py"), "20b", "--base", base], capture_output=True, text=True)
    chk("corrupted part rejected", r.returncode != 0 and "failed verification" in (r.stdout + r.stderr))
finally:
    hp.save_config(saved); shutil.rmtree(T, ignore_errors=True)
print("FETCH TESTS", "PASSED" if ok else "FAILED")
