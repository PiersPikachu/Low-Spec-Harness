"""does the model now write a large page in parts? (gpt-oss-20b, recommended settings, tools on; test artifact removed afterwards)"""
import shutil, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
vault = sys.argv[1]
cfg = dict(hp.load_config()); rec, _ = core.recommended_settings(hp.resolve_vault(vault, cfg)); cfg.update(rec); cfg.update(vault=vault)
P = core.load_plugins(); tools = core.tool_schemas(P, {"artifact"}); A = P["artifact"]; shutil.rmtree(A.ROOT / "rubik-test", ignore_errors=True)
s = core.Session(cfg); print("loaded |", s.how, "| max_new", cfg["max_new"], "temp", cfg["temp"], flush=True); t0 = time.time()
def on_text(k, t):
    if k == "tool": print(f"[{time.time() - t0:5.0f}s] {t.strip()[:230]}", flush=True)
new, stats = s.reply([{"role": "user", "content": "Create a page artifact named rubik-test: an interactive 3D Rubik's cube in one HTML file (CSS 3D transforms, buttons to rotate faces)."}],
                     "", tools, P, on_text, lambda: False, 6)
print("\nfinal answer:", ascii(new[-1]["content"])[:300]); print(stats, f"| total {time.time() - t0:.0f}s")
d = A.ROOT / "rubik-test"
if d.exists():
    m = A._meta(d); f = d / m["main"]; print("file:", f.name, f.stat().st_size if f.exists() else 0, "bytes; changes:", [(c["n"], c["action"]) for c in m["changes"]])
s.close(); shutil.rmtree(d, ignore_errors=True); print("test artifact removed")
