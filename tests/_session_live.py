"""real model: are live events emitted while an artifact is written, and does the model use harness_dev? (test files removed after)"""
import shutil, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
vault = sys.argv[1]
cfg = dict(hp.load_config()); rec, _ = core.recommended_settings(hp.resolve_vault(vault, cfg)); cfg.update(rec); cfg.update(vault=vault, greedy=True, max_new=1500)
P = core.load_plugins(); tools = core.tool_schemas(P, {"artifact", "harness_dev"}); A = P["artifact"]
s = core.Session(cfg); print("loaded |", s.how, flush=True)
for q in ["Create a note artifact named live-test containing a numbered list of five fruits.",
          "Use harness_dev to save a skill named fruit-facts that tells you to answer fruit questions with one fun fact."]:
    ev = []; t0 = time.time()
    new, stats = s.reply([{"role": "user", "content": q}], "", tools, P, lambda k, t: ev.append((round(time.time() - t0, 1), k, t)), lambda: False, 4)
    acts = [(ts, e["ev"], e.get("name"), e.get("action"), len(e.get("text") or "") if e["ev"] == "call_stream" else str(e.get("text", ""))[:60]) for ts, k, e in ev if k == "act"]
    print(f"\nQ: {q}\n  {stats}"); [print("   ", a) for a in acts[:14]]
    if len(acts) > 14: print(f"    ... {len(acts)} activity events in total")
s.close()
shutil.rmtree(A.ROOT / "live-test", ignore_errors=True); (hp.SKILLS / "fruit-facts.md").unlink(missing_ok=True); print("test files removed")
