"""the harness Session end to end on a MoE vault (what the window does when you press Load + Send)"""
import sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
cfg = dict(hp.load_config()); cfg.update(vault=sys.argv[1], max_new=int(sys.argv[2]) if len(sys.argv) > 2 else 80, greedy=True)
t0 = time.time(); s = core.Session(cfg); print("loaded in", round(time.time() - t0, 1), "s |", s.how); print(s.banner, flush=True)
out = []
msgs, stats = s.reply([{"role": "user", "content": "In two sentences, what is a mixture-of-experts language model?"}], "", None, {}, lambda k, t: out.append(t) if isinstance(t, str) else None, lambda: False)
print("STREAMED:", ascii("".join(out))[:900]); print("STORED:", ascii([{k: v for k, v in m.items() if k != "meta"} for m in msgs])[:700]); print("STATS:", stats); s.close()
