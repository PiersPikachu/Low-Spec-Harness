"""Session end to end with tools on (calculator), two turns, the way the window calls it: streamed kinds, stored messages, template round trip"""
import json, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
vault = sys.argv[1]
cfg = dict(hp.load_config()); rec, _ = core.recommended_settings(hp.resolve_vault(vault, cfg)); cfg.update(rec); cfg.update(vault=vault, greedy=True, max_new=int(sys.argv[2]) if len(sys.argv) > 2 else 700)
P = core.load_plugins(); tools = core.tool_schemas(P, {"calculator"})
s = core.Session(cfg); print("loaded |", s.how, flush=True)
hist = []
for q in ["Use the calculator tool to compute 1234 * 5678, then tell me the result.", "Now divide that result by 2 using the tool."]:
    hist.append({"role": "user", "content": q}); kinds = {}
    def on_text(k, t): kinds[k] = kinds.get(k, "") + t
    t0 = time.time(); new, stats = s.reply([{k: v for k, v in m.items() if k != "meta"} for m in hist], "", tools, P, on_text, lambda: False, 3)
    print(f"\nQ: {q}\n  streamed kinds: { {k: len(v) for k, v in kinds.items()} }")
    print("  thinking preview:", ascii(kinds.get("thinking", "")[:150]))
    print("  tool lines:", ascii(kinds.get("tool", "").strip()))
    for m in new: print("  stored:", ascii({k: (v if k != "thinking" else v[:60] + '...') for k, v in m.items() if k != "meta"})[:300])
    print("  answer:", ascii(new[-1]["content"][:300]), "|", stats, f"| {time.time() - t0:.0f}s", flush=True)
    hist += new
s.close()
