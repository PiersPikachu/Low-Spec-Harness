"""gpt-oss-20b with the recommended long-context settings (131k ctx, fp16 KV): load, RAM, a short reply, then a ~6k-token document"""
import ctypes, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
class M(ctypes.Structure): _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "a", "b", "c", "d", "e")]
def free_gb():
    m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ap / 2**30
vault = "vault_gptoss_20b_mxf2"; cfg = dict(hp.load_config()); rec, notes = core.recommended_settings(hp.resolve_vault(vault, cfg)); cfg.update(rec); cfg.update(vault=vault, greedy=True)
print("settings:", {k: cfg[k] for k in ("ctx", "max_new", "kv16", "temp")}, f"| free RAM before {free_gb():.1f} GB", flush=True)
t0 = time.time(); s = core.Session(cfg); print(f"loaded in {time.time() - t0:.0f}s | {s.how} | free RAM {free_gb():.1f} GB", flush=True)
log = (hp.LOGS / "engine.log").read_text(errors="ignore") if (hp.LOGS / "engine.log").exists() else ""
print("engine:", [l.strip()[:120] for l in log.splitlines() if "[kv]" in l or "expert cache" in l or "VRAM tier" in l][-3:], flush=True)
new, st = s.reply([{"role": "user", "content": "In one sentence, what does a KV cache store?"}], "", None, {}, lambda k, t: None, lambda: False, 0)
print("short:", ascii(new[-1]["content"])[:200], "|", st, flush=True)
doc = (H / "app" / "harness_core.py").read_text(encoding="utf-8")[:24000]
q = "Here is a Python source file:\n\n" + doc + "\n\nQuestion: what is the name of the function that archives trimmed turns, and what artifact name prefix does it use? Answer in one sentence."
t1 = time.time(); new, st = s.reply([{"role": "user", "content": q}], "", None, {}, lambda k, t: None, lambda: False, 0)
print(f"long ({len(s._ids([{'role': 'user', 'content': q}], None))} prompt tokens):", ascii(new[-1]["content"])[:300], "|", st, f"| wall {time.time() - t1:.0f}s | free RAM {free_gb():.1f} GB", flush=True)
s.close()
