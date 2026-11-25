"""assess known models and compare the estimate with what was measured on this machine"""
import sys, warnings, logging
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "app")]; warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import model_hub as mh
MEASURED = {"openai/gpt-oss-20b": 3.66, "openai/gpt-oss-120b": 2.13, "Qwen/Qwen3-30B-A3B-Instruct-2507": 1.38, "NousResearch/Hermes-4-14B": 0.9, "Qwen/Qwen3-0.6B": 54}
hw = mh.hardware()
print(f"hardware: RAM for weights {hw['ram_w'] / 1e9:.1f} GB, VRAM {hw['vram'] / 1e9:.1f} GB, vault dir free {hw['vault_free'] / 1e9:.0f} GB, download dir free {hw['dl_free'] / 1e9:.0f} GB\n")
for r in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "Qwen/Qwen3-30B-A3B-Instruct-2507", "NousResearch/Hermes-4-14B", "Qwen/Qwen3-0.6B",
          "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "google/gemma-3-4b-it", "Qwen/Qwen3-4B-AWQ", "meta-llama/Llama-3.2-1B-Instruct"]:
    a = mh.assess(r, hw); m = MEASURED.get(r)
    est = f"{a['est_tok_s']:.2f}" if a["est_tok_s"] else "-"
    print(f"{r:40s} {a['verdict']:28s} est {est:>6s} tok/s" + (f"  measured {m}" if m else "") + (f"  vault ~{a['vault_gb']:.1f} GB" if a["vault_gb"] else ""))
    if r in ("openai/gpt-oss-120b", "google/gemma-3-4b-it", "Qwen/Qwen3-4B-AWQ"):
        for q, ans, st in a["tree"]: print(f"     [{st:4s}] {q}: {ans}")
