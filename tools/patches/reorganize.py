"""reorganize.py - one-time move of the cube-harness folder into sub-folders (old -> new recorded in tools/reorg_manifest.txt).

  cube-harness/
    cube-harness.bat, README.md      the only files at the top
    app/        the harness itself (window, core, paths, engine launcher, planner, model hub, setup, launch check)
    engine/     nvme_engine_u.exe + gpu_experts.dll, their C/CUDA sources and the make_engine_* patch chain, gpu/ (GPU engine)
    tools/      packers and vault tools; research/ (experiments + results), patches/ (one-time code patches), bat/ (old launchers)
    tests/      test scripts; tests/data/ small test vaults and reference outputs
    data/       everything the harness writes: vaults/, artifacts/, conversations/, skills/, plugins/, workspace/, hub_cache/,
                packages/, logs/, harness_config.json, hw_profile.json, drafts.txt
Same volume, so every move is a rename (the 73 GB of vaults move instantly). Run once with no engine or harness running."""
import ast, re, shutil, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
running = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
if "nvme_engine_u.exe" in running or "gpu_engine.exe" in running: sys.exit("an engine is running - close the harness first")
moves = []
def mv(src, dst):
    s, d = R / src, R / dst
    if not s.exists(): return
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists() and d.is_dir() and s.is_dir():
        for c in list(s.iterdir()): mv(str(Path(src) / c.name), str(Path(dst) / c.name))
        s.rmdir(); return
    shutil.move(str(s), str(d)); moves.append(f"{src} -> {dst}")

APP = ["harness_app.py", "harness_core.py", "harness_paths.py", "chat.py", "spec.py", "planner.py", "model_hub.py", "setup_harness.py"]
TOOLS = ["pack_universal_v3.py", "pack_gptoss.py", "rans4.py", "transcode_vault.py", "vault_defrag.py", "park_vault.py", "make_mini.py", "verify_engine.py"]
RESEARCH = ["moe_bench.py", "moe_locality.py", "moe_policy_sim.py", "moe_synth.py", "moe_verify.py", "gptoss_ref.py", "mxfp4_entropy.py", "scale_bound.py",
            "spec_sweep.py", "tree_sim.py", "pass_cost.py", "test_specp.py", "lb_probe.py", "lb_train.py", "run_14b.py", "hf_compare.py", "gen_compare.py", "make_pack_moe.py"]
for f in APP: mv(f, f"app/{f}")
for f in TOOLS: mv(f, f"tools/{f}")
for f in RESEARCH: mv(f, f"tools/research/{f}")
for p in sorted(R.glob("make_engine_*.py")) + sorted(R.glob("make_gpu_*.py")): mv(p.name, f"engine/{p.name}")
for f in ["nvme_engine_u.c", "nvme_engine_u.exe", "gpu_experts.dll", "gpu_experts.lib", "gpu_experts.exp", "build.bat"]: mv(f, f"engine/{f}")
mv("gpu", "engine/gpu")
for p in sorted(R.glob("make_core_*.py")) + sorted(R.glob("make_app_*.py")) + [R / "make_chat_stop.py", R / "make_harness_kv.py"]: mv(p.name, f"tools/patches/{p.name}")
for f in ["auto_serve.bat", "chat.bat", "pack_rans4.bat", "run_14b.bat", "spec_14b.bat", "verify_rans4.bat", "setup.bat"]: mv(f, f"tools/bat/{f}")
for d in ["vaults", "artifacts", "conversations", "skills", "plugins", "workspace", "hub_cache", "packages"]: mv(d, f"data/{d}")
for f in ["harness_config.json", "hw_profile.json", "drafts.txt"]: mv(f, f"data/{f}")
# research results and outputs that sat at the top
for p in list(R.iterdir()):
    if p.is_file() and (p.suffix in (".json", ".pt", ".err") or (p.suffix == ".txt" and p.name not in ("README.txt",))) and p.name != "nvme_stream_log.txt":
        mv(p.name, f"tools/research/results/{p.name}")
mv("nvme_stream_log.txt", "data/logs/nvme_stream_log.txt"); mv("ref_logits.npy", "tests/data/ref_logits.npy")
mv("wsp", "tools/research/wsp"); mv("models", "tools/research/models")
# logs/: tests -> tests/, test data -> tests/data/, my one-off fix scripts -> tools/patches/, backups -> data/logs/backups/, rest -> data/logs/
L = R / "logs"
if L.exists():
    for p in sorted(L.iterdir()):
        n = p.name
        if n.startswith("_fix") or n == "pack_qwen30b_when_ready.py": mv(f"logs/{n}", f"tools/patches/{n}")
        elif p.suffix == ".py": mv(f"logs/{n}", f"tests/{n}")
        elif n in ("gptoss_mini2_vault", "gptoss_mini2_vault_mxf2", "moe_synth_model", "moe_synth_vault", "qwen30b_trace.bin", "gptoss120b_trace.bin",
                   "gptoss_ids.txt", "gptoss_hf_logits.npy"): mv(f"logs/{n}", f"tests/data/{n}")
        elif ".before_" in n: mv(f"logs/{n}", f"data/logs/backups/{n}")
        else: mv(f"logs/{n}", f"data/logs/{n}")
    if not any(L.iterdir()): L.rmdir()
for p in list(R.rglob("__pycache__")): shutil.rmtree(p, ignore_errors=True)
if (R / "Cube Harness.bat").exists(): (R / "Cube Harness.bat").unlink(); moves.append("Cube Harness.bat -> (replaced by cube-harness.bat)")
mv("reorganize.py", "tools/patches/reorganize.py")
(R / "tools").mkdir(exist_ok=True); (R / "tools" / "reorg_manifest.txt").write_text("\n".join(moves) + "\n", encoding="utf-8")
print(f"{len(moves)} moves recorded in tools/reorg_manifest.txt")
left = sorted(p.name for p in R.iterdir())
print("top level now:", left)
