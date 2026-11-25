#!/usr/bin/env python3
"""make_mini.py <hf_model_dir> <out_dir> [--layers 3] - a small checkpoint made of the REAL first N layers of a large model
(plus its embedding, final norm and lm_head), so the engine can be checked against Hugging Face on real weights without the
RAM to load the whole model. Tensors are copied byte for byte (any dtype, including MXFP4 blocks/scales); config gets num_hidden_layers=N
(and layer_types cut to N when present). Tokenizer files are copied too."""
import json, re, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2]); N = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 3
out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(src / "config.json")); cfg["num_hidden_layers"] = N
if "layer_types" in cfg: cfg["layer_types"] = cfg["layer_types"][:N]
json.dump(cfg, open(out / "config.json", "w"), indent=1)
idx = src / "model.safetensors.index.json"
wm = json.load(open(idx))["weight_map"] if idx.exists() else {k: "model.safetensors" for k in safe_open(str(src / "model.safetensors"), "pt").keys()}
keep = {}
for name, shard in wm.items():
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None or int(m[1]) < N: keep.setdefault(shard, []).append(name)
T = {}
for shard, names in keep.items():
    with safe_open(str(src / shard), "pt") as f:
        for n in names: T[n] = f.get_tensor(n)
save_file(T, str(out / "model.safetensors"), metadata={"format": "pt"})
for f in src.iterdir():
    if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json", "model.safetensors.index.json"): shutil.copy2(f, out / f.name)
print(f"wrote {out}: {len(T)} tensors, {sum(t.numel() * t.element_size() for t in T.values()) / 1e9:.2f} GB, {N} layers")
