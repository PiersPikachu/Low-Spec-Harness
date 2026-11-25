"""layered_vault.py: quantize offloaded weights in place through accelerate's weights_map (no 8 GB temporary copy on disk:
the first attempt filled C: when the page file grew at the same time)"""
import ast
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "research" / "layered_vault.py"; s = p.read_text(encoding="utf-8")
a = s.index("    if big:     # weights offloaded by accelerate"); b = s.index("    # quantize in place (originals are not needed afterwards")
NEW = '''    if big:     # offloaded weights live in accelerate's weights_map (CPU tensors): quantize those in place, nothing written to disk
        sz = []
        for name, mod in m.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in LINEARS:
                hook = getattr(mod, "_hf_hook", None); t = mod.weight.data
                if t.device.type == "meta": t = hook.weights_map["weight"]
                q, code, e = quantize(t, 4); layer_no = int(name.split(".layers.")[1].split(".")[0])
                if layer_no % sample == 0: sz.append((t.numel(),) + size_bits(t, code, e))
                t.copy_(q.to(t.device)); del q, code, e
        a = agreement(m)
        n = sum(x[0] for x in sz); L = sum(x[0] * x[1] for x in sz) / n; R = sum(x[0] * x[2] for x in sz) / n; T_ = sum(x[0] * x[3] for x in sz) / n
        r = dict(agree=float(np.mean(a)), tpp4=tokens_per_pass(a, 4), tpp8=tokens_per_pass(a, 8), layer_bits=L, residual_bits=R, layered_total=L + R,
                 today_lossless=T_, overhead=(L + R) / T_ - 1)
        res["mx4"] = r; print("4-bit copy:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
        out = Path(__file__).with_name("layered_vault.json"); allr = json.load(open(out)) if out.exists() else {}
        allr[src.name] = res; json.dump(allr, open(out, "w"), indent=1, default=float); print(f"done in {time.time() - t0:.0f}s"); return
'''
s = s[:a] + NEW + s[b:]; ast.parse(s); p.write_text(s, encoding="utf-8"); print("layered_vault: in-place quantization of offloaded weights")
