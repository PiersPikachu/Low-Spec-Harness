from pathlib import Path
import ast
p = Path(r"C:\Users\1fred\Desktop\AI\Hermes\cube-harness\chat.py"); s = p.read_text(encoding="utf-8")
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
rep('''def moe_banner(M, vram_mb):
    return (f"{M['name']}: mixture of experts, {M['nl']} layers x {M['n_experts']} experts (top {M['top_k']}), vault {M['total'] / 1e9:.1f} GB; "
            f"attention + output layer pinned, experts cached in RAM (LRU) + {vram_mb} MB VRAM" + (" computed on the GPU" if M.get("gptoss") else "")
            + ", missing experts read from the SSD with next-layer prefetch" * bool(M.get("gptoss")))''',
'''def moe_banner(M, vram_mb, prefetch=None):
    pf = (2 if M.get("gptoss") else 0) if prefetch is None else int(prefetch)
    return (f"{M['name']}: mixture of experts, {M['nl']} layers x {M['n_experts']} experts (top {M['top_k']}), vault {M['total'] / 1e9:.1f} GB; "
            f"attention + output layer pinned, experts cached in RAM (LRU)" + (f" + {vram_mb} MB VRAM" + (" computed on the GPU" if M.get("gptoss") else "") if vram_mb > 500 else ", no VRAM tier")
            + ", missing experts read from the SSD" + (f" with next-layer prefetch ({pf} per token)" if pf > 0 else ""))''')
rep('''        print(moe_banner(M, vram))
        return args, f"mixture of experts: RAM expert cache + {vram} MB VRAM tier" + (" with GPU experts and prefetch" if M.get("gptoss") else ""), None''',
'''        print(moe_banner(M, vram, pf if M.get("gptoss") else 0))
        return args, (f"mixture of experts: RAM expert cache + {vram} MB VRAM tier" if vram > 500 else "mixture of experts: RAM expert cache") \\
            + (" with GPU experts" if M.get("gptoss") and vram > 500 else "") + (f", prefetch {int(pf)}" if M.get("gptoss") and int(pf) > 0 else ""), None''')
ast.parse(s); p.write_text(s, encoding="utf-8"); print("banner reflects the tuned settings")
