"""the window's model banner shows the tuned MoE settings (prefetch, VRAM tier) like the engine launcher does"""
import ast
from pathlib import Path
R = Path(__file__).resolve().parents[2]
p = R / "app" / "harness_core.py"; s = p.read_text(encoding="utf-8")
old = '''        if M.get("moe"): self.banner = chat.moe_banner(M, max(0, int(H.get("vram_free_mb", 0)) - 900))'''
new = '''        if M.get("moe"):
            try: _t = json.load(open(Path(self.vault) / "tuned.json"))["settings"]
            except Exception: _t = {}
            _pf = cfg.get("moe_prefetch"); _pf = _t.get("moe_prefetch", 2 if M.get("gptoss") else 0) if _pf is None else _pf
            _vr = cfg.get("moe_vram"); _vr = _t.get("moe_vram", True) if _vr is None else _vr
            self.banner = chat.moe_banner(M, max(0, int(H.get("vram_free_mb", 0)) - 900) if _vr else 0, _pf if M.get("gptoss") else 0)'''
assert s.count(old) == 1; s = s.replace(old, new); ast.parse(s); p.write_text(s, encoding="utf-8")
t = R / "tests" / "_session_moe.py"; s = t.read_text(encoding="utf-8")
s = s.replace("lambda k, t: out.append(t)", "lambda k, t: out.append(t) if isinstance(t, str) else None"); t.write_text(s, encoding="utf-8")
print("banner uses the tuned settings; test fixed")
