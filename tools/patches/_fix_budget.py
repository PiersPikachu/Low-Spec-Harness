from pathlib import Path
import ast
p = Path(r"C:\Users\1fred\Desktop\AI\Hermes\cube-harness\harness_core.py"); s = p.read_text(encoding="utf-8")
old = '        budget = 3.5e9 if (fits or not moe) and gptoss else 1.0e9'
new = '        budget = 3.5e9 if gptoss and (fits or not moe) else (1.0e9 if moe else 2.0e9)      # dense: weights stream, 2 GB of KV is affordable'
assert s.count(old) == 1; s = s.replace(old, new)
old2 = '        if gptoss: s["max_new"] = 8192 if fits else 4096;'
assert s.count(old2) == 1
s = s.replace(old2, '        s["max_new"] = min(s.get("max_new", 2048), cap // 2)         # always leave room for the prompt\n' + old2)
ast.parse(s); p.write_text(s, encoding="utf-8"); print("budget / max_new fixed")
