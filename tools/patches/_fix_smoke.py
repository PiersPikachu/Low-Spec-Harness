import re
p = r"C:\Users\1fred\Desktop\AI\Hermes\cube-harness\logs\_live_app_smoke.py"
s = open(p, encoding="utf-8", errors="replace").read()
s = re.sub(r'rows\[0\]\.startswith\("[^"]*thinking"\)', r'rows[0].startswith("\\u25cb thinking")', s)
s = re.sub(r'\[0\]\.startswith\("[^"]{1,4}"\)\)', r'[0].startswith("\\u2713"))', s)
open(p, "w", encoding="utf-8").write(s)
print([l.strip()[:110] for l in s.splitlines() if "u25cb" in l or "u2713" in l])
