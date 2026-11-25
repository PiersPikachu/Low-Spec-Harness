"""make_app_cap.py - Config setting 'Max file growth per reply (characters, 0 = 4 x max reply)'; default 0 in harness_paths."""
import ast
from pathlib import Path
H = Path(__file__).resolve().parent
def patch(fname, old, new):
    p = H / fname; s = p.read_text(encoding="utf-8"); assert s.count(old) == 1, (fname, s.count(old)); s = s.replace(old, new); ast.parse(s); p.write_text(s, encoding="utf-8"); print("patched", fname)
patch("harness_app.py", '''        ("max_new", "Max reply tokens", "int", (16, 131072)),''',
      '''        ("max_new", "Max reply tokens", "int", (16, 131072)), ("max_file_growth", "Max file growth per reply (chars, 0 = 4 x max reply)", "int", (0, 2000000)),''')
patch("harness_paths.py", '''"ctx": 2048, "max_new": 512, "kv16": False,''', '''"ctx": 2048, "max_new": 512, "kv16": False, "max_file_growth": 0,''')
