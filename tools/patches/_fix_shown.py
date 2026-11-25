from pathlib import Path
import ast
p = Path(r"C:\Users\1fred\Desktop\AI\Hermes\cube-harness\harness_core.py"); s = p.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:60], s.count(old)); s = s.replace(old, new)
rep('''            shown = {"thinking": "", "assistant": ""}; last_pc = [0.0, -1];''',
    '''            shown = carry if carry else {"thinking": "", "assistant": ""}; carry = None; last_pc = [0.0, -1];''')
rep('''A = plugins.get("artifact"); cont = []; conts = 0''', '''A = plugins.get("artifact"); cont = []; conts = 0; carry = None''')
rep('''                cont = out; conts += 1; budget += 1; on_text("act", {"ev": "round", "n": rnd + 2})       # compact the history and carry on''',
    '''                cont = out; conts += 1; budget += 1; carry = shown; on_text("act", {"ev": "round", "n": rnd + 2})       # compact the history and carry on''')
ast.parse(s); p.write_text(s, encoding="utf-8"); print("continuation keeps the streamed text")
