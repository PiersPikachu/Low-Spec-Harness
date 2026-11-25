"""make_core_live.py - harness_core.py / chat.py: live activity events, in-progress tool-call streaming, force stop.

  partial_call(raw)  the tool call the model is writing right now (harmony or <tool_call>), with the fields decoded so far,
                     including an unterminated "content"/"new" string - so the window can show an artifact being written live
  Session.reply      emits on_text("act", {...}) events: round, call_stream (throttled), call, result
  Session.kill()     kills the engine process(es) immediately (Force stop); the session is unusable afterwards
  skills             'author' front-matter key kept (model-written skills are marked)"""
import ast
from pathlib import Path
H = Path(__file__).resolve().parent
P = H / "harness_core.py"; s = P.read_text(encoding="utf-8")
def rep(old, new, src=None):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

LIVE = r'''

def _partial_json_str(text, key):
    """value of "key": "..." in possibly unterminated JSON text, decoded as far as it goes (None if the key has not started)"""
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', text)
    if not m: return None
    out, i, n = [], m.end(), len(text)
    esc = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f"}
    while i < n:
        ch = text[i]
        if ch == '"': break
        if ch == "\\":
            if i + 1 >= n: break
            c = text[i + 1]
            if c == "u":
                if i + 6 > n: break
                try: out.append(chr(int(text[i + 2:i + 6], 16)))
                except ValueError: pass
                i += 6; continue
            out.append(esc.get(c, c)); i += 2; continue
        out.append(ch); i += 1
    return "".join(out)


def partial_call(raw):
    """the tool call being written at the end of raw, or None -> dict(name, action, target, path, text, chars)"""
    name = body = None
    if "<|channel|>" in raw or "<|message|>" in raw:
        seg = raw.split("<|start|>")[-1]
        if "<|message|>" in seg:
            head, b = seg.split("<|message|>", 1); to = re.search(r"to=functions\.([\w.\-]+)", head)
            if to and not any(e in b for e in HARMONY_END): name, body = to.group(1), b
    else:
        i = raw.rfind("<tool_call>")
        if i >= 0 and "</tool_call>" not in raw[i:]:
            body = raw[i + len("<tool_call>"):]; nm = _partial_json_str(body, "name"); name = nm if nm else "?"
    if body is None: return None
    f = lambda k: _partial_json_str(body, k)
    text = f("content")
    if text is None: text = f("new")
    if text is None: text = f("code")
    if text is None: text = f("body")
    return dict(name=name, action=f("action"), target=f("name") if name != "?" else None, path=f("path"), text=text, chars=len(body))
'''
rep("\n\n# ------------------------------------------------------------------ recommended settings per model", LIVE + "\n\n# ------------------------------------------------------------------ recommended settings per model")
# live events in reply
rep('''            shown = {"thinking": "", "assistant": ""}
            def on_tok(out):                    # stream reasoning and answer separately; each only ever grows
                r = parse_reply(self._decode(out))''',
'''            shown = {"thinking": "", "assistant": ""}; last_pc = [0.0, -1]
            on_text("act", {"ev": "round", "n": rnd + 1})
            def on_tok(out):                    # stream reasoning and answer separately; each only ever grows
                raw_now = self._decode(out); r = parse_reply(raw_now)
                pc = partial_call(raw_now) if tools else None
                if pc and (time.time() - last_pc[0] > 0.12 or pc["chars"] - last_pc[1] > 400):
                    last_pc[0] = time.time(); last_pc[1] = pc["chars"]; on_text("act", dict(pc, ev="call_stream"))''')
rep('''            for name, args in calls:''', '''            for name, args in calls:
                on_text("act", {"ev": "call", "name": name, "args": args})''')
rep('''                new_msgs.append({"role": "tool", "content": res, "meta": {"tool": name, "args": args}}); on_text("tool", f"[{name}] {res}\\n")''',
'''                new_msgs.append({"role": "tool", "content": res, "meta": {"tool": name, "args": args}}); on_text("tool", f"[{name}] {res}\\n")
                on_text("act", {"ev": "result", "name": name, "args": args, "text": res})''')
# force stop
rep('''    def close(self):
        try: self.eng.close()
        except Exception: pass''', '''    def close(self):
        try: self.eng.close()
        except Exception: pass

    def kill(self):
        """Force stop: kill the engine process(es) now; a generation in progress fails at once and this session must be reloaded"""
        for p in [getattr(self.eng, "p", None)] + [getattr(getattr(self.eng, a, None), "p", None) for a in ("T", "D")]:
            try:
                if p is not None and p.poll() is None: p.kill()
            except Exception: pass''')
# skills: keep the author (model-written skills are marked in the Skills tab)
rep('''        out[name] = {"name": name, "description": meta.get("description", ""), "always_on": meta.get("always_on", "false").lower() == "true",''',
'''        out[name] = {"name": name, "description": meta.get("description", ""), "always_on": meta.get("always_on", "false").lower() == "true", "author": meta.get("author", ""),''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_core: live activity events, partial tool calls, Session.kill, skill author")
