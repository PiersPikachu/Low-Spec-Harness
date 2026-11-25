"""make_core_chunk.py - harness_core.Session.reply: chunked artifact writing, live saving, history compaction, real Stop.

  chunking   while the model streams an artifact write/append/create, once the content reaches CHUNK (3000) characters the
             harness waits for the end of the current line, stops generation there (engine STOP), runs the call with that part,
             and tells the model exactly where to continue ('append' from after line X). A call cut off by the max-new-tokens
             limit is saved the same way (up to its last complete line) instead of failing. Continuation rounds do not count
             against max_tool_rounds (cap: MAX_CHUNKS).
  live save  the content-so-far is written into the artifact file every LIVE_EVERY seconds (artifact.live_update); just before
             the real call runs, the file is put back (live_end) so the history gets one clean change per call
  compaction older tool-call arguments in the prompt are shortened to a stub ('[N characters written to the artifact]'), so a
             long chunked file does not push the user's request out of the context window; the newest call stays complete
  Stop       the Stop button now ends a reply mid-generation on the plain engine too (STOP after the next token)"""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_core.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

HELP = r'''

CHUNK = 3000          # characters of file content per tool call before the harness cuts at the end of the line
HARD_CHUNK = 12000    # cut anyway (one very long line, e.g. minified code)
MAX_CHUNKS = 60
LIVE_EVERY = 0.5      # seconds between live saves of an artifact being written


def _chunk_cut(text, final=False):
    """where to cut streamed file content: end of the line that crosses CHUNK (or HARD_CHUNK); at the end of a reply, the last newline"""
    if final:
        i = text.rfind("\n"); return len(text) if i < 0 else i + 1
    if len(text) >= HARD_CHUNK: return HARD_CHUNK
    if len(text) < CHUNK: return None
    i = text.find("\n", CHUNK); return None if i < 0 else i + 1


def _compact_history(msgs, keep_last=1):
    """shorten the file content of older tool calls (the files hold it); the newest keep_last calls stay complete"""
    idx = [i for i, m in enumerate(msgs) if m.get("tool_calls") or (m["role"] == "assistant" and "<tool_call>" in (m.get("content") or ""))]
    out = list(msgs)
    for i in idx[:-keep_last] if keep_last else idx:
        m = dict(out[i])
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                f = dict(tc.get("function", {})); a = dict(f.get("arguments") or {})
                for k in ("content", "new", "code", "body"):
                    if isinstance(a.get(k), str) and len(a[k]) > 300: a[k] = a[k][:120] + f" ... [{len(a[k])} characters, saved by the tool]"
                f["arguments"] = a; tcs.append(dict(tc, function=f))
            m["tool_calls"] = tcs
        else:
            m["content"] = re.sub(r'("(?:content|new|code|body)"\s*:\s*")((?:[^"\\]|\\.){300,})"',
                                  lambda g: g.group(1) + g.group(2)[:120] + f' ... [{len(g.group(2))} characters, saved by the tool]"', m["content"])
        out[i] = m
    return out
'''
rep("\n\n# ------------------------------------------------------------------ recommended settings per model", HELP + "\n\n# ------------------------------------------------------------------ recommended settings per model")
# partial_call: also report type/title (needed to live-create an artifact)
rep('''    return dict(name=name, action=f("action"), target=f("name") if name != "?" else None, path=f("path"), text=text, chars=len(body))''',
    '''    return dict(name=name, action=f("action"), target=f("name") if name != "?" else None, path=f("path"), text=text, chars=len(body), type=f("type"), title=f("title"))''')

# ---- the reply loop
rep('''        for rnd in range(max_tool_rounds + 1):
            msgs = ([{"role": "system", "content": system}] if system else []) + [m for m in history + new_msgs if m["role"] in ("user", "assistant", "tool")]
            if self.chat_mode:
                ids = self._ids(msgs, tools)
                while len(ids) + int(cfg["max_new"]) > int(cfg["ctx"]) and sum(1 for m in msgs if m["role"] != "system") > 1:
                    del msgs[1 if system else 0]; ids = self._ids(msgs, tools)          # drop the oldest turns to fit the context''',
'''        rnd = -1; budget = max_tool_rounds + 1; chunks = 0; A = plugins.get("artifact")
        if A is not None and hasattr(A, "live_keep"): A.live_keep()       # forget restore points of an interrupted reply (its live saves stay)
        while rnd + 1 < budget:
            rnd += 1
            msgs = ([{"role": "system", "content": system}] if system else []) + [m for m in history + new_msgs if m["role"] in ("user", "assistant", "tool")]
            msgs = _compact_history(msgs)
            if self.chat_mode:
                ids = self._ids(msgs, tools)
                last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=0)
                while len(ids) + int(cfg["max_new"]) > int(cfg["ctx"]):
                    old = next((i for i in range(1 if system else 0, last_user)), None)       # never drop the current request
                    if old is None: break
                    del msgs[old]; last_user -= 1; ids = self._ids(msgs, tools)''')
rep('''            shown = {"thinking": "", "assistant": ""}; last_pc = [0.0, -1]''',
    '''            shown = {"thinking": "", "assistant": ""}; last_pc = [0.0, -1]; cut = {"at": None, "pc": None, "ui": False, "live": 0.0, "lkey": None}''')
rep('''                if pc and (time.time() - last_pc[0] > 0.12 or pc["chars"] - last_pc[1] > 400):
                    last_pc[0] = time.time(); last_pc[1] = pc["chars"]; on_text("act", dict(pc, ev="call_stream"))''',
'''                if pc and (time.time() - last_pc[0] > 0.12 or pc["chars"] - last_pc[1] > 400):
                    last_pc[0] = time.time(); last_pc[1] = pc["chars"]; on_text("act", dict(pc, ev="call_stream"))
                art = pc and pc.get("text") is not None and pc.get("target") and pc.get("action") in ("write", "append", "create") \\
                      and resolve_tool(plugins, pc["name"], {}, {t["function"]["name"] for t in tools})[0] == "artifact"
                if art and cut["at"] is None:
                    k = _chunk_cut(pc["text"])
                    if k is not None:                    # enough for one call: stop at the end of this line
                        cut["at"] = k; cut["pc"] = dict(pc); getattr(self.eng, "request_stop", lambda: None)()
                    elif A is not None and time.time() - cut["live"] > LIVE_EVERY:
                        cut["live"] = time.time(); cut["lkey"] = (pc["target"], pc.get("path"), pc["action"])
                        A.live_update(pc["target"], pc.get("path"), pc["action"], pc["text"], pc.get("title"), pc.get("type"))
                if should_stop and should_stop() and not cut["ui"]:
                    cut["ui"] = True; getattr(self.eng, "request_stop", lambda: None)()''')
rep('''            raw = self._decode(out); r = parse_reply(raw); text = r["answer"]''',
'''            raw = self._decode(out); r = parse_reply(raw); text = r["answer"]
            if cut["at"] is None and reason == "max" and tools:          # the token limit hit mid-file: save what is complete
                pc = partial_call(raw)
                if pc and pc.get("text") and pc.get("target") and pc.get("action") in ("write", "append", "create") and \\
                        resolve_tool(plugins, pc["name"], {}, {t["function"]["name"] for t in tools})[0] == "artifact":
                    cut["at"] = _chunk_cut(pc["text"], final=True); cut["pc"] = dict(pc)
            if cut["lkey"] and A is not None: A.live_end(*cut["lkey"])''')
rep('''            calls = [resolve_tool(plugins, n, a, enabled) for n, a in r["calls"]] if tools else []''',
'''            calls = [resolve_tool(plugins, n, a, enabled) for n, a in r["calls"]] if tools else []
            chunk = None
            if cut["at"] and chunks < MAX_CHUNKS:                         # build the call ourselves from the streamed part
                pc = cut["pc"]; part = pc["text"][:cut["at"]]; act = pc["action"]
                args = {"action": act, "name": pc["target"], "content": part}
                if pc.get("path"): args["path"] = pc["path"]
                if act == "create":
                    for k2 in ("type", "title"):
                        if pc.get(k2): args[k2] = pc[k2]
                calls = [("artifact", args)]; chunk = part; chunks += 1; budget += 1; reason = reason if reason != "stop" else "chunk"''')
rep('''            if reason == "max":
                calls = [(n, dict(a, _error=a["_error"]''', '''            if reason == "max" and not chunk:
                calls = [(n, dict(a, _error=a["_error"]''')
rep('''            if calls and r["harmony"]:          # gpt-oss: structured call; the chat template renders it as 'to=functions.NAME ... <|call|>'
                msg["content"] = ""; msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls[:1]]
                calls = calls[:1]                # the harmony template carries one call per assistant message''',
'''            if calls and r["harmony"]:          # gpt-oss: structured call; the chat template renders it as 'to=functions.NAME ... <|call|>'
                msg["content"] = ""; msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls[:1]]
                calls = calls[:1]                # the harmony template carries one call per assistant message
            elif chunk is not None:             # <tool_call> models: store the call we completed, not the cut-off text
                pre = raw.split("<tool_call>")[0]; pre = parse_reply(pre)["answer"]
                msg["content"] = (pre + "\\n" if pre else "") + "<tool_call>\\n" + json.dumps({"name": calls[0][0], "arguments": calls[0][1]}, ensure_ascii=False) + "\\n</tool_call>"''')
rep('''            if not calls or reason == "stopped" or (should_stop and should_stop()): break''',
    '''            if not calls or reason == "stopped" or cut["ui"] or (should_stop and should_stop()): break''')
rep('''                res = run_tool(plugins, name, args) if "_error" not in args else f"error: {args['_error']}"''',
'''                res = run_tool(plugins, name, args) if "_error" not in args else f"error: {args['_error']}"
                if chunk is not None and not res.startswith("error"):
                    tail = chunk.rstrip("\\n").rsplit("\\n", 1)[-1][-200:]
                    res += (f"\\nThat call was ended by the harness after {len(chunk)} characters (the end of a line), to keep each call small. "
                            f"Continue the same file now: call artifact with action 'append', name '{args['name']}'" + (f", path '{args['path']}'" if args.get("path") else "")
                            + f", with content that starts right after this last saved line:\\n{tail}\\nDo not repeat anything already saved. When the file is complete, stop calling tools and tell the user.")''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_core: chunked artifact writing, live save, history compaction, real Stop")
