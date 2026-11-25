"""make_core_reasoning.py - harness_core.py: one reply parser for both reasoning formats, gpt-oss (harmony) tool calls,
separate thinking/answer streams for the window, and each model's recommended settings. Edits harness_core.py in place."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_core.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

PARSE = r'''

# ------------------------------------------------------------------ reply formats
HARMONY_END = ("<|end|>", "<|return|>", "<|call|>")


def parse_reply(raw):
    """raw decoded reply (special tokens kept) -> dict(thinking, answer, calls, harmony)
    harmony (gpt-oss): segments '<|channel|>analysis|commentary|final ... <|message|>body<|end|>', joined by '<|start|>assistant';
      a segment whose header names a recipient 'to=functions.NAME' is a tool call whose body is the JSON arguments.
    <think> (Qwen3, Hermes 4): '<think>reasoning</think>answer' (the opening tag may sit in the prompt); tool calls are
      '<tool_call>{json}</tool_call>' inside the answer. Works on partial text while streaming: unfinished headers are ignored."""
    if "<|channel|>" in raw or "<|message|>" in raw:
        think, ans, calls = [], [], []
        for seg in raw.split("<|start|>"):
            if "<|message|>" not in seg: continue
            head, body = seg.split("<|message|>", 1)
            for e in HARMONY_END:
                if e in body: body = body.split(e, 1)[0]
            ch = re.search(r"<\|channel\|>(\w+)", head); to = re.search(r"to=functions\.([\w.\-]+)", head)
            if to:
                try: args = json.loads(body.strip() or "{}")
                except Exception: args = {"_error": f"could not parse tool arguments: {body[:200]}"}
                calls.append((to.group(1), args if isinstance(args, dict) else {"value": args}))
            elif ch and ch.group(1) == "analysis": think.append(body)
            else: ans.append(body)
        return dict(thinking="\n".join(think).strip(), answer="".join(ans).strip(), calls=calls, harmony=True)
    t = raw
    if "</think>" in t:
        th, ans = t.split("</think>", 1); th = th.split("<think>", 1)[-1]
    elif "<think>" in t:
        th, ans = t.split("<think>", 1)[1], ""
    else:
        th, ans = "", t
    return dict(thinking=th.strip(), answer=ans.strip(), calls=parse_tool_calls(ans), harmony=False)


# ------------------------------------------------------------------ recommended settings per model
def recommended_settings(vault):
    """the model's own recommended generation settings -> (settings dict, list of note lines)
    sources: the vault's generation_config.json (Qwen3, Llama ship their values there), else the model card values below."""
    vault = Path(vault); notes = []; s = {}
    try: cfgj = json.load(open(vault / "vault.json")).get("config", {})
    except Exception: cfgj = {}
    try: gen = json.load(open(vault / "generation_config.json"))
    except Exception: gen = {}
    name = (str(cfgj.get("_name_or_path", "")) + " " + vault.name).lower(); mtype = str(cfgj.get("model_type", "")).lower()
    tmpl = (vault / "chat_template.jinja").read_text(encoding="utf-8", errors="ignore") if (vault / "chat_template.jinja").exists() else ""
    idx = {}
    try:
        for l in open(vault / "vault.idx"):
            t = l.split()
            if len(t) == 2: idx[t[0]] = t[1]
    except Exception: pass
    moe = idx.get("moe") == "1"
    if mtype == "gpt_oss" or idx.get("gptoss") == "1":
        s = dict(temp=1.0, top_p=1.0, top_k=0, think=True, max_new=2048, ctx=8192)
        notes += ["gpt-oss: OpenAI recommends temperature 1.0, top_p 1.0, top_k 0 (off; this engine caps top-k at 128).",
                  "Reasoning runs before every answer, so replies need room: max new 2048.",
                  "OpenAI suggests >= 16k context; 8k here keeps ~1 GB of RAM for the expert cache (raise it if you need long chats)."]
    elif "hermes-4" in name or "hermes4" in name:
        s = dict(temp=0.6, top_p=0.95, top_k=20, think=True, max_new=2048, ctx=8192)
        notes += ["Hermes 4 (model card): temperature 0.6, top_p 0.95, top_k 20; hybrid reasoning with <think>."]
    elif "qwen3" in name and "instruct-2507" in name:
        s = dict(temp=0.7, top_p=0.8, top_k=20, think=False, max_new=1024, ctx=8192)
        notes += ["Qwen3 Instruct-2507 (model card): temperature 0.7, top_p 0.8, top_k 20, min_p 0; non-thinking model."]
    elif "qwen3" in name and "thinking-2507" in name:
        s = dict(temp=0.6, top_p=0.95, top_k=20, think=True, max_new=4096, ctx=8192)
        notes += ["Qwen3 Thinking-2507 (model card): temperature 0.6, top_p 0.95, top_k 20, min_p 0; always reasons."]
    elif mtype.startswith("qwen3") or "qwen3" in name:
        s = dict(temp=0.6, top_p=0.95, top_k=20, think=True, max_new=2048, ctx=8192)
        notes += ["Qwen3 (model card): thinking mode temperature 0.6, top_p 0.95, top_k 20; non-thinking 0.7 / 0.8 / 20.",
                  "Do not use greedy decoding with thinking on (Qwen warns it can loop - seen here with 0.6B on arithmetic)."]
    for k, gk in (("temp", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if gk in gen:
            if k in s and s[k] != gen[gk]: notes.append(f"generation_config.json says {gk}={gen[gk]} (model card value {s[k]} kept).")
            else: s.setdefault(k, gen[gk])
    if "temp" not in s: s.update(temp=0.7, top_p=0.9, top_k=40); notes.append("No recommendation found for this model: generic 0.7 / 0.9 / 40.")
    if "max_new" not in s: s["max_new"] = int(gen.get("max_new_tokens", 1024))
    s.setdefault("ctx", 4096); s.setdefault("think", "<think>" in tmpl)
    s.update(greedy=False, mode="auto", draft="none" if moe else "auto")
    notes.append("Draft: none (mixture of experts: speculative verification touches ~4x more experts)." if moe else "Draft: auto (the planner picks the 0.6B draft when it helps).")
    src = "generation_config.json" if any(k in gen for k in ("temperature", "top_p", "top_k")) else "model card"
    return s, [f"Recommended settings for {vault.name} (source: {src}):"] + notes
'''
rep("\n\n# ------------------------------------------------------------------ skills (instruction snippets)", PARSE + "\n\n# ------------------------------------------------------------------ skills (instruction snippets)")

# decode: raw text only (parsing does the rest)
a = s.index("    def _decode(self, out):"); b = s.index("    def reply(self,")
s = s[:a] + '''    def _decode(self, out):
        return self.tok.decode([t for t in out if t not in self.eos], skip_special_tokens=False).replace("<|endoftext|>", "")

''' + s[b:]
rep('''            shown = [""]
            def on_tok(out):
                s = self._decode(out)
                if s.startswith(shown[0]) and len(s) > len(shown[0]): on_text("assistant", s[len(shown[0]):]); shown[0] = s''',
'''            shown = {"thinking": "", "assistant": ""}
            def on_tok(out):                    # stream reasoning and answer separately; each only ever grows
                r = parse_reply(self._decode(out))
                for kind, cur in (("thinking", r["thinking"]), ("assistant", r["answer"])):
                    if cur.startswith(shown[kind]) and len(cur) > len(shown[kind]): on_text(kind, cur[len(shown[kind]):]); shown[kind] = cur''')
rep('''            text = self._decode(out).strip()''', '''            raw = self._decode(out); r = parse_reply(raw); text = r["answer"]''')
rep('''            final, thinking = self._split_thinking(text)
            new_msgs.append(dict({"role": "assistant", "content": final, "meta": {"stats": stats}}, **({"thinking": thinking} if thinking else {})))
            calls = parse_tool_calls(text) if tools else []''',
'''            calls = r["calls"] if tools else []
            msg = {"role": "assistant", "content": text, "meta": {"stats": stats}}
            if r["thinking"]: msg["thinking"] = r["thinking"]
            if calls and r["harmony"]:          # gpt-oss: structured call; the chat template renders it as 'to=functions.NAME ... <|call|>'
                msg["content"] = ""; msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls[:1]]
                calls = calls[:1]                # the harmony template carries one call per assistant message
            new_msgs.append(msg)''')
rep('''                new_msgs.append({"role": "tool", "content": res, "meta": {"tool": name, "args": args}}); on_text("tool", f"[{name}] {res}\\n")''',
'''                on_text("tool", f"-> {name}({json.dumps(args)[:160]})\\n")
                new_msgs.append({"role": "tool", "content": res, "meta": {"tool": name, "args": args}}); on_text("tool", f"[{name}] {res}\\n")''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_core: reply parser, harmony tool calls, thinking stream, recommended settings")
