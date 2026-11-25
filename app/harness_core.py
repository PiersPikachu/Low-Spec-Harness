"""harness_core.py - everything the harness window does that is not drawing: model sessions, conversations, skills, plugins.

Used by harness_app.py (the window). No tkinter here, so it can be tested from the command line:
  python harness_core.py selftest      plugins, skills and conversation storage round trips (no model needed)
"""
import importlib.util, json, re, time, types, uuid
from pathlib import Path
import harness_paths as hp

# ------------------------------------------------------------------ plugins (tools the model may call)
def load_plugins():
    """{name: module} for every plugins/*.py that defines NAME, DESCRIPTION, PARAMETERS and run()"""
    out = {}
    for f in sorted(hp.PLUGINS.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(f"plugin_{f.stem}", f); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            if all(hasattr(m, a) for a in ("NAME", "DESCRIPTION", "PARAMETERS", "run")): out[m.NAME] = m
        except Exception as ex:
            out[f.stem] = types.SimpleNamespace(NAME=f.stem, DESCRIPTION=f"(failed to load: {ex})", PARAMETERS={}, run=None, broken=True)
    return out


def tool_schemas(plugins, enabled):
    return [{"type": "function", "function": {"name": n, "description": m.DESCRIPTION, "parameters": m.PARAMETERS}}
            for n, m in plugins.items() if n in enabled and getattr(m, "run", None)]


def _actions(m):
    try: return set(m.PARAMETERS["properties"]["action"]["enum"])
    except Exception: return set()


def resolve_tool(plugins, name, args, enabled=None):
    """(plugin name, args) for a call whose name may be decorated or may be an action of a plugin; unchanged if nothing fits"""
    args = dict(args or {}); pool = {n: m for n, m in plugins.items() if getattr(m, "run", None) and (enabled is None or n in enabled)}
    n = str(name or "").strip()
    for pre in ("functions.", "tools.", "tool."):
        if n.startswith(pre): n = n[len(pre):]
    if n in pool: return n, args
    for sep in (".", "_", "/", ":"):
        if sep in n:
            head, tail = n.split(sep, 1)
            if head in pool:
                if tail in _actions(pool[head]): args.setdefault("action", tail)
                return head, args
    owners = [p for p, m in pool.items() if n in _actions(m)]
    if len(owners) == 1: args.setdefault("action", n); return owners[0], args
    return n, args


def run_tool(plugins, name, args):
    m = plugins.get(name)
    if not m or not getattr(m, "run", None):
        have = ", ".join(n for n, x in plugins.items() if getattr(x, "run", None))
        return f"error: no tool named '{name}'. Available tools: {have}. Call a tool by its name and put the operation in its arguments (e.g. artifact with action='create')."
    try: return str(m.run(**(args or {})))
    except Exception as ex: return f"error: {ex}"


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text):
    calls = []
    for blob in TOOL_RE.findall(text):
        try:
            c = json.loads(blob, strict=False); args = c.get("arguments", {})
            if isinstance(args, str): args = json.loads(args or "{}", strict=False)
            calls.append((c.get("name", ""), args))
        except Exception:
            calls.append(("", {"_error": f"could not parse tool call JSON: {blob[:200]}"}))
    i = text.rfind("<tool_call>")
    if i >= 0 and "</tool_call>" not in text[i:]:          # opened but never closed: the reply ended mid-call
        calls.append(("", {"_error": "the tool call was " + CUT_MSG, "_cut": True}))
    return calls


# ------------------------------------------------------------------ reply formats
HARMONY_END = ("<|end|>", "<|return|>", "<|call|>")
CUT_MSG = "cut off before it finished. Your call was too long for one reply: write the file in parts - artifact action 'write' with the first part (under ~3000 characters), then action 'append' for each next part."


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
                b = body.strip()
                try: args = json.loads(b or "{}", strict=False)
                except Exception as ex:
                    cut = not b.endswith("}")
                    args = {"_error": ("the tool call was " + CUT_MSG) if cut else f"tool arguments are not valid JSON ({ex}). Escape quotes as \\\" and new lines as \\n inside strings.", "_cut": cut}
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
    return dict(name=name, action=f("action"), target=f("name") if name != "?" else None, path=f("path"), text=text, chars=len(body), type=f("type"), title=f("title"))


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


_OUTLINE = re.compile(r"^\s*(#{1,6}\s+\S.*|(?:async\s+)?def\s+\w+.*|class\s+\w+.*|(?:export\s+)?(?:async\s+)?function\s*\*?\s*\w+.*|"
                      r"(?:export\s+)?class\s+\w+.*|(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\(|function|class).*|"
                      r"<(?:h[1-6]|section|header|footer|main|nav|form|canvas|svg|script|style|template)\b[^>]*>.*)$")


def file_context(text, last=40, max_items=60):
    """outline (line: heading / def / class / function / main HTML element) + the last `last` lines of a file"""
    lines = text.splitlines(); out = []
    for i, ln in enumerate(lines, 1):
        if _OUTLINE.match(ln): out.append(f"  L{i}: {ln.strip()[:100]}")
    if len(out) > max_items: out = out[:max_items // 2] + [f"  ... {len(out) - max_items} more ..."] + out[-max_items // 2:]
    tail = "\n".join(f"{i:5d}| {l}" for i, l in enumerate(lines[-last:], max(1, len(lines) - last + 1)))
    return (f"File so far: {len(lines)} lines, {len(text)} characters.\nOutline:\n" + ("\n".join(out) or "  (no headings, functions or sections yet)")
            + f"\nLast {min(last, len(lines))} lines:\n{tail}")


def _cap_cut(text, room):
    """the longest prefix of text that fits in `room` characters and ends on a whole line (or exactly room if there is no newline)"""
    if room <= 0: return 0
    if len(text) <= room: return len(text)
    i = text.rfind("\n", 0, room); return i + 1 if i >= 0 else room


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


# ------------------------------------------------------------------ recommended settings per model
def recommended_settings(vault):
    """the model's own recommended generation settings -> (settings dict, list of note lines)
    order: model-card values for the family (gpt-oss, Hermes 4, Qwen3), then the vault's generation_config.json overrides the
    sampling values when it ships them (Qwen3 and Llama put their recommended values there)."""
    vault = Path(vault); notes = []
    def jload(p):
        try: return json.load(open(vault / p, encoding="utf-8"))
        except Exception: return {}
    cfgj = jload("vault.json").get("config", {}); gen = jload("generation_config.json")
    name = (str(cfgj.get("_name_or_path") or "") + " " + vault.name).lower(); mtype = str(cfgj.get("model_type", "")).lower()
    tmpl = ((vault / "chat_template.jinja").read_text(encoding="utf-8", errors="ignore") if (vault / "chat_template.jinja").exists() else "") + \
           str(jload("tokenizer_config.json").get("chat_template") or "")
    idx = {}
    try:
        for l in open(vault / "vault.idx"):
            t = l.split()
            if len(t) == 2: idx[t[0]] = t[1]
    except Exception: pass
    moe = idx.get("moe") == "1"; gptoss = mtype == "gpt_oss" or idx.get("gptoss") == "1"
    qwen = mtype.startswith("qwen3") or "qwen3" in name; hermes = "hermes" in name
    think = True if gptoss else ("enable_thinking" in tmpl or "<think>" in tmpl)
    if gptoss:
        s = dict(temp=1.0, top_p=1.0, top_k=0, max_new=2048, ctx=8192)
        notes += ["gpt-oss: OpenAI recommends temperature 1.0, top_p 1.0, top_k 0 (off; this engine caps top-k at 128).",
                  "It reasons before every answer, so replies need room: max new 2048.",
                  "OpenAI suggests >= 16k context; 8k keeps ~1 GB of RAM for the expert cache (raise it for long chats)."]
    elif hermes:
        s = dict(temp=0.6, top_p=0.95, top_k=20, max_new=2048, ctx=8192); notes.append("Hermes 4 (model card): temperature 0.6, top_p 0.95, top_k 20; reasons inside <think>.")
    elif qwen:
        s = dict(temp=0.6, top_p=0.95, top_k=20, ctx=8192) if think else dict(temp=0.7, top_p=0.8, top_k=20, ctx=8192)
        notes.append("Qwen3 (model card): thinking 0.6 / 0.95 / top-k 20, non-thinking 0.7 / 0.8 / 20; never greedy with thinking on (it can loop).")
    else:
        s = dict(temp=0.7, top_p=0.9, top_k=40, ctx=4096)
    shipped = {k: gen[g] for k, g in (("temp", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")) if g in gen}
    if shipped:
        s.update(shipped); notes.append("Sampling from the model's own generation_config.json: " + ", ".join(f"{k}={v}" for k, v in shipped.items()) + ".")
    elif not (gptoss or hermes or qwen):
        notes.append("No recommendation shipped with this model: generic 0.7 / 0.9 / 40.")
    s.setdefault("max_new", 2048 if think else int(gen.get("max_new_tokens", 1024)))
    try:                                   # context from the real KV cost: full-attention layers only, fp16 for gpt-oss
        lt = [int(x) for x in open(vault / "vault.idx").read().split("layer_types", 1)[1].split("\n", 1)[0].split()] if "layer_types" in open(vault / "vault.idx").read() else []
        full = int(idx["layers"]) - sum(lt); kv16 = gptoss; bpe = 2 if kv16 else 4
        per_tok = 2 * full * int(idx["kv_heads"]) * int(idx["head_dim"]) * bpe
        native = int(cfgj.get("max_position_embeddings") or 32768)
        try: ex_gb = sum(x["padded"] for x in jload("vault.json")["streams"][int(idx["expert_stream0"]):]) / 1e9 if moe else 0
        except Exception: ex_gb = 0
        fits = moe and ex_gb < 12.0                      # experts fit in RAM + VRAM: spare RAM can go to the KV cache
        budget = 3.5e9 if gptoss and (fits or not moe) else (1.0e9 if moe else 2.0e9)      # dense: weights stream, 2 GB of KV is affordable
        cap = max(2048, min(native, int(budget / per_tok) // 1024 * 1024))
        s["ctx"] = cap; s["kv16"] = kv16
        notes.append(f"Context {cap:,} tokens (model maximum {native:,}): KV cache {'fp16' if kv16 else 'fp32'}, {per_tok / 1024:.0f} KB per token"
                     + (", sliding-window layers excluded" if sum(lt) else "") + f", {per_tok * cap / 1e9:.1f} GB at full length.")
        if moe and not fits: notes.append(f"Experts ({ex_gb:.0f} GB) do not fit in memory, so the KV cache is kept near 1 GB: every GB given to it removes ~75 cached experts and slows decoding. Raise Context if you need it.")
        s["max_new"] = min(s.get("max_new", 2048), cap // 2)         # always leave room for the prompt
        if gptoss: s["max_new"] = 8192 if fits else 4096; notes.append(f"Max reply {s['max_new']:,} tokens (large files are written in 3,000-character tool calls, so this mostly bounds reasoning + answer).")
    except Exception as ex: notes.append(f"(context estimate unavailable: {ex})")
    s.update(think=think, greedy=False, mode="auto", draft="none" if moe else "auto")
    notes.append(("Thinking: on (" + ("harmony analysis channel" if gptoss else "<think>") + ").") if think else "Thinking: off (this model's template has no reasoning mode).")
    notes.append("Draft: none (mixture of experts: verifying k tokens touches ~4x more experts)." if moe else "Draft: auto (the planner uses the 0.6B draft when it helps).")
    tj = vault / "tuned.json"
    if tj.exists():                        # measured on this machine by the Models tab tuner: its engine settings win
        try:
            t = json.load(open(tj)); s.update(t["settings"])
            notes.append(f"Tuned on this machine ({t['date']}): {t['tok_s']} tok/s with {t['settings']} - re-run Tune in the Models tab after hardware changes.")
        except Exception: pass
    return s, [f"Recommended settings for {vault.name}:"] + notes

# ------------------------------------------------------------------ skills (instruction snippets)
def _parse_skill(text):
    meta, body = {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if m:
        body = m.group(2)
        for line in m.group(1).splitlines():
            if ":" in line: k, v = line.split(":", 1); meta[k.strip()] = v.strip()
    return meta, body.strip()


def load_skills():
    out = {}
    for f in sorted(hp.SKILLS.glob("*.md")):
        meta, body = _parse_skill(f.read_text(encoding="utf-8"))
        name = meta.get("name", f.stem)
        out[name] = {"name": name, "description": meta.get("description", ""), "always_on": meta.get("always_on", "false").lower() == "true", "author": meta.get("author", ""), "applies_when": meta.get("applies_when", ""),
                     "body": body, "file": f}
    return out


def tool_skill_text(tools, plugins, harmony):
    """bodies of skills marked 'applies_when: tools', with {tools} filled in and only the sections for this model's call format"""
    lines = []
    for t in tools:
        f = t["function"]; acts = ""
        try: acts = ", ".join(f["parameters"]["properties"]["action"]["enum"])
        except Exception: pass
        lines.append(f"- {f['name']}" + (f" (actions: {acts})" if acts else "") + f": {f['description'].split('. ')[0][:140]}")
    out = []
    for sk in load_skills().values():
        if sk.get("applies_when") != "tools": continue
        body = sk["body"].replace("{tools}", "\n".join(lines)); keep = []; on = True
        for ln in body.splitlines():
            m = re.match(r"^##\s*format:\s*(\w+)", ln)
            if m: on = (m.group(1) == "harmony") == bool(harmony); continue
            if ln.startswith("## "): on = True
            if on: keep.append(ln)
        out.append("\n".join(keep).strip())
    return "\n\n".join(out)


def save_skill(name, description, always_on, body, old_name=None):
    name = re.sub(r"[^A-Za-z0-9_\-]", "-", name.strip()) or "skill"
    if old_name and old_name != name and (hp.SKILLS / f"{old_name}.md").exists(): (hp.SKILLS / f"{old_name}.md").unlink()
    keep = ""
    for src in [hp.SKILLS / f"{name}.md"] + ([hp.SKILLS / f"{old_name}.md"] if old_name else []):
        if src.exists():
            meta, _ = _parse_skill(src.read_text(encoding="utf-8")); keep = "".join(f"{k}: {meta[k]}\n" for k in ("author", "applies_when") if meta.get(k)); break
    (hp.SKILLS / f"{name}.md").write_text(f"---\nname: {name}\ndescription: {description.strip()}\nalways_on: {'true' if always_on else 'false'}\n{keep}---\n{body.strip()}\n", encoding="utf-8")
    return name


def delete_skill(name):
    f = hp.SKILLS / f"{name}.md"
    if f.exists(): f.unlink()


def expand_skills(text, skills):
    """'/name rest of message' -> skill body + rest; several /names may lead the message"""
    used = []
    while True:
        m = re.match(r"^\s*/([A-Za-z0-9_\-]+)\b\s*", text)
        if not m or m.group(1) not in skills: break
        used.append(m.group(1)); text = text[m.end():]
    if not used: return text, []
    return "\n\n".join(skills[u]["body"] for u in used) + ("\n\n" + text if text.strip() else ""), used


# ------------------------------------------------------------------ conversations
def list_conversations():
    convs = []
    for f in hp.CONVERSATIONS.glob("*.json"):
        try: c = json.load(open(f, encoding="utf-8")); convs.append(c)
        except Exception: pass
    return sorted(convs, key=lambda c: c.get("updated", 0), reverse=True)


def new_conversation(vault=""):
    t = time.time(); return {"id": uuid.uuid4().hex[:12], "title": "New chat", "created": t, "updated": t, "vault": vault, "messages": []}


def save_conversation(c):
    c["updated"] = time.time(); json.dump(c, open(hp.CONVERSATIONS / f"{c['id']}.json", "w", encoding="utf-8"), indent=1)


def delete_conversation(cid):
    f = hp.CONVERSATIONS / f"{cid}.json"
    if f.exists(): f.unlink()


# ------------------------------------------------------------------ model session
class Session:
    """one loaded model (plain engine or speculative pair) + its tokenizer"""
    def __init__(self, cfg, log=print):
        import chat, planner as pl
        from transformers import AutoTokenizer
        self.cfg = dict(cfg); self.vault = hp.resolve_vault(cfg["vault"], cfg); self.log = log
        self.tok = AutoTokenizer.from_pretrained(str(self.vault)); self.chat_mode = bool(self.tok.chat_template)
        info = chat.idx_info(self.vault); self.eos = {int(v) for k, v in info.items() if k in ("eos", "eos2")}
        a = types.SimpleNamespace(mode=cfg["mode"], gpu_layers=int(cfg["gpu_layers"] or 0), ctx=int(cfg["ctx"]), draft=cfg["draft"] or "auto", kv16=bool(cfg.get("kv16")), moe_prefetch=cfg.get("moe_prefetch"), moe_vram=cfg.get("moe_vram"),
                                  temp=0.0 if cfg["greedy"] else float(cfg["temp"]), top_p=float(cfg["top_p"]), top_k=int(cfg["top_k"]), k=int(cfg["k"]))
        H = pl.probe(); M = pl.model_from_vault(self.vault)
        if M.get("moe"):
            try: _t = json.load(open(Path(self.vault) / "tuned.json"))["settings"]
            except Exception: _t = {}
            _pf = cfg.get("moe_prefetch"); _pf = _t.get("moe_prefetch", 2 if M.get("gptoss") else 0) if _pf is None else _pf
            _vr = cfg.get("moe_vram"); _vr = _t.get("moe_vram", True) if _vr is None else _vr
            self.banner = chat.moe_banner(M, max(0, int(H.get("vram_free_mb", 0)) - 900) if _vr else 0, _pf if M.get("gptoss") else 0)
        else: H = dict(H, nvme_MBps=pl.drive_MBps(self.vault)); self.banner = pl.banner(M, H, pl.plan(M, H, a.ctx), a.ctx)
        args, self.how, _ = chat.pick_engine(self.vault, a); hp.LOGS.mkdir(exist_ok=True)
        self.eng = chat.SpecPair(args[0], args[1], self.vault, a.k, self.eos) if isinstance(args, tuple) else chat.Engine(args, str(hp.LOGS / "engine.log"))
        self.spec = isinstance(args, tuple)

    def close(self):
        try: self.eng.close()
        except Exception: pass

    def _archive(self, history, dropped, A):
        """append trimmed turns verbatim to the artifact memory-<id> and return the note that replaces them"""
        if A is None: return None
        import hashlib
        first = next((m.get("content", "") for m in history if m.get("role") == "user"), "") or "chat"
        name = "memory-" + hashlib.sha1(first.encode("utf-8", "ignore")).hexdigest()[:8]
        lines = []
        for m in dropped:
            body = m.get("content") or ""
            if m.get("tool_calls"): body = (body + "\n" if body else "") + json.dumps([tc.get("function", tc) for tc in m["tool_calls"]], ensure_ascii=False)
            lines.append(f"### {m['role']}\n{body}\n")
        if not (A.ROOT / name / ".artifact.json").exists(): A.run("create", name=name, type="note", title="Earlier conversation (moved out of the context window)")
        A.run("append", name=name, content="\n".join(lines))
        return (f"[Harness note: {len(dropped)} earlier message(s) of this conversation were moved out of the context window into the artifact "
                f"'{name}' (notes.md), word for word. If you need them, read it with the artifact tool (action 'read', name '{name}').]")

    @property
    def harmony(self):
        """True for models that call tools in the harmony format (gpt-oss)"""
        t = getattr(self.tok, "chat_template", None) or ""
        return "<|channel|>" in (t if isinstance(t, str) else str(t))

    def kill(self):
        """Force stop: kill the engine process(es) now; a generation in progress fails at once and this session must be reloaded"""
        for p in [getattr(self.eng, "p", None)] + [getattr(getattr(self.eng, a, None), "p", None) for a in ("T", "D")]:
            try:
                if p is not None and p.poll() is None: p.kill()
            except Exception: pass

    def _ids(self, msgs, tools):
        kw = {"enable_thinking": self.cfg["think"], "thinking": self.cfg["think"]}
        if tools: kw["tools"] = tools
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def _decode(self, out):
        return self.tok.decode([t for t in out if t not in self.eos], skip_special_tokens=False).replace("<|endoftext|>", "")

    def reply(self, history, system, tools, plugins, on_text, should_stop, max_tool_rounds=4):
        """history: [{role, content}] including the new user message. Streams text through on_text(kind, text).
        Runs tool calls the model makes (enabled plugins only) and continues. Returns the new messages to append and a stats line."""
        cfg = self.cfg; new_msgs = []; stats = ""
        if tools:
            auto = tool_skill_text(tools, plugins, self.harmony)
            if auto: system = (system + "\n\n" + auto).strip() if system else auto
        rnd = -1; budget = max_tool_rounds + 1; chunks = 0; A = plugins.get("artifact"); cont = []; conts = 0; carry = None
        cap_chars = int(cfg.get("max_file_growth") or 0) or 4 * int(cfg["max_new"]); written = [0]; capped = [False]
        if A is not None and hasattr(A, "live_keep"): A.live_keep()       # forget restore points of an interrupted reply (its live saves stay)
        while rnd + 1 < budget:
            rnd += 1
            msgs = ([{"role": "system", "content": system}] if system else []) + [m for m in history + new_msgs if m["role"] in ("user", "assistant", "tool")]
            msgs = _compact_history(msgs)
            if self.chat_mode:
                ids = self._ids(msgs, tools)
                last_user = max((i for i, m in enumerate(msgs) if m["role"] == "user"), default=0)
                dropped = []
                while len(ids) + len(cont) + int(cfg["max_new"]) > int(cfg["ctx"]):
                    old = next((i for i in range(1 if system else 0, last_user)), None)       # never drop the current request
                    if old is None: break
                    dropped.append(msgs.pop(old)); last_user -= 1; ids = self._ids(msgs, tools)
                if dropped:                       # nothing is lost: the dropped turns go to a memory artifact the model can read
                    note = self._archive(history, dropped, A)
                    if note:
                        msgs.insert(1 if system else 0, {"role": "user", "content": note}); ids = self._ids(msgs, tools)
            else:
                ids = self.tok("".join(m["content"] for m in msgs))["input_ids"]
                ids = ids[-(int(cfg["ctx"]) - int(cfg["max_new"]) - 1):]
            pre = list(cont); ids = ids + pre if self.chat_mode else ids
            shown = carry if carry else {"thinking": "", "assistant": ""}; carry = None; last_pc = [0.0, -1]; cut = {"at": None, "pc": None, "ui": False, "live": 0.0, "lkey": None}
            on_text("act", {"ev": "round", "n": rnd + 1})
            def on_tok(out):                    # stream reasoning and answer separately; each only ever grows
                raw_now = self._decode(pre + out); r = parse_reply(raw_now)
                pc = partial_call(raw_now) if tools else None
                if pc and (time.time() - last_pc[0] > 0.12 or pc["chars"] - last_pc[1] > 400):
                    last_pc[0] = time.time(); last_pc[1] = pc["chars"]; on_text("act", dict(pc, ev="call_stream"))
                art = pc and pc.get("text") is not None and pc.get("target") and pc.get("action") in ("write", "append", "create") \
                      and resolve_tool(plugins, pc["name"], {}, {t["function"]["name"] for t in tools})[0] == "artifact"
                if art and cut["at"] is None:
                    room = cap_chars - written[0]
                    if len(pc["text"]) >= room: k = _cap_cut(pc["text"], room); cut["capped"] = True
                    else: k = _chunk_cut(pc["text"])
                    if k is not None:                    # enough for one call: stop at the end of this line
                        cut["at"] = k; cut["pc"] = dict(pc); getattr(self.eng, "request_stop", lambda: None)()
                    elif A is not None and time.time() - cut["live"] > LIVE_EVERY:
                        cut["live"] = time.time(); cut["lkey"] = (pc["target"], pc.get("path"), pc["action"])
                        A.live_update(pc["target"], pc.get("path"), pc["action"], pc["text"], pc.get("title"), pc.get("type"))
                if should_stop and should_stop() and not cut["ui"]:
                    cut["ui"] = True; getattr(self.eng, "request_stop", lambda: None)()
                for kind, cur in (("thinking", r["thinking"]), ("assistant", r["answer"])):
                    if cur.startswith(shown[kind]) and len(cur) > len(shown[kind]): on_text(kind, cur[len(shown[kind]):]); shown[kind] = cur
            seed = cfg["seed"] if cfg.get("seed") not in (None, "") else int(time.time() * 1e6) % (2 ** 62)
            temp, top_p, top_k = (0.0, 1.0, 1) if cfg["greedy"] else (float(cfg["temp"]), float(cfg["top_p"]), int(cfg["top_k"]))
            t0 = time.time()
            if self.spec: out, reason, st = self.eng.generate(ids, int(cfg["max_new"]), temp, top_p, top_k, int(seed), on_tok, int(cfg["ctx"]), should_stop)
            else: out, reason, st = self.eng.generate(ids, int(cfg["max_new"]), temp, top_p, top_k, int(seed), on_tok)
            out = pre + out; cont = []
            if reason == "ctx" and self.chat_mode and conts < 4 and cut["at"] is None and not cut["ui"]:
                cont = out; conts += 1; budget += 1; carry = shown; on_text("act", {"ev": "round", "n": rnd + 2})       # compact the history and carry on
                continue
            raw = self._decode(out); r = parse_reply(raw); text = r["answer"]
            if cut["at"] is None and reason == "max" and tools:          # the token limit hit mid-file: save what is complete
                pc = partial_call(raw)
                if pc and pc.get("text") and pc.get("target") and pc.get("action") in ("write", "append", "create") and \
                        resolve_tool(plugins, pc["name"], {}, {t["function"]["name"] for t in tools})[0] == "artifact":
                    cut["at"] = min(_chunk_cut(pc["text"], final=True), _cap_cut(pc["text"], cap_chars - written[0])); cut["pc"] = dict(pc)
                    if cut["at"] < len(pc["text"].rstrip("\n")) and cut["at"] == _cap_cut(pc["text"], cap_chars - written[0]): cut["capped"] = True
            if cut["lkey"] and A is not None: A.live_end(*cut["lkey"])
            kv = dict(x.split("=", 1) for x in st.split() if "=" in x)
            stats = f"{len(out)} tokens in {time.time() - t0:.1f}s, {kv.get('tok_s', '?')} tok/s decode" + \
                    (f", {kv['tokens_per_pass']} tokens/pass" if "tokens_per_pass" in kv else "") + f", stop: {reason}"
            enabled = {t["function"]["name"] for t in tools} if tools else set()
            calls = [resolve_tool(plugins, n, a, enabled) for n, a in r["calls"]] if tools else []
            chunk = None
            if cut["at"] is not None and cut["at"] == 0 and cut.get("capped"): cut["at"] = None; capped[0] = True       # nothing fits any more
            if cut["at"] and chunks < MAX_CHUNKS:                         # build the call ourselves from the streamed part
                pc = cut["pc"]; part = pc["text"][:cut["at"]]; act = pc["action"]
                args = {"action": act, "name": pc["target"], "content": part}
                if pc.get("path"): args["path"] = pc["path"]
                if act == "create":
                    for k2 in ("type", "title"):
                        if pc.get(k2): args[k2] = pc[k2]
                calls = [("artifact", args)]; chunk = part; chunks += 1; budget += 1; reason = reason if reason != "stop" else "chunk"
            if reason == "max" and not chunk:
                calls = [(n, dict(a, _error=a["_error"] + f" (this reply stopped at the max-new-tokens limit of {cfg['max_new']})")) if "_error" in a else (n, a) for n, a in calls]
            msg = {"role": "assistant", "content": text, "meta": {"stats": stats}}
            if r["thinking"]: msg["thinking"] = r["thinking"]
            if calls and r["harmony"]:          # gpt-oss: structured call; the chat template renders it as 'to=functions.NAME ... <|call|>'
                msg["content"] = ""; msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls[:1]]
                calls = calls[:1]                # the harmony template carries one call per assistant message
            elif chunk is not None:             # <tool_call> models: store the call we completed, not the cut-off text
                pre = raw.split("<tool_call>")[0]; pre = parse_reply(pre)["answer"]
                msg["content"] = (pre + "\n" if pre else "") + "<tool_call>\n" + json.dumps({"name": calls[0][0], "arguments": calls[0][1]}, ensure_ascii=False) + "\n</tool_call>"
            new_msgs.append(msg)
            if not calls or reason == "stopped" or cut["ui"] or (should_stop and should_stop()): break
            for i_c, (name, args) in enumerate(calls):
                if name == "artifact" and "_error" not in args:
                    key = "new" if args.get("action") == "edit" else "content"
                    if args.get("action") in ("write", "append", "create", "edit") and isinstance(args.get(key), str):
                        room = cap_chars - written[0]
                        if capped[0] or room <= 0:
                            args = dict(args, _error=f"the per-reply file growth limit ({cap_chars:,} characters) is already reached. Do not write more in this reply: "
                                                     f"tell the user the file is incomplete; they can say 'continue' in their next message.")
                        elif len(args[key]) > room:
                            k2 = _cap_cut(args[key], room); args = dict(args, **{key: args[key][:k2]}); capped[0] = True
                            if chunk is not None: chunk = args[key]
                        if "_error" not in args: written[0] += len(args[key])
                        calls[i_c] = (name, args)
                        if cut.get("capped"): capped[0] = True
                on_text("act", {"ev": "call", "name": name, "args": args})
                res = run_tool(plugins, name, args) if "_error" not in args else f"error: {args['_error']}"
                fctx = ""
                if name == "artifact" and A is not None and not res.startswith("error") and args.get("action") in ("write", "append", "create", "edit"):
                    try:
                        d = A.ROOT / A._slug(args["name"]); fp, _rel = A._file(d, args.get("path"), A._meta(d))
                        fctx = file_context(fp.read_text(encoding="utf-8", errors="replace"))
                    except Exception: fctx = ""
                if capped[0] and not res.startswith("error"):
                    res += (f"\nThe per-reply file growth limit ({cap_chars:,} characters) is reached, so the harness saved the content up to the last whole line "
                            f"that fits. Stop writing now. Tell the user the file is incomplete, what remains to be written, and that they can say 'continue' "
                            f"in their next message to resume from the last saved line.\n{fctx}")
                elif chunk is not None and not res.startswith("error"):
                    res += (f"\nThat call was ended by the harness after {len(chunk)} characters (the end of a line), to keep each call small. "
                            f"Continue the same file now: call artifact with action 'append', name '{args['name']}'" + (f", path '{args['path']}'" if args.get("path") else "")
                            + f", with content that starts right after the last saved line shown below. Keep it consistent with the outline (same names, "
                            f"styles and structure). Do not repeat anything already saved. When the file is complete, stop calling tools and tell the user.\n{fctx}")
                on_text("tool", f"-> {name}({json.dumps(args)[:160]})\n")
                new_msgs.append({"role": "tool", "content": res, "meta": {"tool": name, "args": args}}); on_text("tool", f"[{name}] {res}\n")
                on_text("act", {"ev": "result", "name": name, "args": args, "text": res})
            on_text("assistant_next", "")
        return new_msgs, stats


def selftest():
    hp.ensure_dirs(); ok = True
    P = load_plugins(); r = run_tool(P, "calculator", {"expression": "17*23"}); ok &= r == "391"; print("calculator 17*23 ->", r)
    print("clock ->", run_tool(P, "clock", {})); print("tool schemas:", [t["function"]["name"] for t in tool_schemas(P, set(P))])
    calls = parse_tool_calls('x <tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'); ok &= calls == [("calculator", {"expression": "2+2"})]
    print("parse tool call ->", calls)
    save_skill("_selftest", "test", False, "BODY TEXT"); S = load_skills(); e, used = expand_skills("/_selftest hello", S)
    ok &= e == "BODY TEXT\n\nhello" and used == ["_selftest"]; delete_skill("_selftest"); print("skill expand ->", repr(e))
    c = new_conversation("x"); c["messages"].append({"role": "user", "content": "hi"}); save_conversation(c)
    ok &= any(x["id"] == c["id"] for x in list_conversations()); delete_conversation(c["id"]); print("conversation round trip ok")
    print("SELFTEST", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["selftest"]: selftest()
