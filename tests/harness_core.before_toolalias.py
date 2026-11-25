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


def run_tool(plugins, name, args):
    m = plugins.get(name)
    if not m or not getattr(m, "run", None): return f"error: no plugin named {name}"
    try: return str(m.run(**(args or {})))
    except Exception as ex: return f"error: {ex}"


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text):
    calls = []
    for blob in TOOL_RE.findall(text):
        try:
            c = json.loads(blob); args = c.get("arguments", {})
            if isinstance(args, str): args = json.loads(args or "{}")
            calls.append((c.get("name", ""), args))
        except Exception:
            calls.append(("", {"_error": f"could not parse tool call: {blob[:200]}"}))
    return calls


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
    try:                                   # context: keep the fp32 KV cache near 1 GB of RAM (it competes with weights and experts)
        per_tok = 2 * int(idx["layers"]) * int(idx["kv_heads"]) * int(idx["head_dim"]) * 4
        cap = max(2048, min(8192, int(1.0e9 / per_tok) // 1024 * 1024))
        if cap < s["ctx"]: notes.append(f"Context {cap} (not {s['ctx']}): the KV cache costs {per_tok / 1024:.0f} KB per token here, ~1 GB at {cap}.")
        s["ctx"] = min(s["ctx"], cap)
    except Exception: pass
    s.update(think=think, greedy=False, mode="auto", draft="none" if moe else "auto")
    notes.append(("Thinking: on (" + ("harmony analysis channel" if gptoss else "<think>") + ").") if think else "Thinking: off (this model's template has no reasoning mode).")
    notes.append("Draft: none (mixture of experts: verifying k tokens touches ~4x more experts)." if moe else "Draft: auto (the planner uses the 0.6B draft when it helps).")
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
        out[name] = {"name": name, "description": meta.get("description", ""), "always_on": meta.get("always_on", "false").lower() == "true",
                     "body": body, "file": f}
    return out


def save_skill(name, description, always_on, body, old_name=None):
    name = re.sub(r"[^A-Za-z0-9_\-]", "-", name.strip()) or "skill"
    if old_name and old_name != name and (hp.SKILLS / f"{old_name}.md").exists(): (hp.SKILLS / f"{old_name}.md").unlink()
    (hp.SKILLS / f"{name}.md").write_text(f"---\nname: {name}\ndescription: {description.strip()}\nalways_on: {'true' if always_on else 'false'}\n---\n{body.strip()}\n", encoding="utf-8")
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
        a = types.SimpleNamespace(mode=cfg["mode"], gpu_layers=int(cfg["gpu_layers"] or 0), ctx=int(cfg["ctx"]), draft=cfg["draft"] or "auto",
                                  temp=0.0 if cfg["greedy"] else float(cfg["temp"]), top_p=float(cfg["top_p"]), top_k=int(cfg["top_k"]), k=int(cfg["k"]))
        H = pl.probe(); M = pl.model_from_vault(self.vault)
        if M.get("moe"): self.banner = chat.moe_banner(M, max(0, int(H.get("vram_free_mb", 0)) - 900))
        else: H = dict(H, nvme_MBps=pl.drive_MBps(self.vault)); self.banner = pl.banner(M, H, pl.plan(M, H, a.ctx), a.ctx)
        args, self.how, _ = chat.pick_engine(self.vault, a); hp.LOGS.mkdir(exist_ok=True)
        self.eng = chat.SpecPair(args[0], args[1], self.vault, a.k, self.eos) if isinstance(args, tuple) else chat.Engine(args, str(hp.LOGS / "engine.log"))
        self.spec = isinstance(args, tuple)

    def close(self):
        try: self.eng.close()
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
        for rnd in range(max_tool_rounds + 1):
            msgs = ([{"role": "system", "content": system}] if system else []) + [m for m in history + new_msgs if m["role"] in ("user", "assistant", "tool")]
            if self.chat_mode:
                ids = self._ids(msgs, tools)
                while len(ids) + int(cfg["max_new"]) > int(cfg["ctx"]) and sum(1 for m in msgs if m["role"] != "system") > 1:
                    del msgs[1 if system else 0]; ids = self._ids(msgs, tools)          # drop the oldest turns to fit the context
            else:
                ids = self.tok("".join(m["content"] for m in msgs))["input_ids"]
                ids = ids[-(int(cfg["ctx"]) - int(cfg["max_new"]) - 1):]
            shown = {"thinking": "", "assistant": ""}
            def on_tok(out):                    # stream reasoning and answer separately; each only ever grows
                r = parse_reply(self._decode(out))
                for kind, cur in (("thinking", r["thinking"]), ("assistant", r["answer"])):
                    if cur.startswith(shown[kind]) and len(cur) > len(shown[kind]): on_text(kind, cur[len(shown[kind]):]); shown[kind] = cur
            seed = cfg["seed"] if cfg.get("seed") not in (None, "") else int(time.time() * 1e6) % (2 ** 62)
            temp, top_p, top_k = (0.0, 1.0, 1) if cfg["greedy"] else (float(cfg["temp"]), float(cfg["top_p"]), int(cfg["top_k"]))
            t0 = time.time()
            if self.spec: out, reason, st = self.eng.generate(ids, int(cfg["max_new"]), temp, top_p, top_k, int(seed), on_tok, int(cfg["ctx"]), should_stop)
            else: out, reason, st = self.eng.generate(ids, int(cfg["max_new"]), temp, top_p, top_k, int(seed), on_tok)
            raw = self._decode(out); r = parse_reply(raw); text = r["answer"]
            kv = dict(x.split("=", 1) for x in st.split() if "=" in x)
            stats = f"{len(out)} tokens in {time.time() - t0:.1f}s, {kv.get('tok_s', '?')} tok/s decode" + \
                    (f", {kv['tokens_per_pass']} tokens/pass" if "tokens_per_pass" in kv else "") + f", stop: {reason}"
            calls = r["calls"] if tools else []
            msg = {"role": "assistant", "content": text, "meta": {"stats": stats}}
            if r["thinking"]: msg["thinking"] = r["thinking"]
            if calls and r["harmony"]:          # gpt-oss: structured call; the chat template renders it as 'to=functions.NAME ... <|call|>'
                msg["content"] = ""; msg["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls[:1]]
                calls = calls[:1]                # the harmony template carries one call per assistant message
            new_msgs.append(msg)
            if not calls or reason == "stopped" or (should_stop and should_stop()): break
            for name, args in calls:
                res = run_tool(plugins, name, args) if "_error" not in args else f"error: {args['_error']}"
                on_text("tool", f"-> {name}({json.dumps(args)[:160]})\n")
                new_msgs.append({"role": "tool", "content": res, "meta": {"tool": name, "args": args}}); on_text("tool", f"[{name}] {res}\n")
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
