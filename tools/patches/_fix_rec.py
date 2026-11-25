from pathlib import Path
import ast
P = Path(r"C:\Users\1fred\Desktop\AI\Hermes\cube-harness\harness_core.py"); s = P.read_text(encoding="utf-8")
a = s.index("def recommended_settings(vault):"); b = s.index("\n\n# ------------------------------------------------------------------ skills")
NEW = '''def recommended_settings(vault):
    """the model's own recommended generation settings -> (settings dict, list of note lines)
    order: model-card values for the family (gpt-oss, Hermes 4, Qwen3), then the vault's generation_config.json overrides the
    sampling values when it ships them (Qwen3 and Llama put their recommended values there)."""
    vault = Path(vault); notes = []
    def jload(p):
        try: return json.load(open(vault / p, encoding="utf-8"))
        except Exception: return {}
    cfgj = jload("vault.json").get("config", {}); gen = jload("generation_config.json")
    name = (str(cfgj.get("_name_or_path") or "") + " " + vault.name).lower(); mtype = str(cfgj.get("model_type", "")).lower()
    tmpl = ((vault / "chat_template.jinja").read_text(encoding="utf-8", errors="ignore") if (vault / "chat_template.jinja").exists() else "") + \\
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
    s.update(think=think, greedy=False, mode="auto", draft="none" if moe else "auto")
    notes.append(("Thinking: on (" + ("harmony analysis channel" if gptoss else "<think>") + ").") if think else "Thinking: off (this model's template has no reasoning mode).")
    notes.append("Draft: none (mixture of experts: verifying k tokens touches ~4x more experts)." if moe else "Draft: auto (the planner uses the 0.6B draft when it helps).")
    return s, [f"Recommended settings for {vault.name}:"] + notes'''
s = s[:a] + NEW + s[b:]; ast.parse(s); P.write_text(s, encoding="utf-8"); print("recommended_settings rewritten")
