"""make_harness_kv.py - harness side of the long-context work.

  chat.py           --kv16 passed to the engine when cfg kv16; SETPOS refused by a sliding-window ring ('ERR ring') -> full re-read
  harness_core.py   recommended ctx / max_new from the real KV cost (sliding layers, fp16) and whether experts fit in memory;
                    trimmed turns are archived verbatim into a 'memory-...' artifact with a note the model can follow (nothing is lost);
                    a reply that reaches the end of the context window is continued: the history is compacted and generation resumes
                    from the same tokens (up to 4 times)
  harness_app.py    'KV cache in fp16' setting; wider ranges for context / max reply / top-k
  harness_paths.py  kv16 default false"""
import ast
from pathlib import Path
H = Path(__file__).resolve().parent
def patch(fname, pairs):
    p = H / fname; s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (fname, old[:70], s.count(old)); s = s.replace(old, new)
    ast.parse(s); p.write_text(s, encoding="utf-8"); print("patched", fname)

patch("chat.py", [
    ('''            self.p.stdin.write(f"SETPOS {lcp}\\n"); self.p.stdin.flush(); self.p.stdout.readline()''',
     '''            self.p.stdin.write(f"SETPOS {lcp}\\n"); self.p.stdin.flush()
            if self.p.stdout.readline().startswith("ERR"): self.reset(); lcp = 0      # e.g. a sliding-window ring cannot roll back that far'''),
    ('''        new = ids[lcp:]; self._stop_sent = False''', '''        new = ids[lcp:]; self._stop_sent = False
        if not new: new = ids[-1:]''') if False else ('''        new = ids[lcp:]; self._stop_sent = False''', '''        new = ids[lcp:]; self._stop_sent = False'''),
    ('''        args = [str(pl.ENGINE), "serve", str(vault), "--ctx", str(a.ctx), "--batch", "128", "--resident-mb", "9000"] + (["--vram-mb", str(vram)] if vram > 500 else []) \\''',
     '''        args = [str(pl.ENGINE), "serve", str(vault), "--ctx", str(a.ctx), "--batch", "128", "--resident-mb", "9000"] + (["--vram-mb", str(vram)] if vram > 500 else []) \\
            + (["--kv16"] if getattr(a, "kv16", False) else []) \\'''),
])

patch("harness_core.py", [
    ('''        a = types.SimpleNamespace(mode=cfg["mode"], gpu_layers=int(cfg["gpu_layers"] or 0), ctx=int(cfg["ctx"]), draft=cfg["draft"] or "auto",''',
     '''        a = types.SimpleNamespace(mode=cfg["mode"], gpu_layers=int(cfg["gpu_layers"] or 0), ctx=int(cfg["ctx"]), draft=cfg["draft"] or "auto", kv16=bool(cfg.get("kv16")),'''),
    # recommended context from the real KV cost
    ('''    try:                                   # context: keep the fp32 KV cache near 1 GB of RAM (it competes with weights and experts)
        per_tok = 2 * int(idx["layers"]) * int(idx["kv_heads"]) * int(idx["head_dim"]) * 4
        cap = max(2048, min(8192, int(1.0e9 / per_tok) // 1024 * 1024))
        if cap < s["ctx"]: notes.append(f"Context {cap} (not {s['ctx']}): the KV cache costs {per_tok / 1024:.0f} KB per token here, ~1 GB at {cap}.")
        s["ctx"] = min(s["ctx"], cap)
    except Exception: pass''',
     '''    try:                                   # context from the real KV cost: full-attention layers only, fp16 for gpt-oss
        lt = [int(x) for x in open(vault / "vault.idx").read().split("layer_types", 1)[1].split("\\n", 1)[0].split()] if "layer_types" in open(vault / "vault.idx").read() else []
        full = int(idx["layers"]) - sum(lt); kv16 = gptoss; bpe = 2 if kv16 else 4
        per_tok = 2 * full * int(idx["kv_heads"]) * int(idx["head_dim"]) * bpe
        native = int(cfgj.get("max_position_embeddings") or 32768)
        try: ex_gb = sum(x["padded"] for x in jload("vault.json")["streams"][int(idx["expert_stream0"]):]) / 1e9 if moe else 0
        except Exception: ex_gb = 0
        fits = moe and ex_gb < 12.0                      # experts fit in RAM + VRAM: spare RAM can go to the KV cache
        budget = 3.5e9 if (fits or not moe) and gptoss else 1.0e9
        cap = max(2048, min(native, int(budget / per_tok) // 1024 * 1024))
        s["ctx"] = cap; s["kv16"] = kv16
        notes.append(f"Context {cap:,} tokens (model maximum {native:,}): KV cache {'fp16' if kv16 else 'fp32'}, {per_tok / 1024:.0f} KB per token"
                     + (", sliding-window layers excluded" if sum(lt) else "") + f", {per_tok * cap / 1e9:.1f} GB at full length.")
        if moe and not fits: notes.append(f"Experts ({ex_gb:.0f} GB) do not fit in memory, so the KV cache is kept near 1 GB: every GB given to it removes ~75 cached experts and slows decoding. Raise Context if you need it.")
        if gptoss: s["max_new"] = 8192 if fits else 4096; notes.append(f"Max reply {s['max_new']:,} tokens (large files are written in 3,000-character tool calls, so this mostly bounds reasoning + answer).")
    except Exception as ex: notes.append(f"(context estimate unavailable: {ex})")'''),
    # never lose trimmed turns: archive them
    ('''                while len(ids) + int(cfg["max_new"]) > int(cfg["ctx"]):
                    old = next((i for i in range(1 if system else 0, last_user)), None)       # never drop the current request
                    if old is None: break
                    del msgs[old]; last_user -= 1; ids = self._ids(msgs, tools)''',
     '''                dropped = []
                while len(ids) + len(cont) + int(cfg["max_new"]) > int(cfg["ctx"]):
                    old = next((i for i in range(1 if system else 0, last_user)), None)       # never drop the current request
                    if old is None: break
                    dropped.append(msgs.pop(old)); last_user -= 1; ids = self._ids(msgs, tools)
                if dropped:                       # nothing is lost: the dropped turns go to a memory artifact the model can read
                    note = self._archive(history, dropped, A)
                    if note:
                        msgs.insert(1 if system else 0, {"role": "user", "content": note}); ids = self._ids(msgs, tools)'''),
    ('''        rnd = -1; budget = max_tool_rounds + 1; chunks = 0; A = plugins.get("artifact")''',
     '''        rnd = -1; budget = max_tool_rounds + 1; chunks = 0; A = plugins.get("artifact"); cont = []; conts = 0'''),
    ('''            shown = {"thinking": "", "assistant": ""}; last_pc = [0.0, -1];''', '''            pre = list(cont); ids = ids + pre if self.chat_mode else ids
            shown = {"thinking": "", "assistant": ""}; last_pc = [0.0, -1];'''),
    ('''                raw_now = self._decode(out); r = parse_reply(raw_now)''', '''                raw_now = self._decode(pre + out); r = parse_reply(raw_now)'''),
    ('''            raw = self._decode(out); r = parse_reply(raw); text = r["answer"]''',
     '''            out = pre + out; cont = []
            if reason == "ctx" and self.chat_mode and conts < 4 and cut["at"] is None and not cut["ui"]:
                cont = out; conts += 1; budget += 1; on_text("act", {"ev": "round", "n": rnd + 2})       # compact the history and carry on
                continue
            raw = self._decode(out); r = parse_reply(raw); text = r["answer"]'''),
    ('''    @property
    def harmony(self):''', '''    def _archive(self, history, dropped, A):
        """append trimmed turns verbatim to the artifact memory-<id> and return the note that replaces them"""
        if A is None: return None
        import hashlib
        first = next((m.get("content", "") for m in history if m.get("role") == "user"), "") or "chat"
        name = "memory-" + hashlib.sha1(first.encode("utf-8", "ignore")).hexdigest()[:8]
        lines = []
        for m in dropped:
            body = m.get("content") or ""
            if m.get("tool_calls"): body = (body + "\\n" if body else "") + json.dumps([tc.get("function", tc) for tc in m["tool_calls"]], ensure_ascii=False)
            lines.append(f"### {m['role']}\\n{body}\\n")
        if not (A.ROOT / name / ".artifact.json").exists(): A.run("create", name=name, type="note", title="Earlier conversation (moved out of the context window)")
        A.run("append", name=name, content="\\n".join(lines))
        return (f"[Harness note: {len(dropped)} earlier message(s) of this conversation were moved out of the context window into the artifact "
                f"'{name}' (notes.md), word for word. If you need them, read it with the artifact tool (action 'read', name '{name}').]")

    @property
    def harmony(self):'''),
])

patch("harness_app.py", [
    ('''("ctx", "Context length (tokens)", "int", (256, 32768)),''', '''("ctx", "Context length (tokens)", "int", (256, 262144)),'''),
    ('''        ("max_new", "Max reply tokens", "int", (16, 8192)),''', '''        ("max_new", "Max reply tokens", "int", (16, 131072)),'''),
    ('''        ("top_k", "Top-k", "int", (1, 128)),''', '''        ("top_k", "Top-k (0 = off)", "int", (0, 128)),'''),
    ('''        ("think", "Let the model think first (Qwen3 / Hermes)", "bool", None),''', '''        ("think", "Let the model think first (Qwen3 / Hermes)", "bool", None), ("kv16", "KV cache in fp16 (half the memory, ~2e-4 rounding)", "bool", None),'''),
])
patch("harness_paths.py", [('''"ctx": 2048, "max_new": 512,''', '''"ctx": 2048, "max_new": 512, "kv16": False,''')])
