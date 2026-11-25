"""make_core_toolskill.py - harness_core.py: auto-applied tool skill, lenient JSON for tool arguments, clear cut-off errors.

  skills with 'applies_when: tools' are added to the system prompt whenever tools are on; '## format: harmony' / '## format: tool_call'
  sections are kept only for the model's own call format; '{tools}' becomes the list of enabled tools and their actions
  tool arguments: parsed with strict=False (raw newlines/tabs inside strings are accepted)
  a call cut off before it finished (reply hit max new tokens, or the model stopped mid-call) gets an error that says so and
  tells the model to write the file in parts (write, then append)"""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_core.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

CUT = ("cut off before it finished. Your call was too long for one reply: write the file in parts - "
       "artifact action 'write' with the first part (under ~3000 characters), then action 'append' for each next part.")
rep('''            c = json.loads(blob); args = c.get("arguments", {})
            if isinstance(args, str): args = json.loads(args or "{}")''',
'''            c = json.loads(blob, strict=False); args = c.get("arguments", {})
            if isinstance(args, str): args = json.loads(args or "{}", strict=False)''')
rep('''            calls.append(("", {"_error": f"could not parse tool call: {blob[:200]}"}))
    return calls''',
'''            calls.append(("", {"_error": f"could not parse tool call JSON: {blob[:200]}"}))
    i = text.rfind("<tool_call>")
    if i >= 0 and "</tool_call>" not in text[i:]:          # opened but never closed: the reply ended mid-call
        calls.append(("", {"_error": "the tool call was " + CUT_MSG, "_cut": True}))
    return calls''')
rep('''HARMONY_END = ("<|end|>", "<|return|>", "<|call|>")''', f'''HARMONY_END = ("<|end|>", "<|return|>", "<|call|>")
CUT_MSG = {CUT!r}''')
rep('''                try: args = json.loads(body.strip() or "{}")
                except Exception: args = {"_error": f"could not parse tool arguments: {body[:200]}"}''',
'''                b = body.strip()
                try: args = json.loads(b or "{}", strict=False)
                except Exception as ex:
                    cut = not b.endswith("}")
                    args = {"_error": ("the tool call was " + CUT_MSG) if cut else f"tool arguments are not valid JSON ({ex}). Escape quotes as \\\\\\" and new lines as \\\\n inside strings.", "_cut": cut}''')
# when the reply stopped on the token limit, say so explicitly
rep('''            calls = [resolve_tool(plugins, n, a, enabled) for n, a in r["calls"]] if tools else []''',
'''            calls = [resolve_tool(plugins, n, a, enabled) for n, a in r["calls"]] if tools else []
            if reason == "max":
                calls = [(n, dict(a, _error=a["_error"] + f" (this reply stopped at the max-new-tokens limit of {cfg['max_new']})")) if "_error" in a else (n, a) for n, a in calls]''')
# auto-applied tool skill
rep('''        cfg = self.cfg; new_msgs = []; stats = ""
        for rnd in range(max_tool_rounds + 1):''',
'''        cfg = self.cfg; new_msgs = []; stats = ""
        if tools:
            auto = tool_skill_text(tools, plugins, self.harmony)
            if auto: system = (system + "\\n\\n" + auto).strip() if system else auto
        for rnd in range(max_tool_rounds + 1):''')
rep('''    def kill(self):''', '''    @property
    def harmony(self):
        """True for models that call tools in the harmony format (gpt-oss)"""
        t = getattr(self.tok, "chat_template", None) or ""
        return "<|channel|>" in (t if isinstance(t, str) else str(t))

    def kill(self):''')
AUTO = r'''

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
'''
rep("\n\ndef save_skill(", AUTO + "\n\ndef save_skill(")
rep('''"always_on": meta.get("always_on", "false").lower() == "true", "author": meta.get("author", ""),''',
    '''"always_on": meta.get("always_on", "false").lower() == "true", "author": meta.get("author", ""), "applies_when": meta.get("applies_when", ""),''')
# saving a skill from the Skills tab keeps its author / applies_when keys
rep('''    (hp.SKILLS / f"{name}.md").write_text(f"---\\nname: {name}\\ndescription: {description.strip()}\\nalways_on: {'true' if always_on else 'false'}\\n---\\n{body.strip()}\\n", encoding="utf-8")''',
'''    keep = ""
    for src in [hp.SKILLS / f"{name}.md"] + ([hp.SKILLS / f"{old_name}.md"] if old_name else []):
        if src.exists():
            meta, _ = _parse_skill(src.read_text(encoding="utf-8")); keep = "".join(f"{k}: {meta[k]}\\n" for k in ("author", "applies_when") if meta.get(k)); break
    (hp.SKILLS / f"{name}.md").write_text(f"---\\nname: {name}\\ndescription: {description.strip()}\\nalways_on: {'true' if always_on else 'false'}\\n{keep}---\\n{body.strip()}\\n", encoding="utf-8")''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_core: tool-calls skill (auto), lenient JSON, cut-off errors")
