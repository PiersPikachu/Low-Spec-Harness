"""harness_dev - lets the model extend the harness with its own skills and plugins, with the user in control.

  skills   written to skills/<name>.md with 'author: model' and always_on: false - usable as /name; the user decides in the
           Skills tab whether one is always on (and can edit or delete it there)
  plugins  written to plugins/pending/<name>.py and NOT loaded: Python runs with the user's permissions, so a model-written
           plugin only becomes a tool after the user reads it and presses 'Approve and enable' in the Plugins tab.
           Checked before saving (without running it): valid Python; defines NAME (= the file name), DESCRIPTION, PARAMETERS, run().
The model may only change or delete skills it wrote itself, and only pending plugins; built-in and user files are never touched."""
import ast, datetime, re
from pathlib import Path

NAME = "harness_dev"
ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"; PENDING = ROOT / "plugins" / "pending"; PLUGINS = ROOT / "plugins"
DESCRIPTION = (
    "Call this tool by the name 'harness_dev' with an 'action'. Extend the harness: write reusable skills (instruction snippets the "
    "user can invoke as /name) and new tool plugins (Python). Plugins are saved for the user's review and are not usable until the "
    "user approves them in the Plugins tab. Actions: list | write_skill | delete_skill | write_plugin | read_plugin | discard_plugin. "
    "A plugin is a Python module with NAME (same as its name), DESCRIPTION (str), PARAMETERS (JSON schema dict) and "
    "def run(**args) -> str, using only the standard library.")
PARAMETERS = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["list", "write_skill", "delete_skill", "write_plugin", "read_plugin", "discard_plugin"], "description": "what to do"},
    "name": {"type": "string", "description": "skill or plugin name: lowercase letters, digits, '_' or '-'"},
    "description": {"type": "string", "description": "write_skill: one line saying when to use the skill"},
    "body": {"type": "string", "description": "write_skill: the skill's instructions (markdown)"},
    "code": {"type": "string", "description": "write_plugin: the complete Python source of the plugin module"},
}, "required": ["action"]}
BUILTIN = {"artifact", "calculator", "clock", "read_file", "harness_dev"}


def _name(n, plugin=False):
    n = str(n or "").strip().lower()
    if plugin: n = n.replace("-", "_")
    if not re.fullmatch(r"[a-z][a-z0-9_\-]{0,40}", n): raise ValueError("name must start with a letter and use only a-z, 0-9, '_' or '-' (max 41 chars)")
    return n


def _skill_author(f):
    m = re.match(r"^---\s*\n(.*?)\n---", f.read_text(encoding="utf-8", errors="replace"), re.S)
    return dict(re.findall(r"^(\w+):\s*(.*)$", m.group(1), re.M)).get("author", "") if m else ""


def _check_plugin(name, code):
    """static checks only - the code is never executed here"""
    if len(code.encode("utf-8")) > 100_000: return "plugin source is larger than 100 KB"
    try: tree = ast.parse(code)
    except SyntaxError as ex: return f"syntax error line {ex.lineno}: {ex.msg}"
    names = {}; has_run = False
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name): names[t.id] = node.value
        if isinstance(node, ast.FunctionDef) and node.name == "run": has_run = True
    for need in ("NAME", "DESCRIPTION", "PARAMETERS"):
        if need not in names: return f"missing module-level {need} = ..."
    if not (isinstance(names["NAME"], ast.Constant) and names["NAME"].value == name): return f"NAME must be the string '{name}'"
    if not has_run: return "missing def run(**args) -> str"
    return None


def run(action="list", name=None, description=None, body=None, code=None, **_):
    try:
        SKILLS.mkdir(exist_ok=True); PENDING.mkdir(parents=True, exist_ok=True)
        if action == "list":
            sk = [f"/{f.stem}" + ("  (written by model)" if _skill_author(f) == "model" else "") for f in sorted(SKILLS.glob("*.md"))]
            pl = sorted(f.stem for f in PLUGINS.glob("*.py")); pe = sorted(f.stem for f in PENDING.glob("*.py"))
            return ("skills: " + (", ".join(sk) or "none") + "\nplugins (loaded): " + (", ".join(pl) or "none")
                    + "\nplugins waiting for the user's approval: " + (", ".join(pe) or "none"))
        if action == "write_skill":
            n = _name(name); f = SKILLS / f"{n}.md"
            if f.exists() and _skill_author(f) != "model": return f"error: skill '{n}' was written by the user - pick another name"
            if not body: return "error: give body (the skill's instructions)"
            desc = " ".join(str(description or "").split())[:200]
            f.write_text(f"---\nname: {n}\ndescription: {desc}\nalways_on: false\nauthor: model\nwritten: {datetime.datetime.now().isoformat(timespec='seconds')}\n---\n{body.strip()}\n", encoding="utf-8")
            return f"skill /{n} saved (off by default; the user can use it as /{n} or switch it on in the Skills tab)"
        if action == "delete_skill":
            n = _name(name); f = SKILLS / f"{n}.md"
            if not f.exists(): return f"error: no skill '{n}'"
            if _skill_author(f) != "model": return f"error: skill '{n}' was written by the user; only the user can delete it"
            f.unlink(); return f"skill /{n} deleted"
        if action == "write_plugin":
            n = _name(name, plugin=True)
            if n in BUILTIN or (PLUGINS / f"{n}.py").exists(): return f"error: a plugin named '{n}' already exists - pick another name"
            if not code: return "error: give code (the full Python module)"
            err = _check_plugin(n, code)
            if err: return f"error: plugin not saved - {err}"
            (PENDING / f"{n}.py").write_text(f"# written by the model on {datetime.datetime.now().isoformat(timespec='seconds')} - review before enabling\n" + code, encoding="utf-8")
            return f"plugin '{n}' saved for review. It is NOT active: the user must open the Plugins tab and press 'Approve and enable'."
        if action == "read_plugin":
            n = _name(name, plugin=True); f = PENDING / f"{n}.py"
            if not f.exists(): f = PLUGINS / f"{n}.py"
            return f.read_text(encoding="utf-8")[:12000] if f.exists() else f"error: no plugin '{n}'"
        if action == "discard_plugin":
            n = _name(name, plugin=True); f = PENDING / f"{n}.py"
            if not f.exists(): return f"error: no pending plugin '{n}' (approved plugins can only be removed by the user)"
            f.unlink(); return f"pending plugin '{n}' discarded"
        return f"error: unknown action '{action}'"
    except ValueError as ex:
        return f"error: {ex}"
