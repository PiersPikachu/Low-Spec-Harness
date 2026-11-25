"""make_core_toolalias.py - harness_core.py: route tool calls whose name is really an action or a decorated tool name.

Seen: '-> create({"action": "create", ...})  [create] error: no plugin named create' - the model put the artifact action in the
function name. resolve_tool() maps, in order: an exact plugin name; 'functions.X' / 'tools.X' prefixes; 'artifact.create',
'artifact_create', 'artifact/create', 'artifact:create' (plugin + action); a bare action name ('create') that exactly one enabled
plugin lists in its 'action' enum. The call is stored under the resolved name, so the chat history the model sees next turn shows
the correct tool. Unresolvable names still return the error, now listing the available tools."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_core.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

rep('''def run_tool(plugins, name, args):
    m = plugins.get(name)
    if not m or not getattr(m, "run", None): return f"error: no plugin named {name}"''',
'''def _actions(m):
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
        return f"error: no tool named '{name}'. Available tools: {have}. Call a tool by its name and put the operation in its arguments (e.g. artifact with action='create')."''')
rep('''            calls = r["calls"] if tools else []''',
'''            enabled = {t["function"]["name"] for t in tools} if tools else set()
            calls = [resolve_tool(plugins, n, a, enabled) for n, a in r["calls"]] if tools else []''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_core: tool name resolution")
