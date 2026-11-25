"""artifact - persistent, editable artifacts the model creates and maintains across chats (notes, HTML pages, multi-file sites,
codebases, data). Stored under the harness folder in artifacts/<name>/ ; every change keeps the previous version of the file
in artifacts/<name>/.versions/ so any edit can be reverted (from the model with action=revert, or from the Artifacts tab).

The model never touches anything outside artifacts/. HTML artifacts open in the browser from the Artifacts tab."""
import datetime, json, re, shutil
from pathlib import Path

NAME = "artifact"
ROOT = Path(__file__).resolve().parent.parent / "artifacts"
TYPES = {"note": "notes.md", "page": "index.html", "site": "index.html", "code": "README.md", "data": "data.csv"}
MAX_WRITE = 2_000_000          # bytes per file write
MAX_FILES = 500                # per artifact
DESCRIPTION = (
    "Call this tool by the name 'artifact' and put the operation in the 'action' argument, e.g. artifact(action='create', "
    "name='my-notes', type='note'). Create and maintain persistent artifacts saved in the harness (kept across chats): notes, a single HTML page, a multi-file "
    "website, a codebase, or data files. Use 'list' to see what exists, 'read' before editing, and 'edit' for small changes "
    "(old text must match exactly once). Actions: list | create | files | read | write | edit | append | delete_file | history | revert. "
    "Types for create: note (notes.md), page (one self-contained index.html), site (index.html + other files), code (any files), "
    "data (csv/json). Paths are relative to the artifact, e.g. 'index.html' or 'src/app.js'. Every change can be undone with revert.")
PARAMETERS = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["list", "create", "files", "read", "write", "edit", "append", "delete_file", "history", "revert"],
               "description": "what to do"},
    "name": {"type": "string", "description": "artifact name (lowercase words joined by '-', e.g. 'trip-notes'); not needed for list"},
    "type": {"type": "string", "enum": list(TYPES), "description": "create only: kind of artifact"},
    "title": {"type": "string", "description": "create only: human-readable title"},
    "path": {"type": "string", "description": "file inside the artifact; defaults to its main file"},
    "content": {"type": "string", "description": "create / write / append: the text to put in the file"},
    "old": {"type": "string", "description": "edit: exact existing text to replace (must occur exactly once)"},
    "new": {"type": "string", "description": "edit: replacement text"},
    "version": {"type": "integer", "description": "revert: change number from history (restores the file as it was before that change)"},
    "start_line": {"type": "integer", "description": "read: first line to show (1-based), for long files"},
}, "required": ["action"]}


def _slug(s):
    s = re.sub(r"[^a-z0-9]+", "-", str(s or "").lower()).strip("-")[:60]
    return s


def _meta(d):
    try: return json.load(open(d / ".artifact.json", encoding="utf-8"))
    except Exception: return {"title": d.name, "type": "code", "main": "README.md", "created": "", "updated": "", "changes": []}


def _save_meta(d, m):
    m["updated"] = datetime.datetime.now().isoformat(timespec="seconds"); json.dump(m, open(d / ".artifact.json", "w", encoding="utf-8"), indent=1)


def _art(name, must_exist=True):
    n = _slug(name)
    if not n: raise ValueError("give the artifact a name, e.g. name='trip-notes'")
    d = ROOT / n
    if must_exist and not (d / ".artifact.json").exists():
        have = ", ".join(sorted(p.name for p in ROOT.iterdir() if (p / ".artifact.json").exists())) if ROOT.exists() else ""
        raise ValueError(f"no artifact named '{n}'" + (f"; existing: {have}" if have else "; create it first with action=create"))
    return d


def _file(d, path, m):
    rel = (path or m.get("main") or "").replace("\\", "/").strip("/")
    if not rel or rel.startswith(".versions") or rel == ".artifact.json": raise ValueError("give a file path inside the artifact")
    p = (d / rel).resolve()
    if d.resolve() not in p.parents: raise ValueError("path must stay inside the artifact folder")
    return p, rel


def _snapshot(d, m, rel, p, action):
    """keep the file as it was before this change so it can be reverted"""
    n = len(m["changes"]) + 1; vd = d / ".versions"; vd.mkdir(exist_ok=True)
    existed = p.exists()
    if existed: shutil.copy2(p, vd / f"{n:05d}.prev")
    m["changes"].append({"n": n, "time": datetime.datetime.now().isoformat(timespec="seconds"), "action": action, "path": rel, "existed": existed})
    return n


def _files(d):
    return sorted(str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file() and ".versions" not in p.parts and p.name != ".artifact.json")


# ---------------------------------------------------------------- live saving (used by the harness, not a tool action)
# While the model streams a write/append/create call, the harness saves the content-so-far into the file every half second,
# so the work is on disk even if the reply is cut off or the engine is killed. live_end() puts the file back exactly as it was
# before the call, so the completed call then runs normally and leaves one clean, revertible change in the history.
_LIVE = {}


def live_update(name, path, action, text, title=None, type=None):
    try:
        n = _slug(name); d = ROOT / n
        if not n: return None
        key = (n, (path or "").replace("\\", "/").strip("/"), action)
        st = _LIVE.get(key)
        if st is None:
            existed_art = (d / ".artifact.json").exists()
            if not existed_art and action != "create": return None
            if not existed_art:
                t = type if type in TYPES else "page"; main = key[1] or TYPES[t]; now = datetime.datetime.now().isoformat(timespec="seconds")
                d.mkdir(parents=True, exist_ok=True); json.dump({"title": title or name, "type": t, "main": main, "created": now, "updated": now, "changes": []},
                                                               open(d / ".artifact.json", "w", encoding="utf-8"), indent=1)
            m = _meta(d); p, rel = _file(d, key[1] or m.get("main"), m)
            st = _LIVE[key] = {"dir": d, "path": p, "rel": rel, "existed_art": existed_art, "existed": p.exists(),
                               "base": p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""}
        body = (st["base"] + ("" if not st["base"] or st["base"].endswith("\n") else "\n") + text) if action == "append" else text
        if len(body.encode("utf-8")) > MAX_WRITE: return st["rel"]
        st["path"].parent.mkdir(parents=True, exist_ok=True); st["path"].write_text(body, encoding="utf-8")
        return st["rel"]
    except Exception:
        return None


def live_end(name, path, action):
    """restore the pre-call state (the real tool call is about to run)"""
    key = (_slug(name), (path or "").replace("\\", "/").strip("/"), action); st = _LIVE.pop(key, None)
    if not st: return
    if not st["existed_art"]: shutil.rmtree(st["dir"], ignore_errors=True); return
    if st["existed"]: st["path"].write_text(st["base"], encoding="utf-8")
    elif st["path"].exists(): st["path"].unlink()


def live_keep():
    """force stop / crash path: keep whatever was saved live, forget the restore points"""
    _LIVE.clear()


def autosave(name, path, text, session):
    """the user's edits in the Artifacts tab: one revertible change per editing session, later saves update it in place"""
    d = _art(name); m = _meta(d); p, rel = _file(d, path, m)
    last = m["changes"][-1] if m.get("changes") else None
    if last and last.get("session") == session and last["path"] == rel:
        p.write_text(text, encoding="utf-8"); _save_meta(d, m); return f"autosaved {rel}"
    n = _snapshot(d, m, rel, p, "user edit"); m["changes"][-1]["session"] = session
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(text, encoding="utf-8"); _save_meta(d, m); return f"saved {rel} (change #{n})"


def run(action="list", name=None, type=None, title=None, path=None, content=None, old=None, new=None, version=None, start_line=None, **_):
    ROOT.mkdir(exist_ok=True)
    try:
        if action == "list":
            rows = []
            for d in sorted(ROOT.iterdir()):
                if (d / ".artifact.json").exists():
                    m = _meta(d); rows.append(f"{d.name}  [{m.get('type')}]  '{m.get('title')}'  {len(_files(d))} file(s), main {m.get('main')}, updated {m.get('updated')}")
            return "\n".join(rows) or "no artifacts yet - create one with action=create"
        if action == "create":
            d = _art(name, must_exist=False)
            if (d / ".artifact.json").exists(): return f"artifact '{d.name}' already exists - use write/edit to change it"
            t = type if type in TYPES else "note"; main = path.replace("\\", "/").strip("/") if path else TYPES[t]
            d.mkdir(parents=True, exist_ok=True); now = datetime.datetime.now().isoformat(timespec="seconds")
            m = {"title": title or name, "type": t, "main": main, "created": now, "updated": now, "changes": []}
            p, rel = _file(d, main, m); p.parent.mkdir(parents=True, exist_ok=True)
            body = content if content is not None else ({"note": f"# {title or name}\n\n", "data": ""}.get(t) or
                   ("<!doctype html>\n<html><head><meta charset='utf-8'><title>" + (title or name) + "</title></head><body>\n</body></html>\n" if t in ("page", "site") else f"# {title or name}\n"))
            if len(body.encode("utf-8")) > MAX_WRITE: return f"error: content larger than {MAX_WRITE} bytes"
            p.write_text(body, encoding="utf-8"); _save_meta(d, m)
            return f"created artifact '{d.name}' ({t}) with {rel} ({len(body)} chars). Open it in the Artifacts tab."
        d = _art(name); m = _meta(d)
        if action == "files":
            fs = _files(d); return "\n".join(f"{f}  {(d / f).stat().st_size} bytes" for f in fs) or "(no files)"
        if action == "history":
            ch = m.get("changes", [])[-30:]
            return "\n".join(f"#{c['n']}  {c['time']}  {c['action']}  {c['path']}" for c in ch) or "no changes recorded yet"
        if action == "revert":
            if version is None: return "error: give version (a change number from action=history)"
            c = next((c for c in m["changes"] if c["n"] == int(version)), None)
            if not c: return f"error: no change #{version}"
            p, rel = _file(d, c["path"], m); n = _snapshot(d, m, rel, p, f"revert #{c['n']}")
            prev = d / ".versions" / f"{c['n']:05d}.prev"
            if c["existed"]: p.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(prev, p)
            elif p.exists(): p.unlink()
            _save_meta(d, m)
            return (f"{rel} restored to how it was before change #{c['n']}" if c["existed"] else f"{rel} removed (it did not exist before change #{c['n']})") + f" (recorded as change #{n})"
        p, rel = _file(d, path, m)
        if action == "read":
            if not p.exists(): return f"error: {rel} does not exist; files: {', '.join(_files(d)) or '(none)'}"
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines(); s0 = max(1, int(start_line or 1)); out, size = [], 0
            for i in range(s0 - 1, len(lines)):
                row = f"{i + 1:5d}| {lines[i]}"; size += len(row) + 1
                if size > 12000: out.append(f"... ({len(lines) - i} more lines; read again with start_line={i + 1})"); break
                out.append(row)
            return f"{rel} ({len(lines)} lines):\n" + "\n".join(out)
        if action in ("write", "append"):
            if content is None: return "error: give content"
            if action == "write" and not p.exists() and len(_files(d)) >= MAX_FILES: return f"error: artifact already has {MAX_FILES} files"
            add = content if action == "write" else ((("\n" if p.exists() and not p.read_text(encoding="utf-8", errors="replace").endswith("\n") else "")) + content)
            new_text = add if action == "write" else (p.read_text(encoding="utf-8", errors="replace") if p.exists() else "") + add
            if len(new_text.encode("utf-8")) > MAX_WRITE: return f"error: file would exceed {MAX_WRITE} bytes"
            n = _snapshot(d, m, rel, p, action); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(new_text, encoding="utf-8"); _save_meta(d, m)
            return f"{action} {rel}: {len(new_text)} chars, {new_text.count(chr(10)) + 1} lines (change #{n})"
        if action == "edit":
            if old is None or new is None: return "error: give old and new"
            if not p.exists(): return f"error: {rel} does not exist"
            text = p.read_text(encoding="utf-8", errors="replace"); k = text.count(old)
            if k != 1: return f"error: old text found {k} times in {rel} - it must match exactly once (read the file and copy a longer, unique piece)"
            n = _snapshot(d, m, rel, p, "edit"); p.write_text(text.replace(old, new, 1), encoding="utf-8"); _save_meta(d, m)
            return f"edited {rel} (change #{n}): replaced {len(old)} chars with {len(new)}"
        if action == "delete_file":
            if not p.exists(): return f"error: {rel} does not exist"
            n = _snapshot(d, m, rel, p, "delete"); p.unlink(); _save_meta(d, m); return f"deleted {rel} (change #{n}, revertible)"
        return f"error: unknown action '{action}'"
    except ValueError as ex:
        return f"error: {ex}"
