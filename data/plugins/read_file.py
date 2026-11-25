"""read_file - let the model read a text file, but only inside the harness folder's 'workspace' subfolder (read-only)."""
from pathlib import Path

NAME = "read_file"
DESCRIPTION = "Read a UTF-8 text file from the harness workspace folder. Path is relative to the workspace, e.g. 'notes.txt'. Returns at most 8000 characters. Use path '.' to list the files."
PARAMETERS = {"type": "object", "properties": {"path": {"type": "string", "description": "file path relative to the workspace folder, or '.' to list files"}}, "required": ["path"]}
WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"


def run(path: str = ".") -> str:
    WORKSPACE.mkdir(exist_ok=True)
    p = (WORKSPACE / path).resolve()
    if WORKSPACE.resolve() not in (p, *p.parents): return "error: outside the workspace folder"
    if p.is_dir(): return "\n".join(sorted(str(x.relative_to(WORKSPACE)) for x in p.rglob("*") if x.is_file())[:200]) or "(workspace is empty)"
    if not p.exists(): return f"error: {path} not found"
    try: return p.read_text(encoding="utf-8", errors="replace")[:8000]
    except Exception as ex: return f"error: {ex}"
