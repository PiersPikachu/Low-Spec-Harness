"""make_core_cap.py - per-reply cap on file growth + better continuity between parts.

  cap        artifact content written in one reply (write / append / create content, edit's new text) is limited to
             cfg max_file_growth characters, or 4 x max_new when that setting is 0. A call that would pass the cap is saved up to
             the last whole line inside it, the model is told the limit is reached and to tell the user the file is incomplete
             (they continue in the next message), and further writes in that reply are refused.
  continuity each harness-ended part and the cap message include the last 40 saved lines and an outline of the whole file
             (headings, functions, classes, main HTML sections with line numbers), so the model keeps the file's structure."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_core.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

HELP = r'''

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
'''
rep("\n\ndef _compact_history(", HELP + "\n\ndef _compact_history(")
rep('''A = plugins.get("artifact"); cont = []; conts = 0; carry = None''',
    '''A = plugins.get("artifact"); cont = []; conts = 0; carry = None
        cap_chars = int(cfg.get("max_file_growth") or 0) or 4 * int(cfg["max_new"]); written = [0]; capped = [False]''')
# streaming: cut at the cap as well as at the 3000-character chunk
rep('''                if art and cut["at"] is None:
                    k = _chunk_cut(pc["text"])''',
'''                if art and cut["at"] is None:
                    room = cap_chars - written[0]
                    if len(pc["text"]) >= room: k = _cap_cut(pc["text"], room); cut["capped"] = True
                    else: k = _chunk_cut(pc["text"])''')
rep('''                    cut["at"] = _chunk_cut(pc["text"], final=True); cut["pc"] = dict(pc)''',
    '''                    cut["at"] = min(_chunk_cut(pc["text"], final=True), _cap_cut(pc["text"], cap_chars - written[0])); cut["pc"] = dict(pc)
                    if cut["at"] < len(pc["text"].rstrip("\\n")) and cut["at"] == _cap_cut(pc["text"], cap_chars - written[0]): cut["capped"] = True''')
rep('''            if cut["at"] and chunks < MAX_CHUNKS:                         # build the call ourselves from the streamed part''',
    '''            if cut["at"] is not None and cut["at"] == 0 and cut.get("capped"): cut["at"] = None; capped[0] = True       # nothing fits any more
            if cut["at"] and chunks < MAX_CHUNKS:                         # build the call ourselves from the streamed part''')
# before running calls: enforce the cap on complete calls too, and refuse writes once the cap is reached
rep('''            for name, args in calls:
                on_text("act", {"ev": "call", "name": name, "args": args})''',
'''            for i_c, (name, args) in enumerate(calls):
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
                on_text("act", {"ev": "call", "name": name, "args": args})''')
# continuity: outline + last 40 lines in the continuation and the cap message
rep('''                if chunk is not None and not res.startswith("error"):
                    tail = chunk.rstrip("\\n").rsplit("\\n", 1)[-1][-200:]
                    res += (f"\\nThat call was ended by the harness after {len(chunk)} characters (the end of a line), to keep each call small. "
                            f"Continue the same file now: call artifact with action 'append', name '{args['name']}'" + (f", path '{args['path']}'" if args.get("path") else "")
                            + f", with content that starts right after this last saved line:\\n{tail}\\nDo not repeat anything already saved. When the file is complete, stop calling tools and tell the user.")''',
'''                fctx = ""
                if name == "artifact" and A is not None and not res.startswith("error") and args.get("action") in ("write", "append", "create", "edit"):
                    try:
                        d = A.ROOT / A._slug(args["name"]); fp, _rel = A._file(d, args.get("path"), A._meta(d))
                        fctx = file_context(fp.read_text(encoding="utf-8", errors="replace"))
                    except Exception: fctx = ""
                if capped[0] and not res.startswith("error"):
                    res += (f"\\nThe per-reply file growth limit ({cap_chars:,} characters) is reached, so the harness saved the content up to the last whole line "
                            f"that fits. Stop writing now. Tell the user the file is incomplete, what remains to be written, and that they can say 'continue' "
                            f"in their next message to resume from the last saved line.\\n{fctx}")
                elif chunk is not None and not res.startswith("error"):
                    res += (f"\\nThat call was ended by the harness after {len(chunk)} characters (the end of a line), to keep each call small. "
                            f"Continue the same file now: call artifact with action 'append', name '{args['name']}'" + (f", path '{args['path']}'" if args.get("path") else "")
                            + f", with content that starts right after the last saved line shown below. Keep it consistent with the outline (same names, "
                            f"styles and structure). Do not repeat anything already saved. When the file is complete, stop calling tools and tell the user.\\n{fctx}")''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_core: per-reply file growth cap, outline + last 40 lines between parts")
