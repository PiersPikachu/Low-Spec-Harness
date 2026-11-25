---
name: tool-calls
description: How to call the harness tools correctly (format, JSON escaping, large files in parts). Added automatically whenever tools are on.
always_on: false
applies_when: tools
---
# Calling tools in this harness

Tools available right now:
{tools}

## Rules for every call
1. The tool name is the plugin name above (for example `artifact`). The operation goes in the `action` argument. Never use an action as the tool name.
2. Arguments must be valid JSON. Inside strings: write `\"` for a double quote, `\n` for a new line, `\\` for a backslash. In HTML/JS content, prefer single quotes for attributes and strings so fewer escapes are needed.
3. Make one tool call, then stop and wait for its result before the next call. Read each result: if it starts with `error`, fix exactly what it says and try again once.
4. Keep every call small. The harness ends a write/append call at the end of the line that passes 3000 characters, saves that part, and tells you the last saved line: continue with `action: "append"` starting right after that line, without repeating anything already saved.
5. Large files are written in parts, in order:
   - first `artifact` with `action: "write"` and the first part (for HTML: doctype, head, styles),
   - then `artifact` with `action: "append"` for each following part (markup, then each script section),
   - finish with `action: "read"` to check the file if it matters.
6. For small changes use `action: "edit"` with `old` set to a short piece of the file that appears exactly once, and `new` set to its replacement. Read the file first if you are not sure of the exact text.
7. To start something new, `action: "create"` first (with `type` and `title`), then write its content.
8. One reply can only add a limited amount of file content (set by the user). When a tool result says the limit is reached, stop writing: tell the user the file is incomplete and what is left, and continue from the last saved line when they say "continue". Each result shows the file's outline and last 40 lines - keep names, styles and structure consistent with them.

## format: harmony
Send each call on the commentary channel, addressed to the tool, with JSON as the message, and end it with the call token:
`<|start|>assistant<|channel|>commentary to=functions.artifact <|constrain|>json<|message|>{"action": "append", "name": "rubiks-cube", "path": "index.html", "content": "<script>\nconst size = 3;\n</script>\n"}<|call|>`
Give the final answer to the user on the final channel only after the tool results are in.

## format: tool_call
Write each call as one JSON object inside tags, then stop:
<tool_call>
{"name": "artifact", "arguments": {"action": "append", "name": "rubiks-cube", "path": "index.html", "content": "<script>\nconst size = 3;\n</script>\n"}}
</tool_call>
The JSON must have exactly two keys: `name` (the tool) and `arguments` (an object).
