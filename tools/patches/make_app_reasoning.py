"""make_app_reasoning.py - harness_app.py: reasoning preview, structured tool-call history, 'Model defaults' button. Edits in place."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_app.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

# styles for reasoning: live stream (dim italic) and the collapsed preview line (clickable)
rep('''        self.tx.tag_configure("info", foreground=DIM, font=("Segoe UI", 9, "italic"))''',
'''        self.tx.tag_configure("info", foreground=DIM, font=("Segoe UI", 9, "italic"))
        self.tx.tag_configure("think", foreground="#8a8f99", font=("Segoe UI", 10, "italic"), lmargin1=14, lmargin2=14)
        self.tx.tag_configure("think_prev", foreground="#a7adb8", font=("Segoe UI", 9), lmargin1=6)
        self._think_n = 0; self._think_full = {}; self._think_live = ""''')

# reasoning helpers: collapse a block into a one-line preview that expands on click
rep('''    def _render(self):''', '''    def _think_preview(self, text, at="end"):
        """insert a collapsed reasoning block: '> Thought for N words - first words...' ; click toggles the full text"""
        n = self._think_n = self._think_n + 1; self._think_full[n] = text.strip()
        words = len(text.split()); head = " ".join(text.split())[:110]; ell = "\\u2026" if len(head) == 110 else ""
        line = f"\\u25b8 Thought for {words} words \\u2014 {head}{ell}  (click to expand)\\n"
        self.tx.configure(state="normal"); self.tx.insert(at, line, ("think_prev", f"tp{n}")); self.tx.configure(state="disabled")
        self.tx.tag_bind(f"tp{n}", "<Button-1>", lambda e, n=n: self._toggle_think(n))
        self.tx.tag_bind(f"tp{n}", "<Enter>", lambda e: self.tx.configure(cursor="hand2")); self.tx.tag_bind(f"tp{n}", "<Leave>", lambda e: self.tx.configure(cursor=""))

    def _toggle_think(self, n):
        self.tx.configure(state="normal"); r = self.tx.tag_ranges(f"tx{n}")
        if r: self.tx.delete(r[0], r[-1])
        else:
            end = self.tx.tag_ranges(f"tp{n}")[-1]; self.tx.insert(end, self._think_full[n] + "\\n", ("think", f"tx{n}"))
        self.tx.configure(state="disabled")

    def _collapse_live_think(self):
        """the reasoning streamed so far becomes a preview line (called when the answer starts, or at the end)"""
        if not self._think_live: return
        r = self.tx.tag_ranges("think_live"); start = r[0] if r else "end"
        if r:
            self.tx.configure(state="normal"); self.tx.delete(r[0], r[-1]); self.tx.configure(state="disabled")
        self._think_preview(self._think_live, start); self._think_live = ""

    def _render(self):''')
rep('''            elif m["role"] == "assistant":
                self._write(f"{name}\\n", "asst_h"); self._write(m["content"] + "\\n")''',
'''            elif m["role"] == "assistant":
                self._write(f"{name}\\n", "asst_h")
                if m.get("thinking"): self._think_preview(m["thinking"])
                for tc in m.get("tool_calls", []):
                    f = tc.get("function", tc); self._write(f"-> {f.get('name')}({json.dumps(f.get('arguments', {}))[:160]})\\n", "tool")
                if m["content"]: self._write(m["content"] + "\\n")''')
# history goes to the model with its reasoning and structured tool calls (gpt-oss's template needs them)
rep('''            new, stats = s.reply([{"role": m["role"], "content": m["content"]} for m in history], self._system(), tools, self.plugins,''',
    '''            new, stats = s.reply([{k: v for k, v in m.items() if k != "meta"} for m in history], self._system(), tools, self.plugins,''')
# stream handlers
rep('''    def _on_assistant(self, t): self._write(t)''',
'''    def _on_thinking(self, t):
        self._think_live += t; self._write(t, ("think", "think_live"))
    def _on_assistant(self, t):
        if self._think_live: self._collapse_live_think(); self.tx.configure(state="normal"); self.tx.tag_remove("think_live", "1.0", "end"); self.tx.configure(state="disabled")
        self._write(t)''')
rep('''    def _on_tool(self, t): self._write("\\n" + t, "tool")''',
'''    def _on_tool(self, t):
        if self._think_live: self._collapse_live_think()
        self._write("\\n" + t, "tool")''')
rep('''        new, stats = val; self.conv["messages"] += new; core.save_conversation(self.conv)''',
'''        new, stats = val; self.conv["messages"] += new; core.save_conversation(self.conv)
        if self._think_live: self._collapse_live_think()
        self.tx.configure(state="normal"); self.tx.tag_remove("think_live", "1.0", "end"); self.tx.configure(state="disabled")''')
rep('''        self._write("\\n" + tb.strip().splitlines()[-1] + "\\n", "info"); self.status.set("error - see logs\\\\harness_error.txt")''',
'''        self._think_live = ""; self._write("\\n" + tb.strip().splitlines()[-1] + "\\n", "info"); self.status.set("error - see logs\\\\harness_error.txt")''')
# Config: 'Model defaults' button
rep('''        ttk.Button(bar, text="Revert", command=self._fill_config).pack(side="left")''',
'''        ttk.Button(bar, text="Revert", command=self._fill_config).pack(side="left")
        ttk.Button(bar, text="Model defaults", command=self._model_defaults).pack(side="left", padx=6)''')
rep('''    def _vault_names(self):''', '''    def _model_defaults(self):
        """fill the fields with the selected model's own recommended settings (not saved until Save)"""
        try: rec, notes = core.recommended_settings(self._sel_vault_path())
        except Exception as ex: messagebox.showerror("Model defaults", str(ex), parent=self); return
        for k, v in rec.items():
            if k in self.vars: self.vars[k].set(bool(v) if isinstance(v, bool) else str(v))
        self.cfg_plan.delete("1.0", "end")
        self.cfg_plan.insert("1.0", "\\n".join(notes) + "\\n\\nApplied to the fields on the left: " + ", ".join(f"{k}={v}" for k, v in rec.items())
                             + "\\nPress Save (or Save and reload model) to keep them.")
        self.status.set("model defaults filled in - press Save to keep them")

    def _vault_names(self):''')
if "import json" not in s: rep("import os, queue, subprocess", "import json, os, queue, subprocess")
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_app: reasoning preview, tool-call history, Model defaults button")
