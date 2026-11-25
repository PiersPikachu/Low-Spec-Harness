#!/usr/bin/env python3
"""harness_app.py - the Cube Harness window.

Tabs:  Messages  new and saved conversations, streaming replies, Stop, tool calls shown inline
       Config    model, execution mode, sampling, speculative draft, system prompt, vault folders; hardware plan
       Plugins   tools the model may call (plugins/*.py) and the harness's connections (engines, vaults, folders)
       Skills    reusable instruction snippets (skills/*.md): '/name' at the start of a message inserts one; 'always on' adds it to the system prompt
Start it with "Cube Harness.bat" (no console window) or: python harness_app.py
"""
import json, os, queue, subprocess, sys, threading, time, traceback
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness_paths as hp
import harness_core as core

BG, PANEL, FIELD, FG, DIM, ACCENT, TOOL = "#1e1f22", "#2b2d31", "#313338", "#e3e5e8", "#949ba4", "#7aa2ff", "#c9a26b"
FONT, MONO = ("Segoe UI", 10), ("Consolas", 10)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        hp.ensure_dirs(); self.cfg = hp.load_config(); self.q = queue.Queue(); self.session = None; self.session_key = None
        self.busy = False; self.stop_flag = False; self.conv = None; self.convs = []
        self.plugins = core.load_plugins(); self.skills = core.load_skills()
        self.title("Cube Harness"); self.geometry("1240x820"); self.minsize(960, 620); self.configure(bg=BG); self._style()
        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=6, pady=(6, 0)); self.nb = nb
        self.t_msg, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(4))
        for f, name in ((self.t_msg, "Messages"), (self.t_cfg, "Config"), (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")
        self.status = tk.StringVar(value="ready - no model loaded")
        ttk.Label(self, textvariable=self.status, style="Status.TLabel", anchor="w").pack(fill="x", side="bottom")
        self._build_messages(); self._build_config(); self._build_plugins(); self._build_skills()
        self._refresh_convs(select_first=True)
        self.protocol("WM_DELETE_WINDOW", self._quit); self.after(40, self._poll)

    # ------------------------------------------------------------------ look
    def _style(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, fieldbackground=FIELD, font=FONT, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
        s.configure("TFrame", background=BG); s.configure("Panel.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=FG); s.configure("Panel.TLabel", background=PANEL)
        s.configure("Dim.TLabel", background=BG, foreground=DIM); s.configure("Head.TLabel", background=BG, foreground=FG, font=("Segoe UI Semibold", 12))
        s.configure("Status.TLabel", background=PANEL, foreground=DIM, padding=(10, 4))
        s.configure("TNotebook", background=BG, borderwidth=0); s.configure("TNotebook.Tab", background=PANEL, foreground=DIM, padding=(14, 6))
        s.map("TNotebook.Tab", background=[("selected", FIELD)], foreground=[("selected", FG)])
        s.configure("TButton", background=FIELD, foreground=FG, padding=(10, 4), borderwidth=0); s.map("TButton", background=[("active", "#3c3f45"), ("disabled", PANEL)], foreground=[("disabled", DIM)])
        s.configure("Accent.TButton", background=ACCENT, foreground="#101114"); s.map("Accent.TButton", background=[("active", "#9ab8ff"), ("disabled", PANEL)])
        s.configure("TEntry", fieldbackground=FIELD, foreground=FG, insertcolor=FG); s.configure("TSpinbox", fieldbackground=FIELD, foreground=FG, arrowcolor=FG)
        s.configure("TCombobox", fieldbackground=FIELD, foreground=FG, arrowcolor=FG); s.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", FG)])
        s.configure("TCheckbutton", background=BG, foreground=FG); s.map("TCheckbutton", background=[("active", BG)])
        s.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=FG, rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL, foreground=DIM, borderwidth=0); s.map("Treeview", background=[("selected", "#404552")])
        s.configure("TLabelframe", background=BG, bordercolor=PANEL); s.configure("TLabelframe.Label", background=BG, foreground=DIM)
        self.option_add("*TCombobox*Listbox.background", FIELD); self.option_add("*TCombobox*Listbox.foreground", FG)

    def _text(self, parent, **kw):
        t = (ScrolledText if kw.pop("scroll", True) else tk.Text)(parent, bg=FIELD, fg=FG, insertbackground=FG, relief="flat", borderwidth=0,
                                                                 font=kw.pop("font", FONT), wrap="word", padx=10, pady=8, **kw)
        return t

    # ------------------------------------------------------------------ worker <-> window
    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait(); getattr(self, f"_on_{kind}")(val) if hasattr(self, f"_on_{kind}") else None
        except queue.Empty:
            pass
        self.after(40, self._poll)

    def _quit(self):
        if self.session: self.session.close()
        self.destroy()

    # ================================================================== Messages
    def _build_messages(self):
        left = ttk.Frame(self.t_msg, style="Panel.TFrame", width=260); left.pack(side="left", fill="y"); left.pack_propagate(False)
        ttk.Button(left, text="+  New chat", style="Accent.TButton", command=self.new_chat).pack(fill="x", padx=10, pady=(10, 6))
        self.conv_list = tk.Listbox(left, bg=PANEL, fg=FG, selectbackground="#404552", selectforeground=FG, activestyle="none",
                                    borderwidth=0, highlightthickness=0, font=FONT)
        self.conv_list.pack(fill="both", expand=True, padx=6); self.conv_list.bind("<<ListboxSelect>>", self.on_select_conv)
        row = ttk.Frame(left, style="Panel.TFrame"); row.pack(fill="x", padx=10, pady=8)
        ttk.Button(row, text="Rename", command=self.rename_chat).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(row, text="Delete", command=self.delete_chat).pack(side="left", expand=True, fill="x")
        right = ttk.Frame(self.t_msg); right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        head = ttk.Frame(right); head.pack(fill="x", pady=(8, 6))
        self.model_var = tk.StringVar(value="no model loaded"); ttk.Label(head, textvariable=self.model_var, style="Head.TLabel").pack(side="left")
        ttk.Button(head, text="Unload", command=self.unload_model).pack(side="right")
        self.load_btn = ttk.Button(head, text="Load model", command=self.load_model); self.load_btn.pack(side="right", padx=6)
        self.tx = self._text(right, state="disabled", font=("Segoe UI", 11)); self.tx.pack(fill="both", expand=True)
        self.tx.tag_configure("user_h", foreground=ACCENT, font=("Segoe UI Semibold", 11), spacing1=10)
        self.tx.tag_configure("asst_h", foreground="#8fd19e", font=("Segoe UI Semibold", 11), spacing1=10)
        self.tx.tag_configure("tool", foreground=TOOL, font=MONO, lmargin1=18, lmargin2=18)
        self.tx.tag_configure("info", foreground=DIM, font=("Segoe UI", 9, "italic"))
        self.tx.tag_configure("think", foreground="#8a8f99", font=("Segoe UI", 10, "italic"), lmargin1=14, lmargin2=14)
        self.tx.tag_configure("think_prev", foreground="#a7adb8", font=("Segoe UI", 9), lmargin1=6)
        self._think_n = 0; self._think_full = {}; self._think_live = ""
        box = ttk.Frame(right); box.pack(fill="x", pady=8)
        self.inp = self._text(box, height=4, scroll=False); self.inp.pack(side="left", fill="x", expand=True)
        self.inp.bind("<Control-Return>", lambda e: (self.send(), "break")[1])
        btns = ttk.Frame(box); btns.pack(side="left", fill="y", padx=(8, 0))
        self.send_btn = ttk.Button(btns, text="Send", style="Accent.TButton", command=self.send); self.send_btn.pack(fill="x")
        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled"); self.stop_btn.pack(fill="x", pady=(6, 0))
        self.hint = tk.StringVar(); ttk.Label(right, textvariable=self.hint, style="Dim.TLabel").pack(anchor="w", pady=(0, 6)); self._update_hint()

    def _update_hint(self):
        tools = ", ".join(n for n in self.cfg["enabled_plugins"] if n in self.plugins) or "none"
        skills = " ".join("/" + s for s in self.skills) or "none"
        self.hint.set(f"Ctrl+Enter sends.   Skills: {skills}   Tools on: {tools}")

    def _write(self, text, tag=None):
        self.tx.configure(state="normal"); self.tx.insert("end", text, tag); self.tx.configure(state="disabled"); self.tx.see("end")

    def _think_preview(self, text, at="end"):
        """insert a collapsed reasoning block: '> Thought for N words - first words...' ; click toggles the full text"""
        n = self._think_n = self._think_n + 1; self._think_full[n] = text.strip()
        words = len(text.split()); head = " ".join(text.split())[:110]; ell = "\u2026" if len(head) == 110 else ""
        line = f"\u25b8 Thought for {words} words \u2014 {head}{ell}  (click to expand)\n"
        self.tx.configure(state="normal"); self.tx.insert(at, line, ("think_prev", f"tp{n}")); self.tx.configure(state="disabled")
        self.tx.tag_bind(f"tp{n}", "<Button-1>", lambda e, n=n: self._toggle_think(n))
        self.tx.tag_bind(f"tp{n}", "<Enter>", lambda e: self.tx.configure(cursor="hand2")); self.tx.tag_bind(f"tp{n}", "<Leave>", lambda e: self.tx.configure(cursor=""))

    def _toggle_think(self, n):
        self.tx.configure(state="normal"); r = self.tx.tag_ranges(f"tx{n}")
        if r: self.tx.delete(r[0], r[-1])
        else:
            end = self.tx.tag_ranges(f"tp{n}")[-1]; self.tx.insert(end, self._think_full[n] + "\n", ("think", f"tx{n}"))
        self.tx.configure(state="disabled")

    def _collapse_live_think(self):
        """the reasoning streamed so far becomes a preview line (called when the answer starts, or at the end)"""
        if not self._think_live: return
        r = self.tx.tag_ranges("think_live"); start = r[0] if r else "end"
        if r:
            self.tx.configure(state="normal"); self.tx.delete(r[0], r[-1]); self.tx.configure(state="disabled")
        self._think_preview(self._think_live, start); self._think_live = ""

    def _render(self):
        self.tx.configure(state="normal"); self.tx.delete("1.0", "end"); self.tx.configure(state="disabled")
        if not self.conv: return
        name = Path(self.conv.get("vault") or self.cfg["vault"] or "model").name
        for m in self.conv["messages"]:
            meta = m.get("meta", {})
            if m["role"] == "user":
                self._write("You\n", "user_h"); self._write(meta.get("typed", m["content"]) + "\n")
                if meta.get("skills"): self._write(f"skills: {', '.join('/' + s for s in meta['skills'])}\n", "info")
            elif m["role"] == "assistant":
                self._write(f"{name}\n", "asst_h")
                if m.get("thinking"): self._think_preview(m["thinking"])
                for tc in m.get("tool_calls", []):
                    f = tc.get("function", tc); self._write(f"-> {f.get('name')}({json.dumps(f.get('arguments', {}))[:160]})\n", "tool")
                if m["content"]: self._write(m["content"] + "\n")
                if meta.get("stats"): self._write(meta["stats"] + "\n", "info")
            elif m["role"] == "tool":
                self._write(f"[{meta.get('tool', 'tool')}] {m['content']}\n", "tool")

    # conversations
    def _refresh_convs(self, select_first=False):
        self.convs = core.list_conversations(); self.conv_list.delete(0, "end")
        for c in self.convs: self.conv_list.insert("end", f"  {c['title'][:34]}")
        if select_first:
            if self.convs: self.conv_list.selection_set(0); self.conv = self.convs[0]; self._render()
            else: self.new_chat()
        elif self.conv:
            for i, c in enumerate(self.convs):
                if c["id"] == self.conv["id"]: self.conv_list.selection_set(i)

    def new_chat(self):
        if self.busy: return
        self.conv = core.new_conversation(self.cfg["vault"]); core.save_conversation(self.conv); self._refresh_convs(); self._render(); self.inp.focus_set()

    def on_select_conv(self, _=None):
        if self.busy: return
        sel = self.conv_list.curselection()
        if sel: self.conv = self.convs[sel[0]]; self._render()

    def rename_chat(self):
        if not self.conv: return
        t = simpledialog.askstring("Rename chat", "New title:", initialvalue=self.conv["title"], parent=self)
        if t: self.conv["title"] = t.strip(); core.save_conversation(self.conv); self._refresh_convs()

    def delete_chat(self):
        if not self.conv or self.busy: return
        if messagebox.askyesno("Delete chat", f"Delete '{self.conv['title']}' permanently?", parent=self):
            core.delete_conversation(self.conv["id"]); self.conv = None; self._refresh_convs(select_first=True)

    # model
    def _key(self):
        c = self.cfg; return (c["vault"], c["mode"], int(c["gpu_layers"] or 0), int(c["ctx"]), c["draft"], int(c["k"]), bool(c["greedy"]))

    def _ensure_session(self):
        if self.session and self.session_key == self._key(): return
        if self.session: self.session.close(); self.session = None
        if not self.cfg["vault"]:
            vs = hp.find_vaults(self.cfg)
            if not vs: raise RuntimeError("no vault found - run setup.bat or add a vault folder in Config")
            self.cfg["vault"] = next(iter(vs)); hp.save_config(self.cfg)
        self.q.put(("info", f"loading {self.cfg['vault']} ...\n")); t0 = time.time()
        self.session = core.Session(self.cfg); self.session_key = self._key()
        self.q.put(("loaded", f"{self.session.vault.name} - {self.session.how}  (ready in {time.time() - t0:.0f}s)"))

    def load_model(self):
        if self.busy: return
        self._begin(); threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try: self._ensure_session(); self.q.put(("idle", None))
        except BaseException: self.q.put(("error", traceback.format_exc()))     # BaseException: planner/engines may raise SystemExit

    def unload_model(self):
        if self.busy: return
        if self.session: self.session.close(); self.session = None; self.session_key = None
        self.model_var.set("no model loaded"); self.status.set("model unloaded")

    def _system(self):
        parts = [self.cfg.get("system", "").strip()] + [s["body"] for s in self.skills.values() if s["always_on"]]
        return "\n\n".join(p for p in parts if p)

    def send(self):
        if self.busy: return
        typed = self.inp.get("1.0", "end").strip()
        if not typed: return
        if not self.conv: self.new_chat()
        self.inp.delete("1.0", "end"); content, used = core.expand_skills(typed, self.skills)
        self.conv["messages"].append({"role": "user", "content": content, "meta": {"typed": typed, "skills": used}})
        if self.conv["title"] == "New chat": self.conv["title"] = typed.splitlines()[0][:40]
        self.conv["vault"] = self.cfg["vault"]; core.save_conversation(self.conv); self._refresh_convs(); self._render()
        self._begin(); threading.Thread(target=self._reply_worker, args=(list(self.conv["messages"]),), daemon=True).start()

    def _reply_worker(self, history):
        try:
            self._ensure_session(); s = self.session
            tools = core.tool_schemas(self.plugins, set(self.cfg["enabled_plugins"])) if s.chat_mode else []
            self.q.put(("start", s.vault.name))
            new, stats = s.reply([{k: v for k, v in m.items() if k != "meta"} for m in history], self._system(), tools, self.plugins,
                                 lambda kind, t: self.q.put((kind, t)), lambda: self.stop_flag, int(self.cfg.get("max_tool_rounds", 4)))
            self.q.put(("done", (new, stats)))
        except BaseException:
            self.q.put(("error", traceback.format_exc()))

    def stop(self):
        self.stop_flag = True; self.status.set("stopping after the current step ...")

    def _begin(self):
        self.busy = True; self.stop_flag = False
        for b in (self.send_btn, self.load_btn): b.configure(state="disabled")
        self.stop_btn.configure(state="normal"); self.status.set("working ...")

    def _on_idle(self, _):
        self.busy = False
        for b in (self.send_btn, self.load_btn): b.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def _on_info(self, t): self._write(t, "info")
    def _on_loaded(self, t): self.model_var.set(t.split("  (")[0]); self.status.set(t); self.cfg_plan.delete("1.0", "end"); self.cfg_plan.insert("1.0", self.session.banner)
    def _on_start(self, name): self._write(f"{name}\n", "asst_h")
    def _on_thinking(self, t):
        self._think_live += t; self._write(t, ("think", "think_live"))
    def _on_assistant(self, t):
        if self._think_live: self._collapse_live_think(); self.tx.configure(state="normal"); self.tx.tag_remove("think_live", "1.0", "end"); self.tx.configure(state="disabled")
        self._write(t)
    def _on_assistant_next(self, _): self._write(f"{self.session.vault.name}\n", "asst_h")
    def _on_tool(self, t):
        if self._think_live: self._collapse_live_think()
        self._write("\n" + t, "tool")

    def _on_done(self, val):
        new, stats = val; self.conv["messages"] += new; core.save_conversation(self.conv)
        if self._think_live: self._collapse_live_think()
        self.tx.configure(state="normal"); self.tx.tag_remove("think_live", "1.0", "end"); self.tx.configure(state="disabled")
        self._write("\n" + stats + "\n", "info"); self.status.set(f"{self.session.how}  |  last reply: {stats}"); self._on_idle(None)

    def _on_error(self, tb):
        self._think_live = ""; self._write("\n" + tb.strip().splitlines()[-1] + "\n", "info"); self.status.set("error - see logs\\harness_error.txt")
        (hp.LOGS / "harness_error.txt").write_text(tb, encoding="utf-8"); self._on_idle(None)

    # ================================================================== Config
    FIELDS = [  # key, label, kind, options
        ("vault", "Model vault", "vault", None), ("mode", "Execution mode", "combo", ["auto", "cpu", "gpu", "split"]),
        ("gpu_layers", "GPU layers (0 = planner decides)", "int", (0, 80)), ("ctx", "Context length (tokens)", "int", (256, 32768)),
        ("max_new", "Max reply tokens", "int", (16, 8192)), ("temp", "Temperature", "float", None), ("top_p", "Top-p", "float", None),
        ("top_k", "Top-k", "int", (1, 128)), ("greedy", "Greedy (deterministic)", "bool", None), ("seed", "Seed (blank = random)", "str", None),
        ("draft", "Speculative draft", "draft", None), ("k", "Draft tokens per round (k)", "int", (1, 16)),
        ("think", "Let the model think first (Qwen3 / Hermes)", "bool", None), ("max_tool_rounds", "Max tool calls per reply", "int", (0, 16)),
    ]

    def _build_config(self):
        left = ttk.Frame(self.t_cfg); left.pack(side="left", fill="y", padx=14, pady=10)
        ttk.Label(left, text="Settings", style="Head.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.vars = {}
        for i, (key, label, kind, opt) in enumerate(self.FIELDS, start=1):
            ttk.Label(left, text=label).grid(row=i, column=0, sticky="w", pady=3, padx=(0, 12))
            if kind == "bool":
                v = tk.BooleanVar(); w = ttk.Checkbutton(left, variable=v)
            elif kind in ("combo", "vault", "draft"):
                v = tk.StringVar(); w = ttk.Combobox(left, textvariable=v, width=34, state="readonly" if kind == "combo" else "normal")
                if kind == "combo": w["values"] = opt
                setattr(self, f"cb_{key}", w)
            elif kind == "int":
                v = tk.StringVar(); w = ttk.Spinbox(left, from_=opt[0], to=opt[1], textvariable=v, width=12)
            else:
                v = tk.StringVar(); w = ttk.Entry(left, textvariable=v, width=14)
            w.grid(row=i, column=1, sticky="w", pady=3); self.vars[key] = v
        r = len(self.FIELDS) + 1
        ttk.Label(left, text="System prompt").grid(row=r, column=0, sticky="nw", pady=(10, 3))
        self.sys_txt = self._text(left, height=6, width=46, scroll=False); self.sys_txt.grid(row=r, column=1, sticky="w", pady=(10, 3))
        ttk.Label(left, text="Extra vault folders").grid(row=r + 1, column=0, sticky="nw", pady=(10, 3))
        vf = ttk.Frame(left); vf.grid(row=r + 1, column=1, sticky="w", pady=(10, 3))
        self.vdirs = tk.Listbox(vf, height=4, width=46, bg=FIELD, fg=FG, borderwidth=0, highlightthickness=0, font=FONT); self.vdirs.pack(side="left")
        vb = ttk.Frame(vf); vb.pack(side="left", padx=6)
        ttk.Button(vb, text="Add...", command=self._add_vdir).pack(fill="x"); ttk.Button(vb, text="Remove", command=self._del_vdir).pack(fill="x", pady=(4, 0))
        bar = ttk.Frame(left); bar.grid(row=r + 2, column=0, columnspan=2, sticky="w", pady=14)
        ttk.Button(bar, text="Save", style="Accent.TButton", command=self.save_config).pack(side="left")
        ttk.Button(bar, text="Save and reload model", command=lambda: (self.save_config(), self.unload_model(), self.load_model())).pack(side="left", padx=6)
        ttk.Button(bar, text="Revert", command=self._fill_config).pack(side="left")
        ttk.Button(bar, text="Model defaults", command=self._model_defaults).pack(side="left", padx=6)
        right = ttk.Frame(self.t_cfg); right.pack(side="left", fill="both", expand=True, padx=(4, 14), pady=10)
        top = ttk.Frame(right); top.pack(fill="x")
        ttk.Label(top, text="Hardware and execution plan", style="Head.TLabel").pack(side="left")
        ttk.Button(top, text="Re-measure", command=lambda: self._planner(["probe", "--force"])).pack(side="right")
        ttk.Button(top, text="Show plan", command=lambda: self._planner(["plan", str(self._sel_vault_path())])).pack(side="right", padx=6)
        self.cfg_plan = self._text(right, font=MONO); self.cfg_plan.pack(fill="both", expand=True, pady=(8, 0))
        self.cfg_plan.insert("1.0", "The plan for the loaded model appears here after it loads, or press 'Plan for selected vault'.")
        self._fill_config()

    def _model_defaults(self):
        """fill the fields with the selected model's own recommended settings (not saved until Save)"""
        try: rec, notes = core.recommended_settings(self._sel_vault_path())
        except Exception as ex: messagebox.showerror("Model defaults", str(ex), parent=self); return
        for k, v in rec.items():
            if k in self.vars: self.vars[k].set(bool(v) if isinstance(v, bool) else str(v))
        self.cfg_plan.delete("1.0", "end")
        self.cfg_plan.insert("1.0", "\n".join(notes) + "\n\nApplied to the fields on the left: " + ", ".join(f"{k}={v}" for k, v in rec.items())
                             + "\nPress Save (or Save and reload model) to keep them.")
        self.status.set("model defaults filled in - press Save to keep them")

    def _vault_names(self):
        self.vault_map = hp.find_vaults(self.cfg); return list(self.vault_map)

    def _sel_vault_path(self):
        try: return hp.resolve_vault(self.vars["vault"].get(), self.cfg)
        except Exception: return self.vars["vault"].get()

    def _fill_config(self):
        names = self._vault_names(); self.cb_vault["values"] = names; self.cb_draft["values"] = ["auto", "none"] + names
        for key, _, kind, _ in self.FIELDS:
            val = self.cfg.get(key); self.vars[key].set(bool(val) if kind == "bool" else ("" if val is None else str(val)))
        self.sys_txt.delete("1.0", "end"); self.sys_txt.insert("1.0", self.cfg.get("system", ""))
        self.vdirs.delete(0, "end")
        for d in self.cfg.get("extra_vault_dirs", []): self.vdirs.insert("end", d)

    def save_config(self):
        c = dict(self.cfg)
        try:
            for key, label, kind, _ in self.FIELDS:
                v = self.vars[key].get()
                if kind == "bool": c[key] = bool(v)
                elif kind == "int": c[key] = int(v)
                elif kind == "float": c[key] = float(v)
                elif key == "seed": c[key] = int(v) if str(v).strip() else None
                else: c[key] = v.strip()
        except ValueError as ex:
            messagebox.showerror("Config", f"Bad value: {ex}", parent=self); return
        c["system"] = self.sys_txt.get("1.0", "end").strip(); c["extra_vault_dirs"] = list(self.vdirs.get(0, "end"))
        self.cfg = c; hp.save_config(c); self._fill_config(); self._refresh_connections(); self.status.set("settings saved to harness_config.json")

    def _add_vdir(self):
        d = filedialog.askdirectory(parent=self, title="Folder containing vaults (or a vault itself)")
        if d: self.vdirs.insert("end", hp.rel(d))

    def _del_vdir(self):
        for i in reversed(self.vdirs.curselection()): self.vdirs.delete(i)

    def _planner(self, args):
        self.cfg_plan.delete("1.0", "end"); self.cfg_plan.insert("1.0", f"planner.py {' '.join(args)} ...\n")
        def work():
            p = subprocess.run([sys.executable, str(hp.ROOT / "planner.py")] + args, capture_output=True, text=True, cwd=str(hp.ROOT),
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.q.put(("plan", (p.stdout + p.stderr).strip()))
        threading.Thread(target=work, daemon=True).start()

    def _on_plan(self, t): self.cfg_plan.delete("1.0", "end"); self.cfg_plan.insert("1.0", t)

    # ================================================================== Plugins
    def _build_plugins(self):
        top = ttk.LabelFrame(self.t_plug, text="  Tools the model can call (plugins\\*.py)  "); top.pack(fill="both", expand=True, padx=14, pady=(10, 6))
        self.ptree = ttk.Treeview(top, columns=("on", "name", "desc"), show="headings", height=7)
        for c, w, t in (("on", 70, "Enabled"), ("name", 140, "Plugin"), ("desc", 820, "What it does")): self.ptree.heading(c, text=t); self.ptree.column(c, width=w, anchor="w")
        self.ptree.pack(fill="both", expand=True, padx=8, pady=8); self.ptree.bind("<Double-1>", lambda e: self._toggle_plugin())
        bar = ttk.Frame(top); bar.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bar, text="Enable / disable", command=self._toggle_plugin).pack(side="left")
        ttk.Button(bar, text="Test...", command=self._test_plugin).pack(side="left", padx=6)
        ttk.Button(bar, text="Reload plugins", command=self._reload_plugins).pack(side="left")
        ttk.Button(bar, text="Open plugins folder", command=lambda: os.startfile(hp.PLUGINS)).pack(side="left", padx=6)
        ttk.Label(bar, text="A plugin is a .py file with NAME, DESCRIPTION, PARAMETERS (JSON schema) and run(**args) -> str.", style="Dim.TLabel").pack(side="left", padx=10)
        bot = ttk.LabelFrame(self.t_plug, text="  Connections (all relative to the harness folder)  "); bot.pack(fill="both", expand=True, padx=14, pady=(6, 10))
        self.ctree = ttk.Treeview(bot, columns=("what", "where", "status"), show="headings", height=9)
        for c, w, t in (("what", 230, "Component"), ("where", 560, "Location"), ("status", 240, "Status")): self.ctree.heading(c, text=t); self.ctree.column(c, width=w, anchor="w")
        self.ctree.pack(fill="both", expand=True, padx=8, pady=8)
        bar2 = ttk.Frame(bot); bar2.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bar2, text="Refresh", command=self._refresh_connections).pack(side="left")
        ttk.Button(bar2, text="Open harness folder", command=lambda: os.startfile(hp.ROOT)).pack(side="left", padx=6)
        ttk.Button(bar2, text="Open logs", command=lambda: os.startfile(hp.LOGS)).pack(side="left")
        self._refresh_plugins(); self._refresh_connections()

    def _refresh_plugins(self):
        self.ptree.delete(*self.ptree.get_children()); on = set(self.cfg["enabled_plugins"])
        for n, m in self.plugins.items():
            self.ptree.insert("", "end", iid=n, values=("yes" if n in on else "no", n, m.DESCRIPTION))
        self._update_hint()

    def _toggle_plugin(self):
        sel = self.ptree.selection()
        if not sel: return
        on = list(self.cfg["enabled_plugins"])
        for n in sel: on.remove(n) if n in on else on.append(n)
        self.cfg["enabled_plugins"] = on; hp.save_config(self.cfg); self._refresh_plugins()

    def _test_plugin(self):
        sel = self.ptree.selection()
        if not sel: return
        n = sel[0]; m = self.plugins[n]; props = (m.PARAMETERS or {}).get("properties", {}); args = {}
        for p, spec in props.items():
            v = simpledialog.askstring(f"Test {n}", f"{p}: {spec.get('description', '')}", parent=self)
            if v is None: return
            args[p] = v
        messagebox.showinfo(f"{n} result", core.run_tool(self.plugins, n, args)[:3000], parent=self)

    def _reload_plugins(self):
        self.plugins = core.load_plugins(); self._refresh_plugins(); self.status.set(f"{len(self.plugins)} plugins loaded")

    def _refresh_connections(self):
        import chat
        t = self.ctree; t.delete(*t.get_children()); ok = lambda b: "found" if b else "MISSING - run setup.bat"
        t.insert("", "end", values=("CPU / split engine", hp.rel(hp.ENGINE), ok(hp.ENGINE.exists())))
        t.insert("", "end", values=("GPU engine", hp.rel(hp.GPU_ENGINE), ok(hp.GPU_ENGINE.exists())))
        for name, p in hp.find_vaults(self.cfg).items():
            try:
                info = chat.idx_info(p); size = (p / "vault.bin").stat().st_size / 1e9
                codec = "rans4" if info.get("codec", "0") != "0" else "huffman4"; tok = (p / "tokenizer.json").exists() or (p / "vocab.json").exists()
                st = f"{size:.2f} GB {codec}, {info.get('layers', '?')} layers" + ("" if tok else ", NO TOKENIZER")
            except Exception as ex:
                st = f"unreadable: {ex}"
            t.insert("", "end", values=(f"vault  {name}", hp.rel(p), st))
        for label, d in (("conversations", hp.CONVERSATIONS), ("skills", hp.SKILLS), ("plugins", hp.PLUGINS), ("logs", hp.LOGS), ("settings", hp.CONFIG)):
            t.insert("", "end", values=(label, hp.rel(d), "found" if d.exists() else "missing"))

    # ================================================================== Skills
    def _build_skills(self):
        left = ttk.Frame(self.t_skill, style="Panel.TFrame", width=250); left.pack(side="left", fill="y"); left.pack_propagate(False)
        ttk.Button(left, text="+  New skill", style="Accent.TButton", command=self._new_skill).pack(fill="x", padx=10, pady=(10, 6))
        self.slist = tk.Listbox(left, bg=PANEL, fg=FG, selectbackground="#404552", activestyle="none", borderwidth=0, highlightthickness=0, font=FONT)
        self.slist.pack(fill="both", expand=True, padx=6, pady=(0, 10)); self.slist.bind("<<ListboxSelect>>", self._sel_skill)
        right = ttk.Frame(self.t_skill); right.pack(side="left", fill="both", expand=True, padx=14, pady=10)
        self.sk_name, self.sk_desc, self.sk_on = tk.StringVar(), tk.StringVar(), tk.BooleanVar(); self.sk_old = None
        g = ttk.Frame(right); g.pack(fill="x")
        ttk.Label(g, text="Name (use as /name)").grid(row=0, column=0, sticky="w", pady=3); ttk.Entry(g, textvariable=self.sk_name, width=30).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(g, text="Description").grid(row=1, column=0, sticky="w", pady=3); ttk.Entry(g, textvariable=self.sk_desc, width=80).grid(row=1, column=1, sticky="w", padx=8)
        ttk.Checkbutton(g, text="Always on (added to every system prompt)", variable=self.sk_on).grid(row=2, column=1, sticky="w", padx=8, pady=3)
        ttk.Label(right, text="Instructions").pack(anchor="w", pady=(10, 3))
        self.sk_body = self._text(right); self.sk_body.pack(fill="both", expand=True)
        bar = ttk.Frame(right); bar.pack(fill="x", pady=(8, 0))
        ttk.Button(bar, text="Save", style="Accent.TButton", command=self._save_skill).pack(side="left")
        ttk.Button(bar, text="Delete", command=self._del_skill).pack(side="left", padx=6)
        ttk.Button(bar, text="Use in next message", command=self._use_skill).pack(side="left")
        ttk.Label(bar, text="Start a message with /name to insert a skill; several can be chained: /concise /verify ...", style="Dim.TLabel").pack(side="left", padx=10)
        self._refresh_skills()

    def _refresh_skills(self, select=None):
        self.skills = core.load_skills(); self.slist.delete(0, "end"); names = list(self.skills)
        for n in names: self.slist.insert("end", f"  {'* ' if self.skills[n]['always_on'] else ''}/{n}")
        if select in names: self.slist.selection_set(names.index(select)); self._sel_skill()
        self._update_hint()

    def _sel_skill(self, _=None):
        sel = self.slist.curselection()
        if not sel: return
        s = self.skills[list(self.skills)[sel[0]]]; self.sk_old = s["name"]
        self.sk_name.set(s["name"]); self.sk_desc.set(s["description"]); self.sk_on.set(s["always_on"])
        self.sk_body.delete("1.0", "end"); self.sk_body.insert("1.0", s["body"])

    def _new_skill(self):
        self.sk_old = None; self.sk_name.set("new-skill"); self.sk_desc.set(""); self.sk_on.set(False); self.sk_body.delete("1.0", "end")
        self.slist.selection_clear(0, "end")

    def _save_skill(self):
        body = self.sk_body.get("1.0", "end").strip()
        if not body: messagebox.showwarning("Skill", "Write some instructions first.", parent=self); return
        n = core.save_skill(self.sk_name.get(), self.sk_desc.get(), self.sk_on.get(), body, self.sk_old); self.sk_old = n
        self._refresh_skills(select=n); self.status.set(f"skill /{n} saved")

    def _del_skill(self):
        if self.sk_old and messagebox.askyesno("Skill", f"Delete /{self.sk_old}?", parent=self):
            core.delete_skill(self.sk_old); self._new_skill(); self._refresh_skills()

    def _use_skill(self):
        if not self.sk_old: return
        self.inp.insert("1.0", f"/{self.sk_old} "); self.nb.select(self.t_msg); self.inp.focus_set()


if __name__ == "__main__":
    App().mainloop()
