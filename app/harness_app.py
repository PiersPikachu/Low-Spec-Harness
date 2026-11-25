#!/usr/bin/env python3
"""harness_app.py - the Cube Harness window.

Tabs:  Messages  new and saved conversations, streaming replies, Stop, tool calls shown inline
       Config    model, execution mode, sampling, speculative draft, system prompt, vault folders; hardware plan
       Plugins   tools the model may call (plugins/*.py) and the harness's connections (engines, vaults, folders)
       Skills    reusable instruction snippets (skills/*.md): '/name' at the start of a message inserts one; 'always on' adds it to the system prompt
Start it with cube-harness.bat (no console window) or: python harness_app.py
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
        self.t_msg, self.t_art, self.t_mod, self.t_store, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(7))
        for f, name in ((self.t_msg, "Messages"), (self.t_art, "Artifacts"), (self.t_mod, "Models"), (self.t_store, "Model folders"), (self.t_cfg, "Config"),
                        (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")
        self.status = tk.StringVar(value="ready - no model loaded")
        ttk.Label(self, textvariable=self.status, style="Status.TLabel", anchor="w").pack(fill="x", side="bottom")
        self._build_messages(); self._build_artifacts(); self._build_models(); self._build_storage(); self._build_config(); self._build_plugins(); self._build_skills()
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
        self.kill_btn = ttk.Button(btns, text="Force stop", command=self.force_stop, state="disabled"); self.kill_btn.pack(fill="x", pady=(6, 0))
        self.forced = False; self._live_answer = ""; self.act = None
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
            if not vs: raise RuntimeError("no vault found - run cube-harness.bat or add a vault folder in Config")
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

    def force_stop(self):
        """kill the engine now; the reply so far is kept, and the model reloads with the next message"""
        if not self.busy: return
        self.forced = True; self.stop_flag = True
        if self.session: self.session.kill()
        self.status.set("force stop: engine killed")

    def _after_force(self):
        self.forced = False
        try: self._A().live_keep(); self._refresh_artifacts()
        except Exception: pass
        if self.session:
            try: self.session.close()
            except Exception: pass
        self.session = None; self.session_key = None; self.model_var.set("model stopped - it reloads with your next message")
        if self.conv is not None and (self._live_answer or self._think_live or self.act):
            m = {"role": "assistant", "content": self._live_answer.strip() + "\n[force-stopped]", "meta": {"stats": "force-stopped"}}
            self.conv["messages"].append(m); core.save_conversation(self.conv)
        if self._think_live: self._collapse_live_think()
        self._write("\n[force-stopped - the engine was killed; it reloads with the next message]\n", "info")
        self.status.set("force-stopped"); self._on_idle(None)

    def stop(self):
        self.stop_flag = True; self.status.set("stopping after the current step ...")

    def _begin(self):
        self.busy = True; self.stop_flag = False
        for b in (self.send_btn, self.load_btn): b.configure(state="disabled")
        self.stop_btn.configure(state="normal"); self.kill_btn.configure(state="normal"); self.status.set("working ...")

    def _on_idle(self, _):
        self.busy = False
        for b in (self.send_btn, self.load_btn): b.configure(state="normal")
        self.stop_btn.configure(state="disabled"); self.kill_btn.configure(state="disabled")

    def _on_info(self, t): self._write(t, "info")
    def _on_loaded(self, t): self.model_var.set(t.split("  (")[0]); self.status.set(t); self.cfg_plan.delete("1.0", "end"); self.cfg_plan.insert("1.0", self.session.banner)
    def _on_start(self, name):
        self._write(f"{name}\n", "asst_h"); self._live_answer = ""
        fr = tk.Frame(self.tx, bg=FIELD); tree = ttk.Treeview(fr, columns=("status",), show="tree", height=1, selectmode="browse")
        tree.column("#0", width=560, stretch=True); tree.column("status", width=380, stretch=True); tree.pack(fill="x")
        tree.bind("<Double-1>", lambda e, t=tree: self._act_detail(t))
        self.tx.configure(state="normal"); self.tx.window_create("end", window=fr); self.tx.insert("end", "\n"); self.tx.configure(state="disabled")
        self.act = {"tree": tree, "round": None, "think": None, "call": None, "text": {}, "think_txt": ""}

    def _act_grow(self):
        t = self.act["tree"]; n = 0; stack = list(t.get_children())
        while stack:
            i = stack.pop(); n += 1
            if t.item(i, "open"): stack += list(t.get_children(i))
        t.configure(height=max(1, min(14, n)))

    def _act_node(self, parent, label, status="", detail=None):
        t = self.act["tree"]; iid = t.insert(parent, "end", text=label, values=(status,), open=True)
        if detail is not None: self.act["text"][iid] = detail
        self._act_grow(); self.tx.see("end"); return iid

    def _act_detail(self, tree):
        sel = tree.selection()
        if not sel or not self.act: return
        txt = self.act["text"].get(sel[0]) if self.act.get("tree") is tree else None
        if not txt: return
        w = tk.Toplevel(self); w.title(tree.item(sel[0], "text")[:80]); w.geometry("820x520"); w.configure(bg=BG)
        t = self._text(w, font=MONO); t.pack(fill="both", expand=True); t.insert("1.0", txt); t.configure(state="disabled")

    def _on_act(self, ev):
        if not self.act: return
        t = self.act["tree"]; k = ev.get("ev")
        if k == "round":
            self.act["round"] = self._act_node("", f"Round {ev['n']}", "working ..."); self.act["think"] = None; self.act["call"] = None; self.act["think_txt"] = ""
            if ev["n"] > 1: t.item(t.get_children()[-2], values=("done",))
            return
        r = self.act["round"] or self._act_node("", "Round 1", "working ...")
        if k == "call_stream":
            where = (ev.get("target") or "") + ("/" + ev["path"] if ev.get("path") else "")
            label = f"\u2699 {ev.get('name')} \u00b7 {ev.get('action') or '...'} \u00b7 {where}".rstrip(" \u00b7")
            st = f"writing ... {len(ev['text'])} chars" if ev.get("text") is not None else f"composing call ... {ev.get('chars', 0)} chars"
            if self.act["call"] is None: self.act["call"] = self._act_node(r, label, st)
            else: t.item(self.act["call"], text=label, values=(st,))
            if ev.get("text") is not None: self.act["text"][self.act["call"]] = ev["text"]
            if ev.get("name") == "artifact": self._art_live(ev)
        elif k == "call":
            a = ev.get("args", {}); where = (a.get("name") or "") + ("/" + a["path"] if a.get("path") else "")
            label = f"\u2699 {ev['name']} \u00b7 {a.get('action', '')} \u00b7 {where}".rstrip(" \u00b7")
            if self.act["call"] is None: self.act["call"] = self._act_node(r, label, "running ...")
            else: t.item(self.act["call"], text=label, values=("running ...",))
            self.act["text"][self.act["call"]] = json.dumps(a, indent=1, ensure_ascii=False)
        elif k == "result":
            c = self.act["call"] or self._act_node(r, f"\u2699 {ev['name']}")
            res = str(ev.get("text", "")); ok = not res.startswith("error")
            t.item(c, values=(("\u2713 " if ok else "\u2717 ") + " ".join(res.split())[:70],))
            self.act["text"][c] = self.act["text"].get(c, "") + "\n\n--- result ---\n" + res; self.act["call"] = None; self.act["think"] = None
            if ev["name"] == "harness_dev": self._dev_changed()

    def _act_think(self, t):
        if not self.act: return
        r = self.act["round"] or self._act_node("", "Round 1", "working ...")
        self.act["think_txt"] += t; txt = self.act["think_txt"]; words = len(txt.split())
        if self.act["think"] is None: self.act["think"] = self._act_node(r, "\u25cb thinking", "")
        self.act["tree"].item(self.act["think"], text=f"\u25cb thinking ({words} words)", values=(" ".join(txt.split())[-70:],)); self.act["text"][self.act["think"]] = txt
    def _on_thinking(self, t):
        self._think_live += t; self._write(t, ("think", "think_live")); self._act_think(t)
    def _on_assistant(self, t):
        self._live_answer += t
        if self.act and self.act["round"]: self.act["think"] = None
        if self._think_live: self._collapse_live_think(); self.tx.configure(state="normal"); self.tx.tag_remove("think_live", "1.0", "end"); self.tx.configure(state="disabled")
        self._write(t)
    def _on_assistant_next(self, _): self._write(f"{self.session.vault.name}\n", "asst_h")
    def _on_tool(self, t):
        if self._think_live: self._collapse_live_think()
        self._write("\n" + t, "tool")
        if t.startswith("[artifact]"): self._refresh_artifacts()

    def _on_done(self, val):
        new, stats = val; self.conv["messages"] += new; core.save_conversation(self.conv)
        if self.forced: self._after_force(); return   # a force stop landed between rounds
        if self.act and self.act["tree"].get_children(): self.act["tree"].item(self.act["tree"].get_children()[-1], values=("done",))
        if self._think_live: self._collapse_live_think()
        self.tx.configure(state="normal"); self.tx.tag_remove("think_live", "1.0", "end"); self.tx.configure(state="disabled")
        self._write("\n" + stats + "\n", "info"); self.status.set(f"{self.session.how}  |  last reply: {stats}"); self._on_idle(None)

    def _on_error(self, tb):
        if self.forced: self._after_force(); return
        self._think_live = ""; self._write("\n" + tb.strip().splitlines()[-1] + "\n", "info"); self.status.set("error - see logs\\harness_error.txt")
        (hp.LOGS / "harness_error.txt").write_text(tb, encoding="utf-8"); self._on_idle(None)

    # ================================================================== Artifacts
    def _A(self):
        m = self.plugins.get("artifact")
        if not m or not getattr(m, "run", None): raise RuntimeError("plugins\\artifact.py is missing or failed to load")
        return m

    def _build_artifacts(self):
        left = ttk.Frame(self.t_art, style="Panel.TFrame", width=270); left.pack(side="left", fill="y"); left.pack_propagate(False)
        ttk.Button(left, text="+  New artifact", style="Accent.TButton", command=self._art_new).pack(fill="x", padx=10, pady=(10, 6))
        self.art_list = tk.Listbox(left, bg=PANEL, fg=FG, selectbackground="#404552", selectforeground=FG, activestyle="none", borderwidth=0, highlightthickness=0, font=FONT)
        self.art_list.pack(fill="both", expand=True, padx=6); self.art_list.bind("<<ListboxSelect>>", self._art_select)
        row = ttk.Frame(left, style="Panel.TFrame"); row.pack(fill="x", padx=10, pady=8)
        ttk.Button(row, text="Refresh", command=self._refresh_artifacts).pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(row, text="Delete", command=self._art_delete).pack(side="left", expand=True, fill="x")
        right = ttk.Frame(self.t_art); right.pack(side="left", fill="both", expand=True, padx=(8, 8))
        head = ttk.Frame(right); head.pack(fill="x", pady=(8, 6))
        self.art_head = tk.StringVar(value="no artifact selected - the model creates them with the 'artifact' tool, or press New")
        ttk.Label(head, textvariable=self.art_head, style="Head.TLabel").pack(side="left")
        bar = ttk.Frame(right); bar.pack(fill="x", pady=(0, 6))
        for txt, cmd in (("Open in browser", self._art_browser), ("Open folder", self._art_folder), ("Export .zip", self._art_zip), ("New file...", self._art_newfile)):
            ttk.Button(bar, text=txt, command=cmd).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Save file", style="Accent.TButton", command=self._art_save).pack(side="right")
        self.art_auto = tk.BooleanVar(value=True); self._as_job = None; self._as_session = None
        ttk.Checkbutton(bar, text="Autosave", variable=self.art_auto).pack(side="right", padx=4)
        self.art_follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Follow the model live", variable=self.art_follow).pack(side="right", padx=10)
        mid = ttk.Frame(right); mid.pack(fill="both", expand=True)
        fl = ttk.Frame(mid, width=230); fl.pack(side="left", fill="y"); fl.pack_propagate(False)
        ttk.Label(fl, text="Files", style="Dim.TLabel").pack(anchor="w")
        self.art_files = tk.Listbox(fl, bg=FIELD, fg=FG, selectbackground="#404552", activestyle="none", borderwidth=0, highlightthickness=0, font=MONO, exportselection=False)
        self.art_files.pack(fill="both", expand=True); self.art_files.bind("<<ListboxSelect>>", self._art_open_file)
        ttk.Label(fl, text="History (newest first)", style="Dim.TLabel").pack(anchor="w", pady=(8, 0))
        self.art_hist = tk.Listbox(fl, height=9, bg=FIELD, fg=FG, selectbackground="#404552", activestyle="none", borderwidth=0, highlightthickness=0, font=("Consolas", 9), exportselection=False)
        self.art_hist.pack(fill="x"); ttk.Button(fl, text="Revert selected change", command=self._art_revert).pack(fill="x", pady=(4, 0))
        ed = ttk.Frame(mid); ed.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.art_file_var = tk.StringVar(value=""); ttk.Label(ed, textvariable=self.art_file_var, style="Dim.TLabel").pack(anchor="w")
        self.art_text = self._text(ed, font=MONO, undo=True); self.art_text.configure(wrap="none"); self.art_text.pack(fill="both", expand=True)
        self.art_text.bind("<KeyRelease>", self._art_typed); self.art_text.bind("<<Paste>>", lambda e: self.after(10, self._art_typed))
        self.art_name = None; self.art_path = None; self._refresh_artifacts()

    def _refresh_artifacts(self, select=None):
        try: A = self._A()
        except Exception as ex: self.art_head.set(str(ex)); return
        A.ROOT.mkdir(exist_ok=True)
        self.art_names = [d.name for d in sorted(A.ROOT.iterdir(), key=lambda d: -(d / ".artifact.json").stat().st_mtime if (d / ".artifact.json").exists() else 0)
                          if (d / ".artifact.json").exists()]
        self.art_list.delete(0, "end")
        for n in self.art_names:
            m = A._meta(A.ROOT / n); self.art_list.insert("end", f"  {m.get('title') or n}  [{m.get('type')}]")
        sel = select or self.art_name
        if sel in self.art_names: i = self.art_names.index(sel); self.art_list.selection_set(i); self._art_show(sel, keep_file=True)

    def _art_select(self, _=None):
        sel = self.art_list.curselection()
        if sel: self._art_show(self.art_names[sel[0]])

    def _art_show(self, name, keep_file=False):
        A = self._A(); d = A.ROOT / name; m = A._meta(d); self.art_name = name
        self.art_head.set(f"{m.get('title') or name}   [{m.get('type')}]   artifacts\\{name}   updated {m.get('updated', '')}")
        files = A._files(d); self.art_files.delete(0, "end")
        for f in files: self.art_files.insert("end", f)
        self.art_hist.delete(0, "end")
        for c in reversed(m.get("changes", [])[-200:]): self.art_hist.insert("end", f"#{c['n']} {c['time'][5:16]} {c['action']} {c['path']}")
        target = self.art_path if keep_file and self.art_path in files else (m.get("main") if m.get("main") in files else (files[0] if files else None))
        if target: self.art_files.selection_clear(0, "end"); self.art_files.selection_set(files.index(target)); self._art_load(target)
        else: self.art_path = None; self.art_text.delete("1.0", "end"); self.art_file_var.set("(no files)")

    def _art_open_file(self, _=None):
        sel = self.art_files.curselection()
        if sel and self.art_name: self._art_load(self.art_files.get(sel[0]))

    def _art_load(self, rel):
        if self._as_job: self.after_cancel(self._as_job); self._art_autosave()
        self._as_session = None
        A = self._A(); p = A.ROOT / self.art_name / rel; self.art_path = rel
        try: body = p.read_text(encoding="utf-8")
        except Exception as ex: body = f"(cannot show this file as text: {ex})"
        self.art_text.delete("1.0", "end"); self.art_text.insert("1.0", body); self.art_text.edit_reset()
        self.art_file_var.set(f"{rel}   {p.stat().st_size if p.exists() else 0} bytes   - edit here and press Save file (versioned, revertible)")

    def _art_live(self, ev):
        """show an artifact file while the model is still writing it (the call's content as decoded so far)"""
        if not getattr(self, "art_follow", None) or not self.art_follow.get() or ev.get("text") is None: return
        if self._as_job: self.after_cancel(self._as_job); self._art_autosave()      # save the user's typing before the live view takes over
        A = self._A(); name = A._slug(ev.get("target") or ""); rel = (ev.get("path") or "").replace("\\", "/").strip("/")
        if name and (A.ROOT / name / ".artifact.json").exists():
            if not rel: rel = A._meta(A.ROOT / name).get("main", "")
            self.art_name = name; self.art_path = rel or self.art_path
        verb = {"create": "creating", "write": "writing", "append": "appending to", "edit": "editing (new text)"}.get(ev.get("action"), ev.get("action") or "writing")
        self.art_head.set(f"\u25cf LIVE  {name or '(new artifact)'}{'/' + rel if rel else ''}  -  model is {verb}  ({len(ev['text'])} chars)")
        self.art_file_var.set("live view of the model's tool call - the file is saved when the call completes")
        self.art_text.delete("1.0", "end"); self.art_text.insert("1.0", ev["text"]); self.art_text.see("end")

    def _art_typed(self, _=None):
        """typing in the editor: save 1.5 s after the last key (Autosave on)"""
        if not (self.art_auto.get() and self.art_name and self.art_path): return
        if self._as_job: self.after_cancel(self._as_job)
        self._as_job = self.after(1500, self._art_autosave)

    def _art_autosave(self):
        self._as_job = None
        if not (self.art_name and self.art_path): return
        key = (self.art_name, self.art_path)
        if not self._as_session or self._as_session[0] != key: self._as_session = (key, f"{time.time():.0f}")
        try: self.status.set(self._A().autosave(self.art_name, self.art_path, self.art_text.get("1.0", "end-1c"), self._as_session[1]))
        except Exception as ex: self.status.set(f"autosave failed: {ex}")

    def _art_save(self):
        if not (self.art_name and self.art_path): return
        out = self._A().run("write", name=self.art_name, path=self.art_path, content=self.art_text.get("1.0", "end-1c"))
        self.status.set(out); self._refresh_artifacts()

    def _art_new(self):
        A = self._A(); name = simpledialog.askstring("New artifact", "Name (e.g. trip-notes):", parent=self)
        if not name: return
        t = simpledialog.askstring("New artifact", "Type: note, page, site, code or data", initialvalue="note", parent=self) or "note"
        out = A.run("create", name=name, type=t.strip().lower(), title=name); self.status.set(out)
        self._refresh_artifacts(select=A._slug(name))

    def _art_newfile(self):
        if not self.art_name: return
        rel = simpledialog.askstring("New file", "Path inside the artifact (e.g. css/style.css):", parent=self)
        if rel: self.status.set(self._A().run("write", name=self.art_name, path=rel, content="")); self.art_path = rel.replace("\\", "/").strip("/"); self._refresh_artifacts()

    def _art_delete(self):
        if not self.art_name: return
        if messagebox.askyesno("Delete artifact", f"Delete artifact '{self.art_name}' and all its files and history permanently?", parent=self):
            import shutil; shutil.rmtree(self._A().ROOT / self.art_name); self.art_name = None; self.art_path = None
            self.art_text.delete("1.0", "end"); self.art_head.set("artifact deleted"); self._refresh_artifacts()

    def _art_revert(self):
        sel = self.art_hist.curselection()
        if not (sel and self.art_name): return
        n = int(self.art_hist.get(sel[0]).split()[0][1:])
        if messagebox.askyesno("Revert", f"Restore the file as it was before change #{n}? (The current version stays in the history.)", parent=self):
            self.status.set(self._A().run("revert", name=self.art_name, version=n)); self._refresh_artifacts()

    def _art_browser(self):
        if not self.art_name: return
        import webbrowser; A = self._A(); d = A.ROOT / self.art_name; m = A._meta(d)
        rel = self.art_path if (self.art_path or "").lower().endswith((".html", ".htm")) else m.get("main")
        p = d / (rel or "")
        if not p.is_file(): messagebox.showinfo("Open in browser", "This artifact has no HTML file to open.", parent=self); return
        webbrowser.open(p.resolve().as_uri()); self.status.set(f"opened {p.name} in the browser")

    def _art_folder(self):
        if self.art_name: os.startfile(str(self._A().ROOT / self.art_name))

    def _art_zip(self):
        if not self.art_name: return
        dst = filedialog.asksaveasfilename(parent=self, defaultextension=".zip", initialfile=f"{self.art_name}.zip", filetypes=[("zip", "*.zip")])
        if not dst: return
        import zipfile; d = self._A().ROOT / self.art_name
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
            for f in self._A()._files(d): z.write(d / f, f)
        self.status.set(f"exported {self.art_name} to {dst} (current files, no history)")

    # ================================================================== Models (Hugging Face -> compressed vault)
    def _build_models(self):
        top = ttk.Frame(self.t_mod); top.pack(fill="x", padx=12, pady=(10, 6))
        ttk.Label(top, text="Search Hugging Face:").pack(side="left")
        self.hub_q = tk.StringVar(value="qwen3"); e = ttk.Entry(top, textvariable=self.hub_q, width=40); e.pack(side="left", padx=6); e.bind("<Return>", lambda _: self._hub_search())
        ttk.Button(top, text="Search", style="Accent.TButton", command=self._hub_search).pack(side="left")
        self.hub_fit = tk.BooleanVar(value=False); ttk.Checkbutton(top, text="Only models that run well here", variable=self.hub_fit, command=self._hub_filter).pack(side="left", padx=12)
        self.hub_hw = tk.StringVar(value=""); ttk.Label(top, textvariable=self.hub_hw, style="Dim.TLabel").pack(side="right")
        body = ttk.Frame(self.t_mod); body.pack(fill="both", expand=True, padx=12)
        left = ttk.Frame(body); left.pack(side="left", fill="both", expand=True)
        self.hub_res = ttk.Treeview(left, columns=("params", "vault", "speed", "verdict", "dl"), show="tree headings", height=14)
        self.hub_res.heading("#0", text="Model"); self.hub_res.column("#0", width=330)
        for c, w, t in (("params", 70, "Params"), ("vault", 90, "Vault"), ("speed", 90, "Est. tok/s"), ("verdict", 210, "Verdict"), ("dl", 90, "Downloads")):
            self.hub_res.heading(c, text=t); self.hub_res.column(c, width=w, anchor="w")
        self.hub_res.pack(fill="both", expand=True); self.hub_res.bind("<<TreeviewSelect>>", self._hub_select)
        ttk.Label(left, text="Installed vaults", style="Dim.TLabel").pack(anchor="w", pady=(8, 0))
        self.hub_inst = ttk.Treeview(left, columns=("size", "kind", "tuned"), show="tree headings", height=6)
        self.hub_inst.heading("#0", text="Vault"); self.hub_inst.column("#0", width=330)
        for c, w, t in (("size", 90, "Size"), ("kind", 120, "Type"), ("tuned", 380, "Tuned")): self.hub_inst.heading(c, text=t); self.hub_inst.column(c, width=w, anchor="w")
        self.hub_inst.pack(fill="x")
        ib = ttk.Frame(left); ib.pack(fill="x", pady=4)
        ttk.Button(ib, text="Tune selected", command=self._hub_tune).pack(side="left")
        ttk.Button(ib, text="Use in chat", style="Accent.TButton", command=self._hub_use).pack(side="left", padx=6)
        ttk.Button(ib, text="Refresh", command=self._hub_installed).pack(side="left")
        right = ttk.Frame(body, width=520); right.pack(side="left", fill="both", padx=(10, 0)); right.pack_propagate(False)
        ttk.Label(right, text="Decision tree for the selected model", style="Head.TLabel").pack(anchor="w")
        self.hub_tree = ttk.Treeview(right, columns=("ans",), show="tree", height=12); self.hub_tree.column("#0", width=170); self.hub_tree.column("ans", width=340)
        self.hub_tree.pack(fill="x"); self.hub_tree.bind("<Double-1>", self._hub_tree_detail)
        rb = ttk.Frame(right); rb.pack(fill="x", pady=6)
        self.hub_install_btn = ttk.Button(rb, text="Install: download, compress, tune", style="Accent.TButton", command=self._hub_install); self.hub_install_btn.pack(side="left")
        ttk.Button(rb, text="Open on Hugging Face", command=lambda: self._hub_sel and __import__("webbrowser").open(f"https://huggingface.co/{self._hub_sel}")).pack(side="left", padx=6)
        self.hub_stop_btn = ttk.Button(rb, text="Stop", command=lambda: setattr(self, "_hub_stop", True), state="disabled"); self.hub_stop_btn.pack(side="left")
        self.hub_log = self._text(right, font=MONO); self.hub_log.pack(fill="both", expand=True)
        self._hub_sel = None; self._hub_rows = {}; self._hub_stop = False; self._hub_busy = False
        self._hub_installed()

    def _hub_logw(self, t): self.q.put(("hub", lambda: (self.hub_log.insert("end", t.rstrip() + "\n"), self.hub_log.see("end"))))
    def _on_hub(self, fn): fn()

    def _hub_search(self):
        if self._hub_busy: return
        q = self.hub_q.get().strip(); self.hub_res.delete(*self.hub_res.get_children()); self._hub_rows = {}; self._hub_order = []; self._hub_busy = True
        self.status.set(f"searching Hugging Face for '{q}' ...")
        def work():
            try:
                import model_hub as mh
                hw = mh.hardware()
                self.q.put(("hub", lambda: self.hub_hw.set(f"this machine: {hw['ram_w'] / 1e9:.0f} GB RAM for weights, {hw['vram'] / 1e9:.1f} GB VRAM, "
                                                             f"vaults {hw['vault_free'] / 1e9:.0f} GB free, downloads {hw['dl_free'] / 1e9:.0f} GB free")))
                res = mh.search(q, 30)
                def fill():
                    for m in res:
                        p = f"{m['params'] / 1e9:.1f}B" if m["params"] else "?"
                        self.hub_res.insert("", "end", iid=m["repo"], text=m["repo"], values=(p, "", "", "checking ...", f"{m['downloads']:,}"))
                    self.status.set(f"{len(res)} models - checking each against this machine ...")
                self.q.put(("hub", fill))
                for m in res:                       # assess each in turn (needs its config.json)
                    a = mh.assess(m["repo"], hw); self._hub_rows[m["repo"]] = a
                    def upd(a=a):
                        if self.hub_res.exists(a["repo"]):
                            self.hub_res.item(a["repo"], values=(self.hub_res.item(a["repo"], "values")[0], f"{a['vault_gb']:.1f} GB" if a["vault_gb"] else "",
                                                                 f"{a['est_tok_s']:.2f}" if a["est_tok_s"] else "", a["verdict"], self.hub_res.item(a["repo"], "values")[4]))
                            self._hub_filter()
                    self.q.put(("hub", upd))
                self.q.put(("hub", lambda: self.status.set("search done - select a model to see its decision tree")))
            except Exception as ex:
                self.q.put(("hub", lambda ex=ex: self.status.set(f"search failed: {ex}")))
            finally: self._hub_busy = False
        threading.Thread(target=work, daemon=True).start()

    def _hub_filter(self):
        if not self.hub_fit.get():
            for r in self._hub_rows: pass
            return
        for iid in list(self.hub_res.get_children()):
            a = self._hub_rows.get(iid)
            if a and not a["verdict"].startswith("runs well"): self.hub_res.detach(iid)

    def _hub_select(self, _=None):
        sel = self.hub_res.selection()
        if not sel: return
        self._hub_sel = sel[0]; a = self._hub_rows.get(sel[0]); self.hub_tree.delete(*self.hub_tree.get_children())
        if not a: self.hub_tree.insert("", "end", text="checking ...", values=("",)); return
        icon = {"ok": "✓", "warn": "!", "bad": "✗"}
        root = self.hub_tree.insert("", "end", text=a["repo"].split("/")[-1], values=(a["verdict"],), open=True)
        for q, ans, st in a["tree"]: self.hub_tree.insert(root, "end", text=f"{icon.get(st, '?')} {q}", values=(ans,))
        self.hub_install_btn.configure(state="normal" if a.get("kind") else "disabled")

    def _hub_tree_detail(self, _=None):
        sel = self.hub_tree.selection()
        if sel: messagebox.showinfo(self.hub_tree.item(sel[0], "text"), self.hub_tree.item(sel[0], "values")[0], parent=self)

    def _hub_installed(self):
        import json as _j
        t = self.hub_inst; t.delete(*t.get_children())
        for name, path in sorted(hp.find_vaults().items()):
            try:
                kv = {l.split()[0]: l.split()[1] for l in open(path / "vault.idx") if len(l.split()) == 2}
                kind = "gpt-oss (MoE)" if kv.get("gptoss") == "1" else ("MoE" if kv.get("moe") == "1" else "dense")
                size = (path / "vault.bin").stat().st_size / 1e9
                tj = path / "tuned.json"; tuned = ""
                if tj.exists(): d = _j.load(open(tj)); tuned = f"{d['tok_s']} tok/s  {d['settings']}  ({d['date']})"
                t.insert("", "end", iid=str(path), text=name, values=(f"{size:.1f} GB", kind, tuned or "not tuned yet"))
            except Exception: pass

    def _hub_busy_ui(self, on):
        self._hub_busy = on; self._hub_stop = False
        self.hub_install_btn.configure(state="disabled" if on else "normal"); self.hub_stop_btn.configure(state="normal" if on else "disabled")

    def _hub_release_chat(self):
        """the tuner and the packer need the RAM and GPU: close the chat model first"""
        if self.busy: messagebox.showinfo("Models", "Wait for the current reply to finish (or Force stop it) first.", parent=self); return False
        if self.session:
            try: self.session.close()
            except Exception: pass
            self.session = None; self.session_key = None; self.model_var.set("model unloaded (Models tab is using the machine)")
        return True

    def _hub_install(self):
        a = self._hub_rows.get(self._hub_sel)
        if not a or not a.get("kind") or self._hub_busy: return
        import model_hub as mh
        hw = mh.hardware()
        msg = (f"Install {a['repo']}?\n\n1. Download ~{a['download_gb']:.1f} GB from huggingface.co to {hw['dl_dir']}\n2. Compress it into a ~{a['vault_gb']:.1f} GB vault in {hw['vault_dir']}\n"
               f"3. Check every stream of the vault\n4. Tune its engine settings on this machine (a few short test replies)\n\nVerdict: {a['verdict']}"
               + ("\n\nWARNING: the disk check failed - see the decision tree." if "disk" in a["verdict"] else ""))
        if not messagebox.askyesno("Install model", msg, parent=self) or not self._hub_release_chat(): return
        self._hub_busy_ui(True); repo = a["repo"]
        def work():
            try: mh.install(repo, self._hub_logw, True, lambda: self._hub_stop); self._hub_logw("done")
            except Exception as ex: self._hub_logw(f"stopped: {ex}")
            finally: self.q.put(("hub", lambda: (self._hub_busy_ui(False), self._hub_installed())))
        threading.Thread(target=work, daemon=True).start()

    def _hub_tune(self):
        sel = self.hub_inst.selection()
        if not sel or self._hub_busy or not self._hub_release_chat(): return
        self._hub_busy_ui(True); v = Path(sel[0]); self._hub_logw(f"tuning {v.name} (short test replies with each candidate setting) ...")
        def work():
            try: import model_hub as mh; mh.tune(v, self._hub_logw)
            except Exception as ex: self._hub_logw(f"tuning failed: {ex}")
            finally: self.q.put(("hub", lambda: (self._hub_busy_ui(False), self._hub_installed())))
        threading.Thread(target=work, daemon=True).start()

    def _hub_use(self):
        sel = self.hub_inst.selection()
        if not sel: return
        name = self.hub_inst.item(sel[0], "text"); self.vars["vault"].set(name); self._model_defaults()
        self.save_config(); self.nb.select(self.t_msg); self.status.set(f"{name} selected with its recommended / tuned settings - it loads with your next message")

    # ================================================================== Model folders
    def _build_storage(self):
        top = ttk.Frame(self.t_store); top.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Label(top, text="Folders that hold compressed models", style="Head.TLabel").pack(side="left")
        ttk.Label(top, text="SSD folders run models at full speed; HDD / USB folders are fine for keeping models - move one to an SSD folder before chatting with it.",
                  style="Dim.TLabel").pack(side="left", padx=12)
        self.st_tree = ttk.Treeview(self.t_store, columns=("drive", "kind", "speed", "free", "vaults", "role"), show="tree headings", height=9)
        self.st_tree.heading("#0", text="Folder"); self.st_tree.column("#0", width=420)
        for c, w, t in (("drive", 60, "Drive"), ("kind", 230, "Disk"), ("speed", 110, "Read speed"), ("free", 90, "Free"), ("vaults", 170, "Models"), ("role", 150, "Role")):
            self.st_tree.heading(c, text=t); self.st_tree.column(c, width=w, anchor="w")
        self.st_tree.pack(fill="x", padx=12)
        bar = ttk.Frame(self.t_store); bar.pack(fill="x", padx=12, pady=6)
        for txt, cmd in (("Add folder...", self._st_add), ("Remove from list", self._st_remove), ("Use for new installs", self._st_default), ("Open", self._st_open),
                         ("Measure read speed", self._st_measure), ("Refresh", self._st_refresh)):
            ttk.Button(bar, text=txt, command=cmd).pack(side="left", padx=(0, 6))
        mv = ttk.LabelFrame(self.t_store, text="  Move a model to the selected folder (copy, verify every stream, then delete the original)  "); mv.pack(fill="x", padx=12, pady=6)
        self.st_vault = tk.StringVar(); self.st_vcombo = ttk.Combobox(mv, textvariable=self.st_vault, width=60, state="readonly"); self.st_vcombo.pack(side="left", padx=8, pady=8)
        self.st_move_btn = ttk.Button(mv, text="Move here", style="Accent.TButton", command=self._st_move); self.st_move_btn.pack(side="left")
        dl = ttk.LabelFrame(self.t_store, text="  Download folder for new models (the original weights; can be deleted after install)  "); dl.pack(fill="x", padx=12, pady=6)
        self.st_dl = tk.StringVar(value=self.cfg.get("hub_download_dir") or ""); ttk.Entry(dl, textvariable=self.st_dl, width=70).pack(side="left", padx=8, pady=8)
        ttk.Button(dl, text="Browse...", command=self._st_dl_browse).pack(side="left"); ttk.Button(dl, text="Save", command=self._st_dl_save).pack(side="left", padx=6)
        self.st_log = self._text(self.t_store, font=MONO, height=10); self.st_log.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self._st_refresh()

    def _st_logw(self, t): self.q.put(("hub", lambda: (self.st_log.insert("end", t.rstrip() + "\n"), self.st_log.see("end"))))

    def _st_refresh(self):
        import storage
        self.cfg = hp.load_config(); t = self.st_tree; t.delete(*t.get_children())
        for o in hp.model_dirs(self.cfg):
            p = o["path"]; sm = storage.folder_summary(p)
            if sm["exists"]:
                media, bus, model = storage.drive_kind(p); sp = storage.read_speed(p)
                kind = f"{media} {bus}".strip() + (f" - {model}" if model and model != "?" else "")
                speed = f"{sp:,.0f} MB/s" if sp else "not measured"
                slow = (sp is not None and sp < 500) or media == "HDD" or bus == "USB"
                vals = (p.drive, kind + ("  (slow: storage)" if slow else ""), speed, f"{sm['free'] / 1e9:,.0f} GB", f"{sm['vaults']} ({sm['size'] / 1e9:,.1f} GB)",
                        ("install target" if o["default"] else "") + (", built-in" if o["builtin"] else ""))
            else: vals = (p.drive, "NOT FOUND (drive disconnected?)", "", "", "", "install target" if o["default"] else "")
            t.insert("", "end", iid=str(p), text=str(p), values=vals)
        self.st_vcombo["values"] = [f"{n}   ({hp.rel(v.parent)})" for n, v in sorted(hp.find_vaults(self.cfg).items())]
        try: self._hub_installed()
        except Exception: pass

    def _st_sel(self):
        sel = self.st_tree.selection(); return Path(sel[0]) if sel else None

    def _st_add(self):
        d = filedialog.askdirectory(parent=self, title="Folder for compressed models (any drive)")
        if d: hp.add_model_dir(d); self._st_refresh(); self.status.set(f"model folder added: {d}")

    def _st_remove(self):
        p = self._st_sel()
        if not p: return
        if p == hp.VAULTS: messagebox.showinfo("Model folders", "The built-in folder data\\vaults cannot be removed.", parent=self); return
        if messagebox.askyesno("Model folders", f"Remove {p} from the list?\n\nNothing is deleted: its models just stop showing in the harness.", parent=self):
            cfg = hp.load_config(); cfg["model_dirs"] = [e for e in cfg["model_dirs"] if hp.absolute(e["path"]) != p]; hp.save_config(cfg); self._st_refresh()

    def _st_default(self):
        p = self._st_sel()
        if p: hp.add_model_dir(p, default=True); self._st_refresh(); self.status.set(f"new installs go to {p}")

    def _st_open(self):
        p = self._st_sel()
        if p and p.exists(): os.startfile(str(p))

    def _st_measure(self):
        p = self._st_sel()
        if not p: return
        def work():
            try:
                import storage; sp = storage.read_speed(p, measure=True)
                self._st_logw(f"{p}: {sp:,.0f} MB/s" if sp else f"{p}: no model in it to measure with")
            except Exception as ex: self._st_logw(f"measure failed: {ex}")
            self.q.put(("hub", self._st_refresh))
        threading.Thread(target=work, daemon=True).start(); self._st_logw(f"measuring {p} (reads a sample of its largest model) ...")

    def _st_move(self):
        p = self._st_sel(); v = self.st_vault.get().split("   ")[0].strip()
        if not (p and v): messagebox.showinfo("Move", "Select a destination folder in the list and a model in the box.", parent=self); return
        src = hp.find_vaults().get(v)
        if src is None or src.parent == p: return
        if self.session and Path(self.session.vault) == src:
            if self.busy: messagebox.showinfo("Move", "Wait for the reply to finish first.", parent=self); return
            self.session.close(); self.session = None; self.session_key = None; self.model_var.set("model unloaded (being moved)")
        if not messagebox.askyesno("Move model", f"Move {v}\nfrom {src.parent}\nto {p}?\n\nIt is copied, every stream of the copy is verified, then the original is deleted.", parent=self): return
        self.st_move_btn.configure(state="disabled")
        def work():
            try: import storage; storage.move_vault(src, p, self._st_logw)
            except Exception as ex: self._st_logw(f"move stopped: {ex}")
            self.q.put(("hub", lambda: (self.st_move_btn.configure(state="normal"), self._st_refresh())))
        threading.Thread(target=work, daemon=True).start()

    def _st_dl_browse(self):
        d = filedialog.askdirectory(parent=self, title="Download folder for new models")
        if d: self.st_dl.set(d); self._st_dl_save()

    def _st_dl_save(self):
        cfg = hp.load_config(); cfg["hub_download_dir"] = self.st_dl.get().strip() or None; hp.save_config(cfg); self.cfg = cfg; self.status.set("download folder saved")

    # ================================================================== Config
    FIELDS = [  # key, label, kind, options
        ("vault", "Model vault", "vault", None), ("mode", "Execution mode", "combo", ["auto", "cpu", "gpu", "split"]),
        ("gpu_layers", "GPU layers (0 = planner decides)", "int", (0, 80)), ("ctx", "Context length (tokens)", "int", (256, 262144)),
        ("max_new", "Max reply tokens", "int", (16, 131072)), ("max_file_growth", "Max file growth per reply (chars, 0 = 4 x max reply)", "int", (0, 2000000)), ("temp", "Temperature", "float", None), ("top_p", "Top-p", "float", None),
        ("top_k", "Top-k (0 = off)", "int", (0, 128)), ("greedy", "Greedy (deterministic)", "bool", None), ("seed", "Seed (blank = random)", "str", None),
        ("draft", "Speculative draft", "draft", None), ("k", "Draft tokens per round (k)", "int", (1, 16)),
        ("think", "Let the model think first (Qwen3 / Hermes)", "bool", None), ("kv16", "KV cache in fp16 (half the memory, ~2e-4 rounding)", "bool", None), ("max_tool_rounds", "Max tool calls per reply", "int", (0, 16)),
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
            else: self.cfg[k] = v
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
            p = subprocess.run([sys.executable, str(hp.APP / "planner.py")] + args, capture_output=True, text=True, cwd=str(hp.ROOT),
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
        pend = ttk.LabelFrame(self.t_plug, text="  Written by the model - not loaded until you approve (plugins\\pending)  "); pend.pack(fill="both", expand=True, padx=14, pady=(6, 6))
        self.pend_tree = ttk.Treeview(pend, columns=("name", "size", "when", "desc"), show="headings", height=4)
        for c, w, t in (("name", 140, "Plugin"), ("size", 80, "Size"), ("when", 160, "Written"), ("desc", 640, "Description (from its source)")): self.pend_tree.heading(c, text=t); self.pend_tree.column(c, width=w, anchor="w")
        self.pend_tree.pack(fill="both", expand=True, padx=8, pady=8); self.pend_tree.bind("<Double-1>", lambda e: self._pend_view())
        pb = ttk.Frame(pend); pb.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(pb, text="View code", command=self._pend_view).pack(side="left")
        ttk.Button(pb, text="Approve and enable", style="Accent.TButton", command=self._pend_approve).pack(side="left", padx=6)
        ttk.Button(pb, text="Discard", command=self._pend_discard).pack(side="left")
        ttk.Label(pb, text="Plugins run as you, with full access to this computer: read the code before approving.", style="Dim.TLabel").pack(side="left", padx=10)
        bot = ttk.LabelFrame(self.t_plug, text="  Connections (all relative to the harness folder)  "); bot.pack(fill="both", expand=True, padx=14, pady=(6, 10))
        self.ctree = ttk.Treeview(bot, columns=("what", "where", "status"), show="headings", height=9)
        for c, w, t in (("what", 230, "Component"), ("where", 560, "Location"), ("status", 240, "Status")): self.ctree.heading(c, text=t); self.ctree.column(c, width=w, anchor="w")
        self.ctree.pack(fill="both", expand=True, padx=8, pady=8)
        bar2 = ttk.Frame(bot); bar2.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bar2, text="Refresh", command=self._refresh_connections).pack(side="left")
        ttk.Button(bar2, text="Open harness folder", command=lambda: os.startfile(hp.ROOT)).pack(side="left", padx=6)
        ttk.Button(bar2, text="Open logs", command=lambda: os.startfile(hp.LOGS)).pack(side="left")
        self._refresh_plugins(); self._refresh_connections(); self._refresh_pending()

    def _refresh_pending(self):
        import re as _re, datetime as _dt
        self.pend_tree.delete(*self.pend_tree.get_children()); d = hp.PLUGINS / "pending"; d.mkdir(exist_ok=True)
        for f in sorted(d.glob("*.py")):
            src = f.read_text(encoding="utf-8", errors="replace"); m = _re.search(r'^DESCRIPTION\s*=\s*\(?\s*["\'](.+?)["\']', src, _re.M)
            self.pend_tree.insert("", "end", iid=f.stem, values=(f.stem, f"{f.stat().st_size} B", _dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"), (m.group(1) if m else "")[:120]))

    def _pend_view(self):
        sel = self.pend_tree.selection()
        if not sel: return
        f = hp.PLUGINS / "pending" / f"{sel[0]}.py"; w = tk.Toplevel(self); w.title(f"{sel[0]}.py - model-written plugin (not loaded)"); w.geometry("900x640"); w.configure(bg=BG)
        t = self._text(w, font=MONO); t.pack(fill="both", expand=True); t.insert("1.0", f.read_text(encoding="utf-8", errors="replace")); t.configure(state="disabled")

    def _pend_approve(self):
        sel = self.pend_tree.selection()
        if not sel: return
        n = sel[0]; src = hp.PLUGINS / "pending" / f"{n}.py"; dst = hp.PLUGINS / f"{n}.py"
        if dst.exists(): messagebox.showerror("Approve", f"plugins\\{n}.py already exists.", parent=self); return
        if not messagebox.askyesno("Approve plugin", f"Load '{n}' as a tool the model can call?\n\nIt will run as Python with your permissions. "
                                   "Only approve code you have read and trust.", parent=self): return
        src.replace(dst); self._reload_plugins()
        if n in self.plugins and getattr(self.plugins[n], "run", None):
            on = list(self.cfg["enabled_plugins"]); on.append(n) if n not in on else None; self.cfg["enabled_plugins"] = on; hp.save_config(self.cfg)
            self._refresh_plugins(); self.status.set(f"plugin '{n}' approved and enabled")
        else:
            self.status.set(f"plugin '{n}' failed to load - see the Plugins list"); messagebox.showwarning("Approve", self.plugins.get(n).DESCRIPTION if n in self.plugins else "not loaded", parent=self)
        self._refresh_pending()

    def _pend_discard(self):
        sel = self.pend_tree.selection()
        if sel and messagebox.askyesno("Discard", f"Delete the pending plugin '{sel[0]}'?", parent=self):
            (hp.PLUGINS / "pending" / f"{sel[0]}.py").unlink(missing_ok=True); self._refresh_pending()

    def _dev_changed(self):
        """the model changed skills or plugins: refresh those views"""
        try: self._refresh_skills(); self._refresh_pending()
        except Exception: pass

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
        t = self.ctree; t.delete(*t.get_children()); ok = lambda b: "found" if b else "MISSING - run cube-harness.bat"
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
        for n in names: self.slist.insert("end", f"  {'* ' if self.skills[n]['always_on'] else ''}/{n}{'   [by model]' if self.skills[n].get('author') == 'model' else ''}{'   [auto when tools are on]' if self.skills[n].get('applies_when') == 'tools' else ''}")
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
