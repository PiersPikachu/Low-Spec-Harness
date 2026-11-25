"""make_app_live.py - harness_app.py: Force stop, live activity tree, live artifact view, review queue for model-written plugins."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_app.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

# ---------------------------------------------------------------- Force stop
rep('''        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled"); self.stop_btn.pack(fill="x", pady=(6, 0))''',
'''        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled"); self.stop_btn.pack(fill="x", pady=(6, 0))
        self.kill_btn = ttk.Button(btns, text="Force stop", command=self.force_stop, state="disabled"); self.kill_btn.pack(fill="x", pady=(6, 0))
        self.forced = False; self._live_answer = ""; self.act = None''')
rep('''        self.stop_btn.configure(state="normal"); self.status.set("working ...")''',
    '''        self.stop_btn.configure(state="normal"); self.kill_btn.configure(state="normal"); self.status.set("working ...")''')
rep('''        for b in (self.send_btn, self.load_btn): b.configure(state="normal")
        self.stop_btn.configure(state="disabled")''',
'''        for b in (self.send_btn, self.load_btn): b.configure(state="normal")
        self.stop_btn.configure(state="disabled"); self.kill_btn.configure(state="disabled")''')
rep('''    def stop(self):''', '''    def force_stop(self):
        """kill the engine now; the reply so far is kept, and the model reloads with the next message"""
        if not self.busy: return
        self.forced = True; self.stop_flag = True
        if self.session: self.session.kill()
        self.status.set("force stop: engine killed")

    def _after_force(self):
        self.forced = False
        if self.session:
            try: self.session.close()
            except Exception: pass
        self.session = None; self.session_key = None; self.model_var.set("model stopped - it reloads with your next message")
        if self.conv is not None and (self._live_answer or self._think_live or self.act):
            m = {"role": "assistant", "content": self._live_answer.strip() + "\\n[force-stopped]", "meta": {"stats": "force-stopped"}}
            self.conv["messages"].append(m); core.save_conversation(self.conv)
        if self._think_live: self._collapse_live_think()
        self._write("\\n[force-stopped - the engine was killed; it reloads with the next message]\\n", "info")
        self.status.set("force-stopped"); self._on_idle(None)

    def stop(self):''')
rep('''    def _on_error(self, tb):
        self._think_live = ""''', '''    def _on_error(self, tb):
        if self.forced: self._after_force(); return
        self._think_live = ""''')
rep('''        new, stats = val; self.conv["messages"] += new; core.save_conversation(self.conv)''',
'''        new, stats = val; self.conv["messages"] += new; core.save_conversation(self.conv)
        if self.forced: self._after_force(); return   # a force stop landed between rounds''')
# ---------------------------------------------------------------- activity tree (one per reply, embedded in the chat)
rep('''    def _on_start(self, name): self._write(f"{name}\\n", "asst_h")''',
'''    def _on_start(self, name):
        self._write(f"{name}\\n", "asst_h"); self._live_answer = ""
        fr = tk.Frame(self.tx, bg=FIELD); tree = ttk.Treeview(fr, columns=("status",), show="tree", height=1, selectmode="browse")
        tree.column("#0", width=560, stretch=True); tree.column("status", width=380, stretch=True); tree.pack(fill="x")
        tree.bind("<Double-1>", lambda e, t=tree: self._act_detail(t))
        self.tx.configure(state="normal"); self.tx.window_create("end", window=fr); self.tx.insert("end", "\\n"); self.tx.configure(state="disabled")
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
            label = f"\\u2699 {ev.get('name')} \\u00b7 {ev.get('action') or '...'} \\u00b7 {where}".rstrip(" \\u00b7")
            st = f"writing ... {len(ev['text'])} chars" if ev.get("text") is not None else f"composing call ... {ev.get('chars', 0)} chars"
            if self.act["call"] is None: self.act["call"] = self._act_node(r, label, st)
            else: t.item(self.act["call"], text=label, values=(st,))
            if ev.get("text") is not None: self.act["text"][self.act["call"]] = ev["text"]
            if ev.get("name") == "artifact": self._art_live(ev)
        elif k == "call":
            a = ev.get("args", {}); where = (a.get("name") or "") + ("/" + a["path"] if a.get("path") else "")
            label = f"\\u2699 {ev['name']} \\u00b7 {a.get('action', '')} \\u00b7 {where}".rstrip(" \\u00b7")
            if self.act["call"] is None: self.act["call"] = self._act_node(r, label, "running ...")
            else: t.item(self.act["call"], text=label, values=("running ...",))
            self.act["text"][self.act["call"]] = json.dumps(a, indent=1, ensure_ascii=False)
        elif k == "result":
            c = self.act["call"] or self._act_node(r, f"\\u2699 {ev['name']}")
            res = str(ev.get("text", "")); ok = not res.startswith("error")
            t.item(c, values=(("\\u2713 " if ok else "\\u2717 ") + " ".join(res.split())[:70],))
            self.act["text"][c] = self.act["text"].get(c, "") + "\\n\\n--- result ---\\n" + res; self.act["call"] = None; self.act["think"] = None
            if ev["name"] == "harness_dev": self._dev_changed()

    def _act_think(self, t):
        if not self.act: return
        r = self.act["round"] or self._act_node("", "Round 1", "working ...")
        self.act["think_txt"] += t; txt = self.act["think_txt"]; words = len(txt.split())
        if self.act["think"] is None: self.act["think"] = self._act_node(r, "\\u25cb thinking", "")
        self.act["tree"].item(self.act["think"], text=f"\\u25cb thinking ({words} words)", values=(" ".join(txt.split())[-70:],)); self.act["text"][self.act["think"]] = txt''')
rep('''    def _on_thinking(self, t):
        self._think_live += t; self._write(t, ("think", "think_live"))''',
'''    def _on_thinking(self, t):
        self._think_live += t; self._write(t, ("think", "think_live")); self._act_think(t)''')
rep('''    def _on_assistant(self, t):
        if self._think_live:''', '''    def _on_assistant(self, t):
        self._live_answer += t
        if self.act and self.act["round"]: self.act["think"] = None
        if self._think_live:''')
rep('''        if self.forced: self._after_force(); return   # a force stop landed between rounds''', '''        if self.forced: self._after_force(); return   # a force stop landed between rounds
        if self.act and self.act["tree"].get_children(): self.act["tree"].item(self.act["tree"].get_children()[-1], values=("done",))''')
# ---------------------------------------------------------------- live artifact view
rep('''        ttk.Button(bar, text="Save file", style="Accent.TButton", command=self._art_save).pack(side="right")''',
'''        ttk.Button(bar, text="Save file", style="Accent.TButton", command=self._art_save).pack(side="right")
        self.art_follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Follow the model live", variable=self.art_follow).pack(side="right", padx=10)''')
rep('''    def _art_save(self):''', '''    def _art_live(self, ev):
        """show an artifact file while the model is still writing it (the call's content as decoded so far)"""
        if not getattr(self, "art_follow", None) or not self.art_follow.get() or ev.get("text") is None: return
        A = self._A(); name = A._slug(ev.get("target") or ""); rel = (ev.get("path") or "").replace("\\\\", "/").strip("/")
        if name and (A.ROOT / name / ".artifact.json").exists():
            if not rel: rel = A._meta(A.ROOT / name).get("main", "")
            self.art_name = name; self.art_path = rel or self.art_path
        verb = {"create": "creating", "write": "writing", "append": "appending to", "edit": "editing (new text)"}.get(ev.get("action"), ev.get("action") or "writing")
        self.art_head.set(f"\\u25cf LIVE  {name or '(new artifact)'}{'/' + rel if rel else ''}  -  model is {verb}  ({len(ev['text'])} chars)")
        self.art_file_var.set("live view of the model's tool call - the file is saved when the call completes")
        self.art_text.delete("1.0", "end"); self.art_text.insert("1.0", ev["text"]); self.art_text.see("end")

    def _art_save(self):''')
# ---------------------------------------------------------------- model-written plugins: review queue; model-written skills marked
rep('''        bot = ttk.LabelFrame(self.t_plug, text="  Connections (all relative to the harness folder)  "); bot.pack(fill="both", expand=True, padx=14, pady=(6, 10))''',
'''        pend = ttk.LabelFrame(self.t_plug, text="  Written by the model - not loaded until you approve (plugins\\\\pending)  "); pend.pack(fill="both", expand=True, padx=14, pady=(6, 6))
        self.pend_tree = ttk.Treeview(pend, columns=("name", "size", "when", "desc"), show="headings", height=4)
        for c, w, t in (("name", 140, "Plugin"), ("size", 80, "Size"), ("when", 160, "Written"), ("desc", 640, "Description (from its source)")): self.pend_tree.heading(c, text=t); self.pend_tree.column(c, width=w, anchor="w")
        self.pend_tree.pack(fill="both", expand=True, padx=8, pady=8); self.pend_tree.bind("<Double-1>", lambda e: self._pend_view())
        pb = ttk.Frame(pend); pb.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(pb, text="View code", command=self._pend_view).pack(side="left")
        ttk.Button(pb, text="Approve and enable", style="Accent.TButton", command=self._pend_approve).pack(side="left", padx=6)
        ttk.Button(pb, text="Discard", command=self._pend_discard).pack(side="left")
        ttk.Label(pb, text="Plugins run as you, with full access to this computer: read the code before approving.", style="Dim.TLabel").pack(side="left", padx=10)
        bot = ttk.LabelFrame(self.t_plug, text="  Connections (all relative to the harness folder)  "); bot.pack(fill="both", expand=True, padx=14, pady=(6, 10))''')
rep('''        self._refresh_plugins(); self._refresh_connections()

    def _refresh_plugins(self):''', '''        self._refresh_plugins(); self._refresh_connections(); self._refresh_pending()

    def _refresh_pending(self):
        import re as _re, datetime as _dt
        self.pend_tree.delete(*self.pend_tree.get_children()); d = hp.PLUGINS / "pending"; d.mkdir(exist_ok=True)
        for f in sorted(d.glob("*.py")):
            src = f.read_text(encoding="utf-8", errors="replace"); m = _re.search(r'^DESCRIPTION\\s*=\\s*\\(?\\s*["\\'](.+?)["\\']', src, _re.M)
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
        if dst.exists(): messagebox.showerror("Approve", f"plugins\\\\{n}.py already exists.", parent=self); return
        if not messagebox.askyesno("Approve plugin", f"Load '{n}' as a tool the model can call?\\n\\nIt will run as Python with your permissions. "
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

    def _refresh_plugins(self):''')
rep('''        for n in names: self.slist.insert("end", f"  {'* ' if self.skills[n]['always_on'] else ''}/{n}")''',
    '''        for n in names: self.slist.insert("end", f"  {'* ' if self.skills[n]['always_on'] else ''}/{n}{'   [by model]' if self.skills[n].get('author') == 'model' else ''}")''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_app: Force stop, activity tree, live artifact view, plugin review queue")
