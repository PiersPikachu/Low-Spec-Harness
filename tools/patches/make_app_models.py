"""make_app_models.py - Models tab (search Hugging Face, decision tree per model, install = download -> compress -> check -> tune,
tune installed vaults, use in chat) + tuned settings honoured by the engine launcher and 'Model defaults'."""
import ast
from pathlib import Path
H = Path(__file__).resolve().parent
def patch(fname, pairs):
    p = H / fname; s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (fname, old[:70], s.count(old)); s = s.replace(old, new)
    ast.parse(s); p.write_text(s, encoding="utf-8"); print("patched", fname)

# engine launcher: tuned MoE settings (cfg first, then <vault>/tuned.json)
patch("chat.py", [
    ('''        H = pl.probe(); vram = max(0, int(H.get("vram_free_mb", 0)) - 900)''',
     '''        H = pl.probe(); vram = max(0, int(H.get("vram_free_mb", 0)) - 900)
        try: tuned = json.load(open(Path(vault) / "tuned.json"))["settings"]
        except Exception: tuned = {}
        pf = getattr(a, "moe_prefetch", None); pf = tuned.get("moe_prefetch", 2 if M.get("gptoss") else 0) if pf is None else pf
        use_vram = getattr(a, "moe_vram", None); use_vram = tuned.get("moe_vram", True) if use_vram is None else use_vram
        if not use_vram: vram = 0'''),
    ('''            + (["--prefetch", "2"] if M.get("gptoss") else [])''', '''            + (["--prefetch", str(int(pf))] if M.get("gptoss") and int(pf) > 0 else [])'''),
])
patch("harness_core.py", [
    ('''draft=cfg["draft"] or "auto", kv16=bool(cfg.get("kv16")),''',
     '''draft=cfg["draft"] or "auto", kv16=bool(cfg.get("kv16")), moe_prefetch=cfg.get("moe_prefetch"), moe_vram=cfg.get("moe_vram"),'''),
    ('''    return s, [f"Recommended settings for {vault.name}:"] + notes''',
     '''    tj = vault / "tuned.json"
    if tj.exists():                        # measured on this machine by the Models tab tuner: its engine settings win
        try:
            t = json.load(open(tj)); s.update(t["settings"])
            notes.append(f"Tuned on this machine ({t['date']}): {t['tok_s']} tok/s with {t['settings']} - re-run Tune in the Models tab after hardware changes.")
        except Exception: pass
    return s, [f"Recommended settings for {vault.name}:"] + notes'''),
])

TAB = r'''    # ================================================================== Models (Hugging Face -> compressed vault)
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

    # ================================================================== Config
'''
patch("harness_app.py", [
    ('''        self.t_msg, self.t_art, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(5))
        for f, name in ((self.t_msg, "Messages"), (self.t_art, "Artifacts"), (self.t_cfg, "Config"), (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")''',
     '''        self.t_msg, self.t_art, self.t_mod, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(6))
        for f, name in ((self.t_msg, "Messages"), (self.t_art, "Artifacts"), (self.t_mod, "Models"), (self.t_cfg, "Config"), (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")'''),
    ("self._build_messages(); self._build_artifacts(); self._build_config();", "self._build_messages(); self._build_artifacts(); self._build_models(); self._build_config();"),
    ("    # ================================================================== Config\n", TAB),
    # Model defaults: keep settings that have no field (tuned MoE options) in the config too
    ('''        for k, v in rec.items():
            if k in self.vars: self.vars[k].set(bool(v) if isinstance(v, bool) else str(v))''',
     '''        for k, v in rec.items():
            if k in self.vars: self.vars[k].set(bool(v) if isinstance(v, bool) else str(v))
            else: self.cfg[k] = v'''),
])
