"""make_app_storage.py - 'Model folders' tab: several folders (on or off the SSD) holding compressed models; drive type and speed;
install target; download folder; move a vault between folders (copy, verify, delete)."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "app" / "harness_app.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
rep('''        self.t_msg, self.t_art, self.t_mod, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(6))
        for f, name in ((self.t_msg, "Messages"), (self.t_art, "Artifacts"), (self.t_mod, "Models"), (self.t_cfg, "Config"), (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")''',
'''        self.t_msg, self.t_art, self.t_mod, self.t_store, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(7))
        for f, name in ((self.t_msg, "Messages"), (self.t_art, "Artifacts"), (self.t_mod, "Models"), (self.t_store, "Model folders"), (self.t_cfg, "Config"),
                        (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")''')
rep("self._build_models(); self._build_config();", "self._build_models(); self._build_storage(); self._build_config();")
TAB = r'''    # ================================================================== Model folders
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
'''
rep("    # ================================================================== Config\n", TAB)
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_app: Model folders tab")
