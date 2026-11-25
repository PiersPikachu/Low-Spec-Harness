"""make_app_artifacts.py - harness_app.py: an Artifacts tab for the artifact plugin (browse, edit, preview, history/revert, export).
Edits in place; every user edit goes through the plugin so it is versioned like the model's."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_app.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

rep('''        self.t_msg, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(4))
        for f, name in ((self.t_msg, "Messages"), (self.t_cfg, "Config"), (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")''',
'''        self.t_msg, self.t_art, self.t_cfg, self.t_plug, self.t_skill = (ttk.Frame(nb) for _ in range(5))
        for f, name in ((self.t_msg, "Messages"), (self.t_art, "Artifacts"), (self.t_cfg, "Config"), (self.t_plug, "Plugins"), (self.t_skill, "Skills")): nb.add(f, text=f"   {name}   ")''')
rep("self._build_messages(); self._build_config(); self._build_plugins(); self._build_skills()",
    "self._build_messages(); self._build_artifacts(); self._build_config(); self._build_plugins(); self._build_skills()")

TAB = r'''    # ================================================================== Artifacts
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
        A = self._A(); p = A.ROOT / self.art_name / rel; self.art_path = rel
        try: body = p.read_text(encoding="utf-8")
        except Exception as ex: body = f"(cannot show this file as text: {ex})"
        self.art_text.delete("1.0", "end"); self.art_text.insert("1.0", body); self.art_text.edit_reset()
        self.art_file_var.set(f"{rel}   {p.stat().st_size if p.exists() else 0} bytes   - edit here and press Save file (versioned, revertible)")

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

    # ================================================================== Config
'''
rep("    # ================================================================== Config\n", TAB)
# keep the tab current while the model works: refresh after artifact tool calls and at the end of each reply
rep('''        if self._think_live: self._collapse_live_think()
        self._write("\\n" + t, "tool")''', '''        if self._think_live: self._collapse_live_think()
        self._write("\\n" + t, "tool")
        if t.startswith("[artifact]"): self._refresh_artifacts()''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_app: Artifacts tab")
