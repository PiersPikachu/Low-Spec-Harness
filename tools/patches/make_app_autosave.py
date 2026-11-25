"""make_app_autosave.py - Artifacts tab: the user's edits save themselves 1.5 s after typing stops (one revertible change per
editing session of a file); live model writes are on disk while they stream (core + artifact.live_update); Force stop keeps them."""
import ast
from pathlib import Path
P = Path(__file__).resolve().parent / "harness_app.py"
s = P.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)
rep('''        self.art_follow = tk.BooleanVar(value=True)''', '''        self.art_auto = tk.BooleanVar(value=True); self._as_job = None; self._as_session = None
        ttk.Checkbutton(bar, text="Autosave", variable=self.art_auto).pack(side="right", padx=4)
        self.art_follow = tk.BooleanVar(value=True)''')
rep('''        self.art_name = None; self.art_path = None; self._refresh_artifacts()''',
    '''        self.art_text.bind("<KeyRelease>", self._art_typed); self.art_text.bind("<<Paste>>", lambda e: self.after(10, self._art_typed))
        self.art_name = None; self.art_path = None; self._refresh_artifacts()''')
rep('''    def _art_save(self):''', '''    def _art_typed(self, _=None):
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

    def _art_save(self):''')
# switching file or artifact ends the editing session (next edit = new revertible change); flush a pending autosave first
rep('''    def _art_load(self, rel):''', '''    def _art_load(self, rel):
        if self._as_job: self.after_cancel(self._as_job); self._art_autosave()
        self._as_session = None''')
# Force stop keeps what the model had written live
rep('''    def _after_force(self):
        self.forced = False''', '''    def _after_force(self):
        self.forced = False
        try: self._A().live_keep(); self._refresh_artifacts()
        except Exception: pass''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("harness_app: autosave in the Artifacts tab; Force stop keeps live-saved content")
s = P.read_text(encoding="utf-8")
old = '''        if not getattr(self, "art_follow", None) or not self.art_follow.get() or ev.get("text") is None: return'''
assert s.count(old) == 1
s = s.replace(old, old + '''
        if self._as_job: self.after_cancel(self._as_job); self._art_autosave()      # save the user's typing before the live view takes over''')
ast.parse(s); P.write_text(s, encoding="utf-8"); print("live view flushes pending autosave")
