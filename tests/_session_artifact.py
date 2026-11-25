"""a real model using the artifact tool through the harness Session (as the window does); test artifacts are removed afterwards"""
import shutil, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(H / "app"), str(H / "tools"), str(H / "tools" / "research")]
import harness_core as core, harness_paths as hp
vault = sys.argv[1]
cfg = dict(hp.load_config()); rec, _ = core.recommended_settings(hp.resolve_vault(vault, cfg)); cfg.update(rec); cfg.update(vault=vault, greedy=True, max_new=1500)
P = core.load_plugins(); tools = core.tool_schemas(P, {"artifact"}); A = P["artifact"]
for n in ("test-notes", "counter-page"): shutil.rmtree(A.ROOT / n, ignore_errors=True)
s = core.Session(cfg); print("loaded |", s.how, flush=True); hist = []
for q in ["Create a note artifact named test-notes with a title and three short tips for writing clear commit messages.",
          "Now create a page artifact named counter-page: one self-contained HTML file with a button that counts clicks.",
          "In counter-page, change the button text to 'Click me' using an edit, not a full rewrite."]:
    hist.append({"role": "user", "content": q}); tl = []
    t0 = time.time(); new, stats = s.reply([{k: v for k, v in m.items() if k != "meta"} for m in hist], "", tools, P, lambda k, t: tl.append(t) if k == "tool" else None, lambda: False, 4)
    print(f"\nQ: {q}\n  tool traffic: {ascii(''.join(tl).strip())[:600]}\n  answer: {ascii(new[-1]['content'])[:240]}\n  {stats} | {time.time() - t0:.0f}s", flush=True)
    hist += new
s.close()
for n in ("test-notes", "counter-page"):
    d = A.ROOT / n
    if d.exists():
        print(f"\n== {n}: files {A._files(d)}"); m = A._meta(d); print(ascii((d / m['main']).read_text(encoding='utf-8'))[:900]); print("history:", [(c['n'], c['action']) for c in m['changes']])
