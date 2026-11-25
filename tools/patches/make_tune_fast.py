"""make_tune_fast.py - model_hub.tune: test only settings that really launch differently, and stop early on losers.

Measured before (Llama-3.2-1B, dense): 4 candidates, 225 s, all four launched the identical engine ("CPU": no draft matches Llama's
vocabulary and 'auto' chose CPU), and the GPU-resident layout was never tried. Dense models also reload the whole model (and the
draft) per candidate, so duplicates are expensive.

  plan first   each candidate is turned into its exact engine command line (chat.pick_engine, nothing loaded); identical ones
               are tested once. Dense candidates: GPU-resident / GPU+CPU split / CPU, each without and (when a draft with the
               same vocabulary exists and the target streams) with the draft at k = 8 and 4. Impossible ones are skipped.
  reuse        candidates that differ only in the draft's k keep the loaded engines (k is set in place)
  early stop   a candidate whose first test reply runs below 60% of the best so far is not given its second reply
  slow drive   a vault on a drive reading under 500 MB/s is flagged: tuning there measures the drive"""
import ast
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "app" / "model_hub.py"
s = P.read_text(encoding="utf-8")
a = s.index("def tune(vault, log=print, tokens=64):"); b = s.index('\nif __name__ == "__main__":')
NEW = r'''def _engine_key(vault, cfg):
    """the exact engine command line this configuration launches (nothing is loaded), or None if it is not possible here"""
    import chat, io, contextlib, types
    a = types.SimpleNamespace(mode=cfg["mode"], gpu_layers=int(cfg.get("gpu_layers") or 0), ctx=int(cfg["ctx"]), draft=cfg.get("draft") or "auto",
                              kv16=bool(cfg.get("kv16")), moe_prefetch=cfg.get("moe_prefetch"), moe_vram=cfg.get("moe_vram"), temp=0.0, top_p=1.0, top_k=1, k=int(cfg.get("k") or 8))
    try:
        with contextlib.redirect_stdout(io.StringIO()): args, how, _ = chat.pick_engine(vault, a)
    except (SystemExit, Exception): return None, None, None
    spec = isinstance(args, tuple)
    return json.dumps(args), (f"k={a.k}" if spec else ""), how


def tune(vault, log=print, tokens=64):
    import harness_core as core
    vault = Path(vault); base = dict(hp.load_config()); rec, _ = core.recommended_settings(vault); base.update(rec); base.update(vault=str(vault), greedy=True, max_new=tokens, think=False)
    idx = dict(l.split()[:2] for l in open(vault / "vault.idx") if len(l.split()) == 2)
    moe, gptoss = idx.get("moe") == "1", idx.get("gptoss") == "1"
    try:
        import storage; sp = storage.read_speed(vault.parent)
        if sp and sp < 500: log(f"note: {vault.name} is on a slow drive ({sp:.0f} MB/s) - tuning will measure the drive; move it to an SSD folder first for real numbers")
    except Exception: pass
    if gptoss: cands = [dict(moe_prefetch=2, moe_vram=True), dict(moe_prefetch=0, moe_vram=True), dict(moe_prefetch=2, moe_vram=False)]
    elif moe: cands = [dict(moe_vram=True), dict(moe_vram=False)]
    else:
        cands = [dict(mode=m, draft=d, k=k) for m in ("gpu", "split", "cpu") for d, ks in (("none", (8,)), ("auto", (8, 4))) for k in ks]
    plans, seen = [], set()
    for cd in cands:
        key, kk, how = _engine_key(vault, dict(base, **cd))
        if key is None or (key, kk) in seen: continue
        seen.add((key, kk)); plans.append((cd, key, kk, how))
    log(f"{len(plans)} distinct configuration(s) to test (of {len(cands)} candidates; duplicates and impossible layouts skipped): " + "; ".join(p[3] for p in plans))
    results, best = [], 0.0; held = None                              # held = (engine key, session) kept for a k-only change
    for cd, key, kk, how in plans:
        cfg = dict(base, **cd); t0 = time.time()
        try:
            if held and held[0] == key: s = held[1]; s.cfg.update(cd); setattr(s.eng, "k", int(cd.get("k", 8)))
            else:
                if held: held[1].close(); held = None
                s = core.Session(cfg)
            speeds, texts, early = [], [], False
            for i, p in enumerate(PROMPTS):
                new, st = s.reply([{"role": "user", "content": p}], "", None, {}, lambda k, t: None, lambda: False, 0)
                m = re.search(r"([\d.]+) tok/s decode", st); speeds.append(float(m.group(1)) if m else 0.0); texts.append(new[-1]["content"] + new[-1].get("thinking", ""))
                if i == 0 and best and speeds[0] < 0.6 * best: early = True; break
            held = (key, s)
            sane = all(t.strip() for t in texts) and not any(_loops(t) for t in texts)
            r = dict(settings=cd, tok_s=round(sum(speeds) / len(speeds), 3), sane=sane, how=s.how, secs=round(time.time() - t0), early_stop=early)
            if sane: best = max(best, r["tok_s"])
        except Exception as ex:
            r = dict(settings=cd, tok_s=0.0, sane=False, error=str(ex)[-300:], secs=round(time.time() - t0))
            if held: held[1].close(); held = None
        results.append(r)
        log(f"  {r.get('how', how)} {cd}: {r['tok_s']} tok/s in {r['secs']} s" + ("  (stopped early: slower)" if r.get("early_stop") else "")
            + ("" if r["sane"] else "  (rejected: " + (r.get("error") or "empty or looping reply") + ")"))
    if held: held[1].close()
    good = [r for r in results if r["sane"]]
    if not good: log("tuning found no working setting - keeping the recommended defaults"); return None
    best_r = max(good, key=lambda r: r["tok_s"])
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), settings=best_r["settings"], tok_s=best_r["tok_s"], results=results)
    json.dump(out, open(vault / "tuned.json", "w"), indent=1)
    log(f"best: {best_r.get('how', '')} {best_r['settings']} -> {best_r['tok_s']} tok/s in {sum(r['secs'] for r in results)} s total (saved to {vault.name}/tuned.json)")
    return out

'''
s = s[:a] + NEW + s[b:]; ast.parse(s); P.write_text(s, encoding="utf-8"); print("model_hub.tune: distinct configurations only, engine reuse, early stop, slow-drive note")
