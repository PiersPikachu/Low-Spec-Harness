#!/usr/bin/env python3
"""
chat.py - talk to a model in the harness, straight from its compressed vault.

  chat.py <vault_dir> [--mode auto|cpu|gpu|split] [--gpu-layers N] [--ctx 2048] [--max-new 512]
                      [--temp 0.7] [--top-p 0.8] [--top-k 20] [--greedy] [--seed N] [--system "..."] [--think]
                      [--draft auto|none|<vault>] [--k 8]

Speculative decoding (default for big streamed models): a small vault with the same vocabulary drafts k tokens on the GPU and the
big model checks them in one pass over its weights. With --greedy the replies are exactly the big model's own greedy replies; with
sampling (the default --temp 0.7 --top-p 0.8 --top-k 20) every token is drawn exactly from the big model's own sampling distribution
(speculative sampling: accept with min(1, p/q), else resample from max(0, p - q)), so quality is the big model's, only faster.
It is chosen automatically when the big model streams from disk and a draft is found (vault_* folders here, or paths in drafts.txt);
--draft none turns it off.

The planner measures this machine, picks the execution mode (GPU-resident / GPU+CPU split / CPU) and says which one it chose.
Chat models use their own chat template (shipped inside the vault); base models (no template) run as plain text continuation.
The engine keeps its KV cache between turns, so only the new tokens of each turn are processed.
In-chat commands:  /reset  clear the conversation     /stats  last turn's timing     /exit  quit
"""
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planner as pl
import harness_paths as hp
from spec import Proc, accept_step, sample_dist

try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass


def moe_banner(M, vram_mb, prefetch=None):
    pf = (2 if M.get("gptoss") else 0) if prefetch is None else int(prefetch)
    return (f"{M['name']}: mixture of experts, {M['nl']} layers x {M['n_experts']} experts (top {M['top_k']}), vault {M['total'] / 1e9:.1f} GB; "
            f"attention + output layer pinned, experts cached in RAM (LRU)" + (f" + {vram_mb} MB VRAM" + (" computed on the GPU" if M.get("gptoss") else "") if vram_mb > 500 else ", no VRAM tier")
            + ", missing experts read from the SSD" + (f" with next-layer prefetch ({pf} per token)" if pf > 0 else ""))


def pick_engine(vault, a):
    M = pl.model_from_vault(vault)
    if M.get("moe"):
        # MoE vaults: CPU engine with the expert cache; VRAM as the second expert tier (gpt-oss experts are computed there);
        # no speculative decoding (verifying k tokens touches ~30 experts per layer instead of 8 - log X6), no draft
        H = pl.probe(); vram = max(0, int(H.get("vram_free_mb", 0)) - 900)
        try: tuned = json.load(open(Path(vault) / "tuned.json"))["settings"]
        except Exception: tuned = {}
        pf = getattr(a, "moe_prefetch", None); pf = tuned.get("moe_prefetch", 2 if M.get("gptoss") else 0) if pf is None else pf
        use_vram = getattr(a, "moe_vram", None); use_vram = tuned.get("moe_vram", True) if use_vram is None else use_vram
        if not use_vram: vram = 0
        args = [str(pl.ENGINE), "serve", str(vault), "--ctx", str(a.ctx), "--batch", "128", "--resident-mb", "9000"] + (["--vram-mb", str(vram)] if vram > 500 else []) \
            + (["--kv16"] if getattr(a, "kv16", False) else []) \
            + (["--prefetch", str(int(pf))] if M.get("gptoss") and int(pf) > 0 else [])
        print(moe_banner(M, vram, pf if M.get("gptoss") else 0))
        return args, (f"mixture of experts: RAM expert cache + {vram} MB VRAM tier" if vram > 500 else "mixture of experts: RAM expert cache") \
            + (" with GPU experts" if M.get("gptoss") and vram > 500 else "") + (f", prefetch {int(pf)}" if M.get("gptoss") and int(pf) > 0 else ""), None
    H = pl.probe(); H = dict(H, nvme_MBps=pl.drive_MBps(vault)); R = pl.plan(M, H, a.ctx)
    print(pl.banner(M, H, R, a.ctx))
    opts = {o["code"]: o for o in R["options"]}; cpu = opts["C"]
    # speculative decoding: only when the target would stream from disk (not fully resident) and a matching draft exists
    draft = None
    if a.draft not in ("none", "off") and (a.draft != "auto" or (cpu["N"] > 0 and R["best_available"]["code"] != "G")):
        draft = Path(a.draft) if a.draft != "auto" else find_draft(vault, H)
    if draft:
        res_mb = str(int(cpu["R"] / 1e6) + 1)
        # VRAM left after the draft (weights + its KV + CUDA contexts) goes to the target's first layers; the GPU worker
        # decodes each of those layers once per VERIFY for all k+1 tokens (measured 2026-11-23: 8 layers, 9 tokens 12.1 s vs 13.5 s CPU-only)
        di = idx_info(draft); dkv = 2 * a.ctx * int(di["kv_heads"]) * int(di["head_dim"]) * 4 * int(di["layers"])
        budget = H.get("vram_free_mb", 0) * 1048576 - ((Path(draft) / "vault.bin").stat().st_size + dkv) - 2 * 350e6 - 250e6
        kvl = 2 * a.ctx * M["nkv"] * M["hd"] * 4; ng = 0
        for b in M["layer_bytes"]:
            if budget < b + kvl + 120e3: break
            budget -= b + kvl + 120e3; ng += 1
        # with a contiguous vault the 14B pass is CPU-bound and rans4 GPU layers no longer help (2026-11-23: VERIFY 9 = 5.9 s CPU-only vs 6.3 s with 8 GPU layers)
        ng = a.gpu_layers if a.gpu_layers else 0
        targs = [str(pl.ENGINE), "serve", str(vault), "--ctx", str(a.ctx), "--batch", "128", "--resident-mb", res_mb] + (["--gpu-layers", str(ng)] if ng else [])
        dargs = [str(pl.GPU_ENGINE), "serve", str(draft), "--ctx", str(a.ctx)]
        where = f"first {ng} layers on GPU, rest on CPU" if ng else "CPU"
        mode = "greedy" if a.temp <= 0 else f"sampling t={a.temp} p={a.top_p} k={a.top_k}"
        return (targs, dargs), f"speculative {mode}: target {where} ({res_mb} MB pinned, rest streamed), draft {draft.name} on GPU, k={a.k}", cpu
    if a.mode == "auto": o = R["best_available"]
    elif a.mode == "gpu": o = opts.get("G") or sys.exit("model does not fit in free VRAM for --mode gpu")
    elif a.mode == "split": o = opts.get("S") or sys.exit("no split layout possible on this machine")
    else: o = cpu
    res_mb = str(int(o["R"] / 1e6) + 1 if o["code"] in ("C", "S") else int(cpu["R"] / 1e6) + 1)
    if o["code"] == "G":
        args = [str(pl.GPU_ENGINE), "serve", str(vault), "--ctx", str(a.ctx)]; how = "GPU-resident (all compressed weights in VRAM)"
    elif o["code"] == "S":
        n = a.gpu_layers or o["gpu_layers"]; args = [str(pl.ENGINE), "serve", str(vault), "--ctx", str(a.ctx), "--gpu-layers", str(n), "--resident-mb", res_mb]
        how = f"GPU+CPU split ({n} layers on the GPU, the rest on the CPU)"
    else:
        args = [str(pl.ENGINE), "serve", str(vault), "--ctx", str(a.ctx), "--resident-mb", res_mb]; how = "CPU"
    return args, how, o


def idx_info(vault):
    kv = {}
    for l in open(Path(vault) / "vault.idx"):
        t = l.split()
        if len(t) == 2: kv[t[0]] = t[1]
    return kv


def find_draft(target, H):
    """a vault with the target's vocabulary and end tokens, under 1/8 of its size, that fits the free VRAM; huffman4 preferred (fast GPU kernel)"""
    ti = idx_info(target); tsize = (Path(target) / "vault.bin").stat().st_size; teos = {ti.get("eos"), ti.get("eos2")}
    cands = list(hp.find_vaults().values())                 # vaults/, legacy vault_* folders, extra folders, drafts.txt
    ok = []
    for d in cands:
        try:
            if d.resolve() == Path(target).resolve() or not (d / "vault.idx").exists(): continue
            di = idx_info(d); dsize = (d / "vault.bin").stat().st_size
            if di.get("vocab") != ti.get("vocab") or {di.get("eos"), di.get("eos2")} != teos or dsize * 8 > tsize: continue
            if dsize > (H.get("vram_free_mb", 0) * 1048576 - 600e6): continue
            ok.append((di.get("codec", "0") != "0", dsize, d))
        except Exception:
            continue
    return sorted(ok)[0][2] if ok else None


class SpecPair:
    """greedy speculative decoding: draft (GPU) proposes k tokens, target (CPU) VERIFYs them in one weight pass"""
    def __init__(self, targs, dargs, log_dir, k, eos):
        hp.LOGS.mkdir(exist_ok=True)
        self.T = Proc(targs, str(hp.LOGS / "engine.log")); self.D = Proc(dargs, str(hp.LOGS / "draft.log"))
        self.k = k; self.eos = eos; self.fed = []; self.batch = 128     # prompt tokens per weight pass
        self.stop_flag = False

    def request_stop(self):
        self.stop_flag = True

    def reset(self):
        for e in (self.T, self.D): e.send("RESET\n"); e.line()
        self.fed = []

    def generate(self, ids, max_new, temp, top_p, top_k, seed, on_token, ctx, should_stop=None):
        # reuse the KV cache up to the longest common prefix (templates may re-render earlier turns, e.g. tool calls),
        # roll back with SETPOS instead of re-reading the whole conversation
        lcp = 0
        for x_, y_ in zip(self.fed, ids):
            if x_ != y_: break
            lcp += 1
        lcp = min(lcp, len(ids) - 1)
        if lcp > 0: self.T.setpos(lcp); self.D.setpos(lcp)
        elif self.fed: self.reset()
        new = ids[lcp:]
        import random
        rng = random.Random(seed); sp = (temp, top_p, top_k); t0 = time.time(); rows = None; self.stop_flag = False
        for i in range(0, len(new), self.batch): rows = self.T.verifyp(new[i:i + self.batch], *sp)
        self.D.gen(new, 1); self.D.setpos(len(ids)); t_pre = time.time() - t0
        gen = [sample_dist(rows[-1], rng)]; S = ids + gen; on_token(gen); passes = 1; drafted = kept = 0; reason = "max"
        while True:
            if gen[-1] in self.eos: reason = "eos"; break
            if len(gen) >= max_new: reason = "max"; break
            if len(S) + self.k + 2 >= ctx: reason = "ctx"; break
            if (should_stop and should_stop()) or self.stop_flag: reason = "stopped" if not self.stop_flag else "stop"; break
            x = S[-1]; k = min(self.k, max_new - len(gen))
            d, qs = self.D.draftp([x], k, *sp, rng.getrandbits(62) | 1); ps = self.T.verifyp([x] + d, *sp); passes += 1
            add, j = accept_step(d, qs, ps, rng)
            cut = next((i for i, t in enumerate(add) if t in self.eos), None)
            if cut is not None: add = add[:cut + 1]
            gen += add; S += add; drafted += len(d); kept += j
            self.T.setpos(len(S) - 1); self.D.setpos(len(S) - 1); on_token(gen)
        gen = gen[:max_new]; keep = len(ids) + len(gen) - 1           # KV holds everything except the last token
        self.T.setpos(keep); self.D.setpos(keep); self.fed = (ids + gen)[:keep]
        dt = time.time() - t0
        stats = (f"tok_s={len(gen) / max(dt - t_pre, 1e-9):.3f} prefill_ms={t_pre * 1e3:.0f} passes={passes} "
                 f"tokens_per_pass={len(gen) / passes:.2f} drafts_kept={kept}/{drafted}")
        return gen, reason, stats

    def close(self):
        self.T.close(); self.D.close()


class Engine:
    def __init__(self, args, log):
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(log, "w"), text=True, bufsize=1)
        line = self.p.stdout.readline()
        if line.strip() != "READY":
            raise SystemExit(f"engine failed to start - see {log}")
        self.fed = []                                      # tokens currently in the engine's KV cache
        self._stop_sent = False

    def request_stop(self):
        """end the running GEN after the next token (the engine polls stdin for STOP between tokens)"""
        if not self._stop_sent:
            self._stop_sent = True
            try: self.p.stdin.write("STOP\n"); self.p.stdin.flush()
            except Exception: pass

    def reset(self):
        self.p.stdin.write("RESET\n"); self.p.stdin.flush(); self.p.stdout.readline(); self.fed = []

    def generate(self, ids, max_new, temp, top_p, top_k, seed, on_token):
        # reuse the KV cache up to the longest common prefix (templates may re-render earlier turns, e.g. tool calls),
        # roll back with SETPOS instead of re-reading the whole conversation
        lcp = 0
        for x_, y_ in zip(self.fed, ids):
            if x_ != y_: break
            lcp += 1
        lcp = min(lcp, len(ids) - 1)
        if lcp > 0:
            self.p.stdin.write(f"SETPOS {lcp}\n"); self.p.stdin.flush()
            if self.p.stdout.readline().startswith("ERR"): self.reset(); lcp = 0      # e.g. a sliding-window ring cannot roll back that far
        elif self.fed: self.reset()
        new = ids[lcp:]; self._stop_sent = False
        self.p.stdin.write(f"GEN {max_new} {temp} {top_p} {top_k} {seed} {len(new)}\n{' '.join(map(str, new))}\n"); self.p.stdin.flush()
        out, stats, reason = [], "", "?"
        while True:
            line = self.p.stdout.readline()
            if not line: raise SystemExit("engine stopped unexpectedly")
            if line.startswith("T "): out.append(int(line.split()[1])); on_token(out)
            elif line.startswith("S "): stats = line[2:].strip()
            elif line.startswith("END"): reason = line.split()[1]; break
            elif line.startswith("ERR"): raise RuntimeError(line.strip())
        # what the engine actually forwarded: everything except a final token that ended the turn (eos / ctx)
        self.fed = ids + (out if reason in ("max", "stop") else out[:-1])
        return out, reason, stats

    def close(self):
        try:
            self.p.stdin.write("QUIT\n"); self.p.stdin.flush(); self.p.wait(timeout=20)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vault"); ap.add_argument("--mode", default="auto", choices=["auto", "cpu", "gpu", "split"]); ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=2048); ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.8); ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--greedy", action="store_true"); ap.add_argument("--seed", type=int); ap.add_argument("--system", default=None); ap.add_argument("--think", action="store_true")
    ap.add_argument("--draft", default="auto"); ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args(); vault = hp.resolve_vault(a.vault)     # a name (searched in vaults/ etc.) or a path
    if a.greedy: a.temp, a.top_p, a.top_k = 0.0, 1.0, 1
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(str(vault))
    except Exception as e:
        raise SystemExit(f"no tokenizer in {vault} ({e}). Repack, or copy the model's tokenizer files into the vault folder.")
    chat_mode = bool(tok.chat_template)
    args, how, o = pick_engine(vault, a)
    print(f"\nloading {vault.name} ... ", end="", flush=True); t0 = time.time()
    if isinstance(args, tuple):
        eos = {int(v) for k_, v in idx_info(vault).items() if k_ in ("eos", "eos2")}
        eng = SpecPair(args[0], args[1], vault, a.k, eos)
    else:
        hp.LOGS.mkdir(exist_ok=True); eng = Engine(args, str(hp.LOGS / "engine.log"))
    print(f"ready in {time.time() - t0:.0f}s - mode: {how}")
    print("chat mode (model's own template)" if chat_mode else "base model: plain text continuation (no chat template)")
    print("commands: /reset /stats /exit\n")
    msgs = [{"role": "system", "content": a.system}] if a.system else []; text = ""; last = ""
    try:
        while True:
            try:
                user = input("you> ")
            except EOFError:
                break
            if not user.strip(): continue
            if user.strip() == "/exit": break
            if user.strip() == "/reset": msgs = msgs[:1] if a.system else []; text = ""; eng.reset(); print("(conversation cleared)\n"); continue
            if user.strip() == "/stats": print(last or "(no turn yet)"); continue
            if chat_mode:
                msgs.append({"role": "user", "content": user})
                kw = {"enable_thinking": a.think, "thinking": a.think}   # Qwen3: enable_thinking, Hermes-4: thinking
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **kw)
                if hasattr(ids, "keys"): ids = ids["input_ids"]
            else:
                text += user; ids = tok(text)["input_ids"]
            while len(ids) + a.max_new > a.ctx and chat_mode and len(msgs) > (2 if a.system else 1):   # drop the oldest turns to fit the context
                del msgs[1 if a.system else 0]; ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, **{"enable_thinking": a.think, "thinking": a.think})
                if hasattr(ids, "keys"): ids = ids["input_ids"]
            if len(ids) + 1 >= a.ctx: print("(prompt longer than --ctx; use /reset or a larger --ctx)\n"); msgs.pop() if chat_mode else None; continue
            shown = [""]
            def on_token(out):
                s = tok.decode(out, skip_special_tokens=True)
                if s.startswith(shown[0]): print(s[len(shown[0]):], end="", flush=True); shown[0] = s
            print("model> ", end="", flush=True); t1 = time.time()
            seed = a.seed if a.seed is not None else random.randrange(1, 2 ** 62)
            if isinstance(eng, SpecPair): out, reason, stats = eng.generate(list(ids), a.max_new, a.temp, a.top_p, a.top_k, seed, on_token, a.ctx)
            else: out, reason, stats = eng.generate(list(ids), a.max_new, a.temp, a.top_p, a.top_k, seed, on_token)
            reply = tok.decode(out, skip_special_tokens=True)
            if chat_mode: msgs.append({"role": "assistant", "content": reply.strip()})
            else: text += reply
            kv = {k: v for k, v in (x.split("=") for x in stats.split() if "=" in x)}
            extra = f", {kv['tokens_per_pass']} tokens per target pass, drafts kept {kv['drafts_kept']}" if "tokens_per_pass" in kv else ""
            last = f"[{len(out)} tokens in {time.time() - t1:.1f}s, {kv.get('tok_s', '?')} tok/s decode, prefill {kv.get('prefill_ms', '?')} ms{extra}, stop: {reason}, mode: {how}]"
            print(f"\n{last}\n")
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
