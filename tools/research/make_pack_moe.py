"""make_pack_moe.py - teach pack_universal_v3.py mixture-of-experts models (Qwen3-MoE layout), in place.

Each expert becomes its own stream (gate, up, down sub-blobs) so the engine can read just the experts a token routes to:
  streams = [layer 0..L-1: q,k,v,o + router] [embed] [lm_head] [expert (0,0) .. (0,E-1), (1,0) .. (L-1,E-1)]
vault.idx gains:  moe 1 / n_experts E / top_k K / moe_inter I / norm_topk 0|1 / expert_stream0 S   (WVAULT 6)
Dense models are packed exactly as before."""
import ast
P = r"C:\Users\1fred\Desktop\AI\Hermes\cube-harness\pack_universal_v3.py"
s = open(P, encoding="utf-8", newline="").read().replace("\r\n", "\n")
if "EXPERT_RE" in s: print("already patched"); raise SystemExit
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old)); s = s.replace(old, new)

rep("DT = {", 'EXPERT_RE = re.compile(r"^mlp\\.experts\\.(\\d+)\\.(gate_proj|up_proj|down_proj)\\.weight$")\nDT = {')
rep('''        for li in range(len(g)):
            ents = g[li]
            mats = sorted([(n, r) for n, r in ents if len(T[n]["shape"]) >= 2], key=lambda x: (_rank(_stem(x[1]), KNOWN_MAT), x[0]))''',
    '''        experts = []
        for li in range(len(g)):
            ents = g[li]
            ex = {}
            for n, r in ents:
                m = EXPERT_RE.match(r)
                if m: ex.setdefault(int(m[1]), {})[m[2]] = n
            if ex:
                for e in sorted(ex): experts.append(dict(kind="expert", layer=li, expert=e, mats=[ex[e]["gate_proj"], ex[e]["up_proj"], ex[e]["down_proj"]]))
                ents = [(n, r) for n, r in ents if not EXPERT_RE.match(r)] + [(n, r) for n, r in ents if EXPERT_RE.match(r)]
            mats = sorted([(n, r) for n, r in ents if len(T[n]["shape"]) >= 2 and not EXPERT_RE.match(r)], key=lambda x: (_rank(_stem(x[1]), KNOWN_MAT), x[0]))''')
rep('''    for n in rest_mats:
        if n not in (embed, head):
            streams.append(dict(kind="extra", layer=None, mats=[n]))
    return streams, resident, len(layers)''', '''    for n in rest_mats:
        if n not in (embed, head):
            streams.append(dict(kind="extra", layer=None, mats=[n]))
    if main:
        streams += experts
    return streams, resident, len(layers)''')

MOE = '''

def engine_moe_layout(cfg, sc, streams, nl, res_names, T):
    """MoE engine contract: per layer q/k/v/o + router (mlp.gate), norms ln_in/ln_post/q_norm/k_norm; experts gate/up/down, SiLU"""
    c = cfg.get("text_config", cfg); why = []
    ex = [s for s in streams if s["kind"] == "expert"]; E = c.get("num_experts") or c.get("num_local_experts"); K = c.get("num_experts_per_tok")
    if not ex: return None
    if len(ex) != nl * (E or 0): why.append(f"{len(ex)} expert streams, expected {nl} x {E}")
    for li in range(nl):
        if streams[li]["stems"] != ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate"]:
            why.append(f"layer {li} matrices {streams[li]['stems']}"); break
    if c.get("hidden_act", "silu") != "silu": why.append("activation")
    if c.get("mlp_only_layers") or c.get("decoder_sparse_step", 1) != 1: why.append("mixed dense/MoE layers")
    if c.get("shared_expert_intermediate_size"): why.append("shared expert")
    if (c.get("rope_scaling") or {}).get("rope_type", "default") not in ("default", None): why.append("rope scaling")
    first = next(i for i, s in enumerate(streams) if s["kind"] == "expert")
    want = {"input_layernorm.weight": "ln_in", "post_attention_layernorm.weight": "ln_post", "self_attn.q_norm.weight": "qn", "self_attn.k_norm.weight": "kn"}
    pre0 = streams[0]["mats"][0].rsplit(".self_attn.", 1)[0] + "."
    mine = [n for n in res_names if n.startswith(pre0)]; cur = []; o = 0
    for n in mine:
        r = want.get(n[len(pre0):])
        if r is None: why.append(f"unsupported per-layer tensor {n[len(pre0):]}"); break
        cnt = int(np.prod(T[n]["shape"])); cur.append((r, o)); o += cnt
    lines = (f"moe 1\\nn_experts {E}\\ntop_k {K}\\nmoe_inter {c.get('moe_intermediate_size')}\\nnorm_topk {1 if c.get('norm_topk_prob') else 0}\\n"
             f"expert_stream0 {first}\\nlres {o}\\n" + "".join(f"r_{r} {off}\\n" for r, off in cur) + "supported 1\\n")
    return dict(supported=not why, reasons=why, idx_lines=lines if not why else "")
'''
rep("\n\ndef pack(model_dir, vault_dir, allow_convert=False, delete_consumed=False, codec=\"huffman4\"):", MOE + "\n\ndef pack(model_dir, vault_dir, allow_convert=False, delete_consumed=False, codec=\"huffman4\"):")
rep('''    eu = engine_u_layout(cfg, sc, streams, nl, res_names, T, lossy)
    with open(vd / "vault.idx", "w") as f:
        f.write(f"WVAULT {3 if compat else 4}\\n''', '''    mo = engine_moe_layout(cfg, sc, streams, nl, res_names, T)
    eu = engine_u_layout(cfg, sc, streams, nl, res_names, T, lossy) if mo is None else mo
    if mo is not None:
        compat = False; sc = dict(sc, inter=cfg.get("text_config", cfg).get("moe_intermediate_size", sc["inter"]))
        if not mo["supported"]: print("MoE engine cannot run this vault:", "; ".join(mo["reasons"]))
    with open(vd / "vault.idx", "w") as f:
        f.write(f"WVAULT {6 if mo is not None else (3 if compat else 4)}\\n''')
ast.parse(s); open(P, "w", encoding="utf-8", newline="").write(s); print("pack_universal_v3.py: MoE streams added")
