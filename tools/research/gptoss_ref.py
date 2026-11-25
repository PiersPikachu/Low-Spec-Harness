#!/usr/bin/env python3
"""gptoss_ref.py - an independent numpy fp32 forward pass of gpt-oss, read straight from the released safetensors (MXFP4 dequantized
exactly: FP4 E2M1 code x 2^(scale-127)). Mirrors transformers' modeling_gpt_oss: RMSNorm, q/k/v/o with biases, YaRN RoPE (half-split
rotation, cos/sin x attention factor), attention sinks (an extra logit per head in the softmax, dropped after), 128-token sliding
window on 'sliding_attention' layers, top-k router on logits then softmax over the k, experts with interleaved gate/up, clamp 7.0,
glu = gate*sigmoid(1.702 gate), (up+1)*glu, expert biases. Used to check the engine (and checked itself against transformers).
  gptoss_ref.py <model_dir> <token ids...>   -> prints argmax and writes ref_logits.npy (last token)"""
import json, math, sys
from pathlib import Path
import numpy as np
from safetensors import safe_open

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def dequant(blocks, scales):
    """blocks [..., rows, cols/32, 16] uint8, scales [..., rows, cols/32] -> fp32 [..., rows, cols]"""
    lo, hi = blocks & 15, blocks >> 4
    v = np.stack([FP4[lo], FP4[hi]], -1).reshape(*blocks.shape[:-1], 32)            # low nibble = even index
    v = v * np.ldexp(np.float32(1), scales.astype(np.int32) - 127)[..., None]
    return v.reshape(*scales.shape[:-1], -1).astype(np.float32)


class Model:
    def __init__(self, d):
        d = Path(d); self.c = json.load(open(d / "config.json"))
        idx = d / "model.safetensors.index.json"
        self.wm = json.load(open(idx))["weight_map"] if idx.exists() else None; self.d = d; self.f = {}
        from transformers import AutoConfig
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        ac = AutoConfig.from_pretrained(str(d)); inv, self.ascale = ROPE_INIT_FUNCTIONS[ac.rope_parameters["rope_type"]](ac, "cpu")
        self.inv = inv.float().numpy()

    def t(self, n):
        fn = self.wm[n] if self.wm else "model.safetensors"
        if fn not in self.f: self.f[fn] = safe_open(str(self.d / fn), "pt")
        x = self.f[fn].get_tensor(n)
        import torch
        return x.numpy() if x.dtype in (torch.uint8,) else x.float().numpy()

    def forward(self, ids):
        c = self.c; L, hid, nh, nkv, hd, eps = c["num_hidden_layers"], c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
        K, lim, W = c["num_experts_per_tok"], c.get("swiglu_limit", 7.0), c.get("sliding_window", 128)
        n = len(ids); pos = np.arange(n, dtype=np.float32)[:, None] * self.inv[None]
        cos, sin = (np.cos(pos) * self.ascale).astype(np.float32), (np.sin(pos) * self.ascale).astype(np.float32)
        def rms(x, w): return (w * (x / np.sqrt(np.mean(x.astype(np.float32) ** 2, -1, keepdims=True) + eps))).astype(np.float32)
        def rope(x):
            a, b = x[..., :hd // 2], x[..., hd // 2:]; cs, sn = cos[:, None], sin[:, None]
            return np.concatenate([a * cs - b * sn, b * cs + a * sn], -1)
        H = self.t("model.embed_tokens.weight")[np.array(ids)]
        for l in range(L):
            p = f"model.layers.{l}."; x = rms(H, self.t(p + "input_layernorm.weight"))
            q = (x @ self.t(p + "self_attn.q_proj.weight").T + self.t(p + "self_attn.q_proj.bias")).reshape(n, nh, hd)
            k = (x @ self.t(p + "self_attn.k_proj.weight").T + self.t(p + "self_attn.k_proj.bias")).reshape(n, nkv, hd)
            v = (x @ self.t(p + "self_attn.v_proj.weight").T + self.t(p + "self_attn.v_proj.bias")).reshape(n, nkv, hd)
            q, k = rope(q), rope(k); K_ = np.repeat(k, nh // nkv, 1); V_ = np.repeat(v, nh // nkv, 1)
            sc = np.einsum("qhd,khd->hqk", q, K_) / np.float32(math.sqrt(hd))
            i, j = np.arange(n)[:, None], np.arange(n)[None]
            allowed = (j <= i) & ((j > i - W) if c.get("layer_types", ["full_attention"] * L)[l] == "sliding_attention" else True)
            sc = np.where(allowed[None], sc, -np.inf)
            sinks = self.t(p + "self_attn.sinks")[:, None, None] * np.ones((1, n, 1), np.float32)
            comb = np.concatenate([sc, sinks], -1); comb -= comb.max(-1, keepdims=True); pr = np.exp(comb); pr /= pr.sum(-1, keepdims=True)
            att = np.einsum("hqk,khd->qhd", pr[..., :-1], V_).reshape(n, nh * hd)
            H = H + att @ self.t(p + "self_attn.o_proj.weight").T + self.t(p + "self_attn.o_proj.bias")
            x = rms(H, self.t(p + "post_attention_layernorm.weight"))
            lg = x @ self.t(p + "mlp.router.weight").T + self.t(p + "mlp.router.bias")
            top = np.argsort(-lg, 1, kind="stable")[:, :K]; tv = np.take_along_axis(lg, top, 1); w = np.exp(tv - tv.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            e = p + "mlp.experts."; GUb, GUs, GUbias = self.t(e + "gate_up_proj_blocks"), self.t(e + "gate_up_proj_scales"), self.t(e + "gate_up_proj_bias")
            Db, Ds, Dbias = self.t(e + "down_proj_blocks"), self.t(e + "down_proj_scales"), self.t(e + "down_proj_bias")
            out = np.zeros_like(H)
            for ex in sorted(set(top.ravel().tolist())):
                rows, cols = np.nonzero(top == ex); Wgu = dequant(GUb[ex], GUs[ex]); Wd = dequant(Db[ex], Ds[ex])
                gu = x[rows] @ Wgu.T + GUbias[ex]; g, u = np.minimum(gu[:, ::2], lim), np.clip(gu[:, 1::2], -lim, lim)
                act = (u + 1) * (g * (1 / (1 + np.exp(-1.702 * g))))
                out[rows] += (act @ Wd.T + Dbias[ex]) * w[rows, cols][:, None]
            H = H + out
        return rms(H[-1:], self.t("model.norm.weight"))[0] @ self.t("lm_head.weight").T


if __name__ == "__main__":
    m = Model(sys.argv[1]); lg = m.forward([int(x) for x in sys.argv[2:]]); np.save("ref_logits.npy", lg)
    print("argmax", int(lg.argmax()), "top5", np.argsort(-lg)[:5].tolist())
