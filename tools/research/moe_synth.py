#!/usr/bin/env python3
"""moe_synth.py <out_dir> - a tiny random Qwen3-MoE checkpoint (bf16 safetensors + config) for testing the MoE packer and engine."""
import sys, torch
from pathlib import Path
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
out = Path(sys.argv[1] if len(sys.argv) > 1 else "moe_synth_model")
cfg = Qwen3MoeConfig(vocab_size=512, hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_hidden_layers=3,
                     num_attention_heads=8, num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2, norm_topk_prob=True,
                     max_position_embeddings=1024, rope_theta=1e6, tie_word_embeddings=False, decoder_sparse_step=1, mlp_only_layers=[])
torch.manual_seed(0); m = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
with torch.no_grad():
    for p in m.parameters(): p.mul_(4.0)                 # larger weights so outputs are not near-uniform
m.save_pretrained(out, safe_serialization=True); print("wrote", out, sum(p.numel() for p in m.parameters()), "params")
