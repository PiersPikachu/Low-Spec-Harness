"""what a 1,000,000-character context costs on this machine, per model (KV cache size, attention reads, prefill time)"""
CHARS = 1_000_000; CPT = 4.0                      # ~4 characters per token for English prose/code (tokenizer-dependent)
T = int(CHARS / CPT)
M = [  # name, layers that keep a full-length KV (sliding-window layers keep only 128 tokens), kv heads, head dim, native max context
    ("gpt-oss-20b", 12, 8, 64, 131072), ("gpt-oss-120b", 18, 8, 64, 131072), ("Qwen3-30B-A3B-2507", 48, 4, 128, 262144),
    ("Hermes-4-14B", 40, 8, 128, 40960), ("Qwen3-0.6B", 28, 8, 128, 40960)]
RAM_BW = 25e9                                     # bytes/s, DDR4-3200 dual channel, realistic
print(f"1,000,000 characters ~ {T:,} tokens at {CPT} chars/token\n")
print(f"{'model':20s} {'native ctx':>10s} {'KV/token fp32':>13s} {'KV @250k fp32':>13s} {'fp16':>8s} {'int8':>8s} {'attn read/token fp16':>21s}")
for n, L, kv, hd, nat in M:
    per = 2 * L * kv * hd                         # values per token (K and V)
    f32, f16, i8 = per * 4, per * 2, per * 1.0625  # int8 with one fp16 scale per 32 values
    print(f"{n:20s} {nat:>10,} {f32 / 1024:>10.0f} KB {f32 * T / 1e9:>10.1f} GB {f16 * T / 1e9:>6.1f} GB {i8 * T / 1e9:>6.1f} GB {f16 * T / RAM_BW:>17.2f} s/tok")
