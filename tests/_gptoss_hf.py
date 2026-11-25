"""HF reference for a small gpt-oss cut, bf16 on the CPU (Mxfp4Config(dequantize=True)); refuses to start without 7 GB free RAM"""
import sys, time, ctypes, numpy as np, torch
class M(ctypes.Structure): _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in ("tp", "ap", "a", "b", "c", "d", "e")]
m_ = M(); m_.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m_))
if m_.ap < 7 * 2**30: sys.exit(f"only {m_.ap / 2**30:.1f} GB RAM free; need 7")
from transformers import AutoModelForCausalLM, Mxfp4Config
d = sys.argv[1]; ids = [int(x) for x in open(sys.argv[2]).read().split()]
t = time.time(); m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16, device_map="cpu", quantization_config=Mxfp4Config(dequantize=True)).eval()
print("loaded", round(time.time() - t, 1), "s", flush=True)
with torch.no_grad(): lg = m(torch.tensor([ids])).logits[0, -1].float().numpy()
np.save(sys.argv[3], lg); print("HF bf16 argmax", int(lg.argmax()), "top5", np.argsort(-lg)[:5].tolist())
