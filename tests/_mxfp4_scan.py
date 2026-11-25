"""which Hugging Face models ship MXFP4 weights (the format pack_gptoss --mxf2 packs losslessly)?"""
import warnings; warnings.filterwarnings("ignore")
from huggingface_hub import HfApi
api = HfApi(); seen = set()
for q in ("mxfp4", "gpt-oss"):
    for m in api.list_models(search=q, sort="downloads", limit=15, expand=["downloads", "config"]):
        if m.id in seen: continue
        seen.add(m.id); c = getattr(m, "config", None) or {}; qc = (c.get("quantization_config") or {}).get("quant_method")
        print(f"{m.id:58s} type={str(c.get('model_type')):12s} quant={str(qc):10s} downloads={m.downloads}")
