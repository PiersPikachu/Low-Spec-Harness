# Notice - gpt-oss vaults

The vaults in this folder and on the release `gpt-oss-vaults-v1` are derived from:

- **openai/gpt-oss-20b** - https://huggingface.co/openai/gpt-oss-20b
- **openai/gpt-oss-120b** - https://huggingface.co/openai/gpt-oss-120b

Copyright © OpenAI. Licensed under the **Apache License, Version 2.0** (a copy is in `LICENSE` at the top of this repository,
also at http://www.apache.org/licenses/LICENSE-2.0). OpenAI also publishes a usage policy for these models with the original
release; follow it when you use them.

**Modifications (Apache 2.0 §4(b)).** The original safetensors weights were repackaged, without changing any weight value:

- bf16 tensors (attention, router, embeddings, norms, output layer): Huffman-coded, lossless;
- MXFP4 expert tensors: the released FP4 blocks and E8M0 scales repacked in the MXF2 layout (lossless);
- written as one `vault.bin` stream file with an index (`vault.idx`, `vault.json`), and published in parts under 2 GiB.

The tokenizer, chat template and generation configuration files are copied unchanged from the original repositories.
The conversion code is in `tools/pack_gptoss.py` (packer) and `engine/` (the engine that reads the vaults).
