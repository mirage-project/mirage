# GLM-4.6 (Glm4MoeForCausalLM) demo

Runs [zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6) with the Mirage
persistent megakernel. Graph construction is delegated to
`mirage.mpk.models.glm4_moe.Glm4MoeBuilder`; `demo.py` handles the tokenizer,
meta tensors, the decode loop, and an optional HuggingFace reference run.

## Architecture

GLM-4.6 is a 355B-parameter MoE model: 92 decoder layers, hidden 5120, 96 Q /
8 KV heads (GQA), per-head q/k RMSNorm, **partial RoPE** (rotary_dim 64 of 128,
theta 1e6), attention bias on q/k/v. The first 3 layers are dense; the rest are
MoE with 160 routed experts (top-8, sigmoid routing + correction bias,
routed_scaling_factor 2.5) and 1 shared expert.

## Usage

```bash
# Mirage megakernel over the full checkpoint (needs a B200/H100-class GPU with
# enough HBM for the loaded layers):
python demo/glm4_moe/demo.py --model-path /path/to/GLM-4.6 --use-mirage \
    --prompt "Give me a short introduction to large language models."

# Functional smoke test on a slice of layers (the builder builds whatever
# layer indices are present in the loaded state dict):
python demo/glm4_moe/demo.py --model-path /path/to/GLM-4.6 --use-mirage \
    --layers 0-4 --max-new-tokens 8

# HuggingFace reference (no Mirage) for a parity check:
python demo/glm4_moe/demo.py --model-path /path/to/GLM-4.6 --max-new-tokens 16
```

Key flags: `--layers` (subset, e.g. `0-4` or `0,3,5`), `--page-size` (must be a
multiple of 64), `--max-new-tokens`, `--profiling` + `--trace-name` (Perfetto
trace), `--save-tokens` (dump token ids/text/latency to JSON).

## Status / limitations (v1)

- Decode-only, `world_size == 1`, BF16 weights.
- Skips the MTP nextn layer (`num_nextn_predict_layers == 1`).
- Hidden-state RMSNorm uses eps 1e-6 vs GLM's 1e-5 (negligible; the q/k norms
  use the exact 1e-5). Correctness is validated in
  `tests/runtime_python/blackwell/sm100_glm4_moe/` — a 4-layer end-to-end run
  matches HuggingFace at cosine 0.9999 with an identical argmax token.
- The full 355B model needs substantial HBM; use `--layers` for smoke tests on
  a single GPU.
