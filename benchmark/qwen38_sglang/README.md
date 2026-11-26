# Qwen3.8-27B fixed-batch decode comparison

`compare_decode_three.py` reproduces the draft PR's MPK, HF eager, stock
SGLang, and vLLM comparison at B4/B8/B16/B32. It runs each backend in its own
Python environment on one B200 with the same 128-token prompts and 64 output
tokens. SGLang and vLLM are external packages; this repository contains only
the comparison harness.

Prepare shared token IDs once with the checkpoint tokenizer:

```bash
/path/to/hf-python benchmark/qwen38_sglang/compare_decode_three.py \
  --backend prepare --model /path/to/Qwen3.8-27B \
  --output /path/to/inputs.json
```

For each batch `B` in `4 8 16 32`, run the following commands sequentially on
the same GPU. Use the MPK environment for `mpk` and separate environments with
the respective external packages for the other backends.

```bash
CUDA_VISIBLE_DEVICES=0 /path/to/mpk-python benchmark/qwen38_sglang/compare_decode_three.py \
  --backend mpk --batch "$B" --model /path/to/Qwen3.8-27B \
  --inputs /path/to/inputs.json --output "/path/to/mpk-b${B}.json"
CUDA_VISIBLE_DEVICES=0 /path/to/hf-python benchmark/qwen38_sglang/compare_decode_three.py \
  --backend hf --batch "$B" --model /path/to/Qwen3.8-27B \
  --inputs /path/to/inputs.json --output "/path/to/hf-b${B}.json"
CUDA_VISIBLE_DEVICES=0 /path/to/sglang-python benchmark/qwen38_sglang/compare_decode_three.py \
  --backend sglang --batch "$B" --model /path/to/Qwen3.8-27B \
  --inputs /path/to/inputs.json --output "/path/to/sglang-b${B}.json"
CUDA_VISIBLE_DEVICES=0 /path/to/vllm-python benchmark/qwen38_sglang/compare_decode_three.py \
  --backend vllm --batch "$B" --model /path/to/Qwen3.8-27B \
  --inputs /path/to/inputs.json --output "/path/to/vllm-b${B}.json"
```

The default is one warmup and three timed trials. Each trial times the 63
decode intervals after the first generated token. Model loading, compilation,
engine startup, and prefill are excluded. MPK and HF use host-observed decode
steps; SGLang and vLLM use client-visible streamed token arrivals. Compare
output IDs against HF before interpreting the timing. The draft PR reports
exploratory numbers only; MPK decode performance remains far from optimal.
