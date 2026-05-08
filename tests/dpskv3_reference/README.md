# DeepSeek V3 PyTorch Reference

A vLLM-aligned, plain-PyTorch reference implementation of DeepSeek V3
(with the MTP head from the official checkpoint) used as the **immutable
source-of-truth** for verifying MPK's DeepSeek V3 builder + kernels.

## Why this exists

MPK's regression suite (`scripts/regression_test.sh`) verifies "the
megakernel runs without crashing and prints a per-token latency line".
It does **not** verify that the produced token IDs or per-layer hidden
states are numerically correct.

This reference closes that gap. It runs the same forward pass MPK does,
but in plain PyTorch, with operations that exactly mirror vLLM's
implementation (down to the order of operations and the
`mscale^2`-in-softmax-scale subtlety). It dumps:

- per-iteration embedding output
- per-layer pre-norm / post-attn / post-MLP hidden states
- final RMSNorm output
- logits + argmax token IDs
- (when MTP is enabled) MTP eh_proj output, MTP decoder output, MTP
  final norm, MTP draft token IDs per spec step

The MPK side has a parallel dump path that writes the same tensor names
to disk. A comparator script diffs them with cosine + max-abs-diff
metrics.

## Immutability contract

**Do not edit the math in `modeling.py` without independently
re-validating against the cited vLLM source.** Each module/function has
a docstring with `vllm/...:line-range` citations. If the math in this
file diverges from vLLM, MPK can be "verified correct" against a wrong
reference, which is worse than having no reference.

The acceptable changes:

- **Comments / docstrings** — clarification, additional citations
- **Performance** — only if the optimised path produces bit-identical
  output (same float ordering)
- **Weight loading / dump format / runner orchestration** — yes, these
  are not part of the math contract
- **New tracking points** — adding new hidden-state dump locations is
  fine; renaming or removing them needs a coordinated MPK-side update

The unacceptable changes:

- Changing operation order in `decoder_forward()` (input_layernorm →
  attn → residual → post_attn_layernorm → mlp → residual). Has been
  verified against `vllm/model_executor/models/deepseek_v2.py:1119-1166`.
- Changing the softmax scale. The factor is `(1/sqrt(192)) * mscale**2`,
  not `(1/sqrt(192)) * mscale` or `(1/sqrt(128))`. See
  `vllm/model_executor/models/deepseek_v2.py:524-525`.
- Changing the MTP `eh_proj` formulation to be a sum of two matmuls
  instead of `Linear(concat([enorm_out, hnorm_out]))`. The MPK builder
  uses two matmuls because `Linear([x;y]) = x @ W[:H]^T + y @ W[H:]^T`
  is mathematically identical, but the **reference uses concat-then-matmul**
  to match vLLM exactly. See `vllm/model_executor/models/deepseek_mtp.py:96-118`.
- Changing the routed-experts MoE topk from sigmoid + correction-bias +
  grouped (n_group=8, topk_group=4, topk=8) to plain softmax-topk.

## File layout

| File | Purpose |
|---|---|
| `config.py` | DeepSeek V3 architecture constants (qk_nope_head_dim, kv_lora_rank, num_routed_experts, etc.) |
| `modeling.py` | The vLLM-aligned PyTorch modules (RMSNorm, YaRN RoPE, MLA, Dense MLP, MoE, MTP, DecoderLayer, Model) |
| `loader.py` | Selective weight loading from HF safetensors with FP8 dequantization |
| `runner.py` | Forward-pass orchestration + per-layer hidden state dumping |
| `comparator.py` | Diff a reference dump against an MPK dump (cosine + max-abs-diff per tensor) |
| `test_dpskv3_reference.py` | Pytest smoke that builds tiny config + runs forward + checks shape + checks no NaN |
| `README.md` | This file |

## Three test cases

The reference must produce correct outputs for all three:

1. **No MTP** — main model only. `enable_mtp=False`.
2. **Prefill + MTP** — main model runs over the full prompt, then MTP
   runs once over all prefill positions to populate its KV cache and
   produce a draft for position `prompt_len`. `enable_mtp=True,
   prompt_length > 1`.
3. **Decode + MTP** — after a prefill iteration, an autoregressive draft
   chain produces `spec_length` draft tokens; each draft step depends on
   the prior. `enable_mtp=True`, multiple iterations.

## Usage

```python
from tests.dpskv3_reference.runner import run_reference

result = run_reference(
    model_path="/raid/catalyst/models/DeepSeek-V3",
    prompt="Hello, world.",
    layers=[0, 1, 2, 3],   # subset for sub-671B testing
    enable_mtp=True,
    spec_length=2,
    max_new_tokens=4,
    dump_dir="outputs/dpskv3_reference_dump_<timestamp>",
)
print(result.token_ids)
```

To compare against an MPK run:

```bash
python -m tests.dpskv3_reference.comparator \
    --reference outputs/dpskv3_reference_dump_20260507_220000 \
    --mpk       outputs/regression_postfix2_*/A_prefill_mtp2_dump
```

## What is NOT covered (by design)

- **Tensor parallelism** — the reference is single-GPU. TP correctness
  in MPK is verified separately by running MPK with TP=1 (no allreduce)
  vs TP>1 and confirming token-level equivalence.
- **Paged KV cache** — the reference uses contiguous KV. MPK's paged
  layout is an implementation detail orthogonal to the math.
- **FP8 attention math** — the reference dequantizes weights to BF16
  upfront. Attention runs in BF16. MPK may use FP8 internally (e.g., FP8
  GEMM in MoE); the dump-and-compare is at hidden-state granularity, so
  small numeric drift from FP8 quantization is expected and shows up as
  a non-zero max-abs-diff (typically &lt; 1e-2 cosine = &gt; 0.999).
- **Speculative decoding verify/reject** — the reference computes both
  target and draft tokens; whether they get accepted / rejected at
  runtime is MPK's job (`mtp_verify_strict_layer` etc.). The reference
  just exposes the draft tokens themselves.

## Citations

vLLM source files this reference is aligned to (snapshot
`/home/muhengl/vllm` as of 2026-05-07):

- `vllm/model_executor/models/deepseek_v2.py` — main model + decoder + MLA + MoE
- `vllm/model_executor/models/deepseek_mtp.py` — MTP layer + shared head
- `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` — MTP runtime / draft loop
- `vllm/model_executor/layers/layernorm.py` — RMSNorm (DeepseekV2RMSNorm uses identical math)
- `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py` — YaRN RoPE

Every non-trivial function in `modeling.py` has a docstring with the
`<file>:<line>` citation. When you read a function and it doesn't make
sense, go to the cited line in vLLM and read that — it will.
