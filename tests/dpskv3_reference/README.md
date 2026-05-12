# DeepSeek V3 PyTorch reference (official-inference-backed)

Built on top of the **official DeepSeek-V3 inference code** at
`/home/muhengl/DeepSeek-V3/inference/` (clone of
[github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)).
The previous in-tree re-implementation has been deleted; this version
defers all math to DeepSeek's own code so we can't have an
implementation-divergence bug masking MPK correctness errors.

## Files

| File | Purpose |
|------|---------|
| `model_wrapper.py` | `DeepseekV3SubsetModel` — composes the official `Block` modules for a requested layer subset and adds per-layer hidden-state recording. Configures the official module's globals (`world_size`, `rank`, `gemm_impl`, `attn_impl`, `Linear.dtype`) for single-rank BF16 execution. |
| `loader.py` | Loads an HF DSv3 safetensors checkpoint, translates parameter names from HF (`q_a_proj`, `gate_proj`, `e_score_correction_bias`, ...) to the official scheme (`wq_a`, `w1`/`w2`/`w3`, `gate.bias`, ...), and dequantizes FP8 → BF16 on the fly. |
| `runner.py` | Orchestration: build subset model, load weights, run prefill+decode, dump per-iter `embed.pt` / `layer_<L>_residual.pt` / `argmax.pt`. |
| `comparator.py` | Diffs reference dump rows vs MPK `--dump-hidden-dir` rows (skip row 0 by default — see `feedback_row0_dump_artifact` memory). |

## Quick start

```python
from tests.dpskv3_reference.runner import run_reference

result = run_reference(
    model_path="/raid/catalyst/models/DeepSeek-V3",
    prompt_length=128,          # synthetic prompt matching MPK's --prompt-length
    layer_indices=[0, 1, 2, 3],
    max_new_tokens=1,
    dump_dir="outputs/dpskv3_ref_official_<ts>",
    verbose=True,
)
print(result.token_ids)
```

Compare against an MPK run:

```bash
python -m tests.dpskv3_reference.comparator \
    --ref outputs/dpskv3_ref_official_<ts>/iter_0000 \
    --mpk outputs/<mpk_run>/dump
```

## What this reference covers / doesn't cover

Covered:
- DSv3 forward math at any layer subset (dense layers 0-2, MoE
  layers 3-60). Layer ID is preserved so `Block` picks MoE-vs-MLP correctly.
- Selective weight loading (only the requested layers' weights are
  read off disk).
- FP8 → BF16 dequantization at load time.
- Per-layer hidden state dumping.

Not (yet) covered:
- Tensor parallelism (single-rank only). For TP correctness checks,
  compare MPK TP=N vs MPK TP=1 separately.
- MTP head. The old reference had this; we'll add when needed.
- Force-accept spec decode. Removed when we deleted the old runner.

## Why this exists

MPK's regression suite verifies "the megakernel runs without crashing
and prints a per-token latency line". It does NOT verify numerical
correctness against the model's intended math.

The OLD in-tree reference re-implemented DSv3 from vLLM citations.
That was risky: any divergence between our re-implementation and the
official model can silently mask MPK bugs (or surface phantom bugs).

The NEW reference uses DeepSeek's own model code directly, eliminating
that re-implementation risk. We only own:
1. A subset-of-layers wrapper (composition, not new math).
2. A name translator + FP8 dequantizer (data wrangling).
3. The forward/dump driver (no math).
