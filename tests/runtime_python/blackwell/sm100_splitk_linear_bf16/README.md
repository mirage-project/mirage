# `splitk_linear_layer` (BF16, sm100) regression matrix

**Bug summary (confirmed 2026-05-02):** `splitk_linear_layer` hangs the MPK
runtime when the per-task `BATCH_SIZE` template parameter is **< 16**. The
kernel itself enters `linear_sm100_mpk_task_impl` (workers print
`[worker] _execute_task EXECUTE_TASK 251`) and never returns. The hang is
reproducible across both `accumulate=True` and `accumulate=False` and across
shape configurations.

| Per-task BATCH_SIZE | accumulate=True | accumulate=False |
|---:|---|---|
| 1  | TIMEOUT | TIMEOUT |
| 2  | TIMEOUT | TIMEOUT |
| 4  | TIMEOUT | TIMEOUT |
| 8  | TIMEOUT | TIMEOUT |
| 12 | TIMEOUT | TIMEOUT |
| **16** | **PASS** | **PASS** |

`BATCH_SIZE = output_ops[0]->output_tensors[0].dim[0]` (per-task tile batch
dim). It comes from the layer's `output` DTensor's first dim, which in
test_mode equals `max_num_batched_tokens`.

## Why this surfaced now

The qwen3 BUILDER (`python/mirage/mpk/models/qwen3/builder.py:381,496`)
calls `splitk_linear_layer(accumulate=True)` on B200 — but the qwen3
**demo** (`demo/qwen3/demo.py`) does NOT use that builder; it constructs
its own graph using `linear_layer` and `linear_with_residual_layer`. So
`splitk_linear_layer` (BF16) was never actually exercised in production
end-to-end. The bug only became visible when the DSv3 builder was wired to
use it for the MoE router gate (`accumulate=False`, batch_size=1 in
single-token decode).

The FP8 splitk path (`linear_splitk_swapAB_fp8_layer`,
`linear_fp8_swapAB_sm100_task_impl`) is a **different kernel** and works
correctly across batch sizes — it's not affected by this bug.

## Workaround in production

`python/mirage/mpk/models/deepseek_v3/builder.py` gates the gate-splitk
replacement behind `_BF16_GATE_SPLITK_ENABLED = False`. The other 5
DSv3 splitk replacements use the FP8 path, which is unaffected.

## How to run the matrix

```bash
CUDA_VISIBLE_DEVICES=<free-gpu> python run_matrix.py --timeout 90
```

Cells time out at the supplied limit; sweep total stays bounded. When the
underlying kernel bug is fixed:

1. Re-run the matrix — all cells should turn green.
2. Flip `_BF16_GATE_SPLITK_ENABLED = True` in the DSv3 builder.
3. Re-run the DSv3 smoke (`demo/deepseek_v3/demo.py --layers 0-8 --mtp 0
   --max-num-batched-tokens 1 --max-seq-length 16 --max-new-tokens 8`) to
   confirm no regression.

## Files

- `pytorch_reference.py` — canonical reference for BF16 splitk semantics.
- `test_splitk_linear_bf16_testmode.py` — parametric single-config runner
  (one cell per process invocation; CLI args).
- `run_matrix.py` — driver that spawns each `(shape × accumulate)` cell
  as a subprocess with a hard timeout and prints a result table.
- `test_splitk_linear_bf16_accfalse_testmode.py` — minimal repro at the
  DSv3 gate shape with `accumulate=False`. Expected to TIME OUT until the
  kernel is fixed.
