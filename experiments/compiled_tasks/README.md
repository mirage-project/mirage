# B200 Compiled Tasks

## Accomplished

- Ported Mirage's SM100 muGraph backend from standalone kernels to inline MPK task bodies (`emit_device_body`/`TASK_GENERATED`). The runtime now builds TMA descriptors, passes operand-indexed pointers through task descriptors, and preserves graph dependencies.
- Adapted code generation to B200's BF16 1-SM `tcgen05`/TMEM path: CUTLASS-derived UMMA layouts, planned shared-memory swizzles and alignment, pipelined TMA loads, barriers/proxy fences, persistent-worker-safe TMEM reuse, and `swapAB` for decode-sized token dimensions.
- Generated and chained linear, fused activation, RMSNorm, SwiGLU, and attention-core tasks; integrated fused/separate MLP and hybrid attention into all 28 layers of Qwen3-0.6B.

## Results on B200

### Full-model decode

Each result is throughput in tokens/s; fused MLP and generated attention were tested independently.

| Sequence | Batch | Handwritten | Fused generated MLP | Generated attention |
|---:|---:|---:|---:|---:|
| 128 | 1 | 789.8 | 648.3 | 516.0 |
| 128 | 4 | 3,203.9 | 2,546.5 | illegal memory access |
| 128 | 8 | 6,137.0 | 5,089.5 | 4,090.7 |
| 512 | 1 | 726.7 | 602.0 | invalid token IDs |
| 512 | 4 | 2,926.9 | 2,379.6 | launch failure |
| 512 | 8 | 5,599.3 | 4,571.3 | 3,547.6 |

At B8/S512, the separate generated MLP reached 4,074 tok/s versus 4,699 fused and 5,582 handwritten. Compiled-attention scaling was approximately linear:

| Compiled attention layers | 0 | 7 | 14 | 21 | 28 |
|---:|---:|---:|---:|---:|---:|
| Throughput (tok/s) | 5,569 | 4,924 | 4,426 | 4,034 | 3,595 |

Enabling both compiled attention and compiled MLP exceeded an integral task/event-ID bound at roughly 19,000 tasks.

### Complete-model prefill

Throughput is flattened prompt tokens divided by TTFT, in tokens/s, and is shown to one decimal place. TTFT includes prefill and first-token production; compilation is excluded. Attention is handwritten in every mode.

| Prompt | Batch | Flattened tokens | Handwritten | Fused generated MLP | Separate generated MLP |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 16 | 6,525.6 | 5,873.9 | 5,426.6 |
| 16 | 2 | 32 | 10,232.6 | 9,412.6 | 9,003.1 |
| 16 | 4 | 64 | 12,312.8 | 12,358.5 | 11,516.0 |
| 32 | 1 | 32 | 7,564.9 | 7,505.5 | 7,052.3 |
| 32 | 2 | 64 | 10,240.0 | 10,760.4 | 9,924.8 |
| 64 | 1 | 64 | 7,576.3 | 7,902.0 | 7,566.2 |

Geometric mean: fused was 1.7% slower than handwritten; separate was 8.4% slower than handwritten and 6.6% slower than fused. The P32/B2 fused value is a same-GPU retry replacing a transient 400.8 tok/s outlier. Flattened-token configurations `(P32,B4)`, `(P64,B2)`, and `(P128,B1)` all failed at 128 tokens because handwritten attention exceeded the SM100 shared-memory limit.

Generated-task numerical error was approximately `1e-3`–`1e-2`, and complete-model execution produced valid tokens. The remaining gap comes from generated-task pipeline overhead, decode-unfriendly tile floors, intermediate memory traffic, and attention compiled for the maximum sequence length.

## Current Limit

Generated attention is decode-oriented and computes only the last query row, so full prefill still uses handwritten attention; prefill above 64 flattened tokens also exceeds the current handwritten SM100 attention shared-memory limit.
