# B200 Compiled Tasks

## Accomplished

- Ported Mirage's SM100 muGraph backend from standalone kernels to inline MPK task bodies (`emit_device_body`/`TASK_GENERATED`). The runtime now builds TMA descriptors, passes operand-indexed pointers through task descriptors, and preserves graph dependencies.
- Adapted code generation to B200's BF16 1-SM `tcgen05`/TMEM path: CUTLASS-derived UMMA layouts, planned shared-memory swizzles and alignment, pipelined TMA loads, barriers/proxy fences, persistent-worker-safe TMEM reuse, and `swapAB` for decode-sized token dimensions.
- Generated and chained linear, fused activation, RMSNorm, SwiGLU, and attention-core tasks; integrated fused/separate MLP and hybrid attention into all 28 layers of Qwen3-0.6B.

## Results on B200

### Full-model decode

The complete MLP compute was measured with three generated-task partitions.
The three-task baseline materializes the outputs between gate+up projections,
SiLU-multiply, and down projection. The two partial-fusion experiments instead
fuse either gate+up projections with SwiGLU ("up+SiLU") or SwiGLU with the down
projection ("SiLU+down"). Residual addition remains handwritten in every case.

| Decode context length | Batch size | Handwritten baseline (tok/s) | Three-task MLP (tok/s) | Up+SiLU fused (tok/s) | SiLU+down fused (tok/s) | Compiled attention + three-task MLP (tok/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 789.8 | 370.9 | 421.7 | 273.1 | 276.3 |
| 128 | 4 | 3,203.9 | 1,489.9 | 1,675.1 | 1,090.0 | - |
| 128 | 8 | 6,137.0 | 2,942.3 | 3,294.7 | 2,143.8 | 2,232.8 |
| 512 | 1 | 726.7 | 356.8 | 396.5 | 260.1 | - |
| 512 | 4 | 2,926.9 | 1,409.5 | 1,590.8 | 1,050.3 | - |
| 512 | 8 | 5,599.3 | 2,789.1 | 3,120.2 | 2,066.9 | 2,024.2 |

Up+SiLU fusion improved the geometric-mean throughput by 12.3% over the
three-task partition, but remained 46.0% below handwritten. SiLU+down fusion
was 26.5% slower than the three-task partition. The latter recomputes each
activation tile for every down-projection output tile, and its 16-CTA output
grid substantially underfills the 148-SM B200. The B8/context-512 up+SiLU
value is an isolated retry replacing one run that produced an invalid token
ID; the retry completed all 4,088 requested tokens with the expected decoded
prefix.

All reported MLP-only cases produced the expected decoded prefix. The requested one-task fused complete MLP is not currently representable: its prototype was rejected by the fused chained-matmul path and planned 562,176 bytes of shared memory, versus the current 205,824-byte task budget. Consequently, neither MLP-only nor attention+MLP has a valid one-task-fused result.

Earlier compiled-attention-only scaling was approximately linear:

| Compiled attention layers | 0 | 7 | 14 | 21 | 28 |
|---:|---:|---:|---:|---:|---:|
| Throughput (tok/s) | 5,569 | 4,924 | 4,426 | 4,034 | 3,595 |

The combined three-task MLP corrected that earlier graph-limit diagnosis: valid combined cases compile and run, while the remaining failures are shape-dependent attention runtime failures.

### Complete-model prefill

Throughput is flattened prompt tokens divided by TTFT, in tokens/s, and is shown to one decimal place. TTFT includes prefill and first-token production; compilation is excluded. Attention is handwritten in every mode.

The earlier generated-MLP measurements generated only the gate/up/SwiGLU
front half and kept the down projection handwritten:

| Prompt | Batch | Flattened tokens | Handwritten | Fused generated front half | Separate generated front half |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 16 | 6,525.6 | 5,873.9 | 5,426.6 |
| 16 | 2 | 32 | 10,232.6 | 9,412.6 | 9,003.1 |
| 16 | 4 | 64 | 12,312.8 | 12,358.5 | 11,516.0 |
| 32 | 1 | 32 | 7,564.9 | 7,505.5 | 7,052.3 |
| 32 | 2 | 64 | 10,240.0 | 10,760.4 | 9,924.8 |
| 64 | 1 | 64 | 7,576.3 | 7,902.0 | 7,566.2 |

Geometric mean: fused was 1.7% slower than handwritten; separate was 8.4% slower than handwritten and 6.6% slower than fused. The P32/B2 fused value is a same-GPU retry replacing a transient 400.8 tok/s outlier. Flattened-token configurations `(P32,B4)`, `(P64,B2)`, and `(P128,B1)` all failed at 128 tokens because handwritten attention exceeded the SM100 shared-memory limit.

The complete generated MLP includes the down projection. The three compiled
partitions are defined as follows; in particular, both sides of either fusion
boundary remain compiler-generated:

- Three-task: `[gate+up projections] -> [SwiGLU] -> [down projection]`.
- Up+SiLU two-task: `[gate+up projections+SwiGLU] -> [down projection]`.
- SiLU+down two-task: `[gate+up projections] -> [SwiGLU+down projection]`.

The benchmark driver now treats each requested token count as the prompt length
for each sequence; batch is the number of prompts, so the flattened workload is
`prompt_length * batch`. The table below records the earlier completed sweep,
which used flattened token count as its primary axis (`Prompt/request` shows the
corresponding per-sequence length):

| Total tokens | Batch | Prompt/request | Handwritten (tok/s) | Three-task compiled (tok/s) | Up+SiLU two-task (tok/s) | SiLU+down two-task (tok/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 32 | 7,564.9 | 5,842.7 | 6,082.5 | 3,638.4 |
| 32 | 2 | 16 | 10,232.6 | 6,820.4 | 7,400.3 | 4,128.2 |
| 32 | 4 | 8 | 11,903.5 | 7,577.4 | 8,280.8 | 4,352.3 |
| 32 | 8 | 4 | 12,979.1 | 8,119.1 | 8,945.3 | 4,629.9 |
| 32 | 16 | 2 | 13,515.3 | 8,202.1 | 8,886.8 | 4,540.3 |
| 64 | 1 | 64 | 7,576.3 | 6,645.3 | 6,861.1 | 3,989.7 |
| 64 | 2 | 32 | 10,240.0 | 8,094.5 | 8,906.0 | 2,977.0 |
| 64 | 4 | 16 | 12,312.8 | 9,291.0 | 8,110.7 | 4,837.7 |
| 64 | 8 | 8 | 13,191.0 | 10,467.4 | 7,201.2 | 5,038.7 |
| 64 | 16 | 4 | 12,335.4 | 9,820.2 | 10,987.8 | 4,921.0 |

Across these ten points, up+SiLU improved geometric-mean throughput by 1.2%
over three-task, while remaining 26.5% below handwritten. SiLU+down was 46.6%
slower than three-task and 61.2% slower than handwritten. Its activation
recomputation and 16-CTA output grid remain a poor fit for the 148-SM B200.
P32/B1 up+SiLU is an isolated retry replacing a post-compilation hang; P16/B1
SiLU+down is an isolated retry replacing a run whose first token differed from
the deterministic reference. Both retries completed with the expected output.

In that earlier flattened-token sweep, every requested `(tokens, batch)` pair
at 128, 256, 512, and 1,024 tokens was compiled through the common
handwritten-attention path. All 20 combinations failed its `S_TOTAL_OFFSET`
shared-memory assertion before reaching the MLP. Consequently, those
complete-model points are unsupported for all four MLP modes and have no valid
throughput result.

Generated-task numerical error was approximately `1e-3`–`1e-2`, and complete-model execution produced valid tokens. The remaining gap comes from generated-task pipeline overhead, decode-unfriendly tile floors, intermediate memory traffic, and attention compiled for the maximum sequence length.

## Current Limit

Generated attention is decode-oriented and computes only the last query row, so full prefill still uses handwritten attention. Complete-model prefill above 64 flattened tokens exceeds the current handwritten SM100 attention shared-memory limit for every tested batch size.
