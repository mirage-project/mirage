# M2-I6 — attention probes P3/P4 and the acceptance evidence

Owner issue: **M2-I6** (kernel wave #4 — adapt MPK's SM100 attention for Qwen3.5's
full-attention layers). Specs: `docs/qwen35/v1-architecture.md` §4 and §14 rows P3/P4;
semantics: `docs/qwen35/vllm-graph.md` §2.2.

Everything below is a saved measurement. Where the architecture doc and HF disagree in
low-order bits, **HF wins** — so each ruling is stated with the counterfactual that was
asserted to miss.

## Contents

| file | what |
|---|---|
| `p3_attn_smem.cu`, `run_p3.py` | probe P3 — the smem instantiation sweep |
| `p3_smem_sweep_pre_pick.json`, `p3_smem_sweep_post_pick.json` | P3 results, with and without the `5715c6f` cherry-pick |
| `p4_rope_perm.py`, `p4_rope_perm_result.json` | probe P4 — partial-RoPE column-permutation exactness |
| `oracle_result.json` | per-intermediate numerics vs the M2-I3 HF oracle |
| `qloop_result.json` | `max_tokens_per_pass` Q-loop equivalence |
| `qwen3_8b_codegen_byte_identity.json` | the INTERMEDIATE Qwen3-8B codegen byte-identity proof |
| `qwen3_8b_ci_run.log` | the Qwen3-8B CI run (`tests/ci-tests/run_ci_tests_qwen3.sh`) |

The kernel tests that produce `oracle_result.json` / `qloop_result.json` live in
`tests/runtime_python/blackwell/sm100_attention_qwen35/`.

## Provenance of the cherry-pick

`5715c6f2a6cce5d0d18da4e6776332b6ad04d7e4` — *"Fix SM100 attention smem overflow blocking
GQA >= 8:1 (#702) (#739)"* — reachable from `origin/pr-dsv3-v1` and `origin/pr-skills`, not
an ancestor of `qwen3-5_support`. It applies to
`include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh` with zero conflicts
(87 insertions / 69 deletions, matching the upstream commit exactly) and is included in the
M2-I6 working tree. It makes `S_O_BUFFER` independent of `MAX_TOKENS × NUM_QO_PER_KV` by
chunking the cross-warp output reduction one MMA m-tile at a time; P3 below quantifies what
that buys at our shape.

Two further defects in the same kernel, invisible below head_dim 128, were found by this
issue and are fixed on top of the pick — see "Latent head_dim-256 bugs" at the end.

## P3 — smem instantiation sweep (validates §4.3's pass size)

Compiles a bare TU that instantiates the attention task at the Qwen3.5 full-attention
shape (16 Q / 2 KV heads → `NUM_QO_PER_KV = 8`, `head_dim = 256`) once per `MAX_TOKENS`
value, and lets the kernel's own
`static_assert(S_TOTAL_OFFSET <= MAX_DYNAMIC_SHARED_MEMORY_SIZE)` decide. Budget is
201 KiB = 205824 B (`runtime_header.h`, `MPK_TARGET_CC=100` + `MODE_OFFLINE`).

| MAX_TOKENS | arena, pre-pick | pre-pick | arena, post-pick | post-pick |
|---|---|---|---|---|
| 1 | 174128 | COMPILES | 174128 | COMPILES |
| 2 | 182320 | COMPILES | 182320 | COMPILES |
| 4 | 233520 | STATIC_ASSERT | 200752 | **COMPILES** |
| 8 | 335920 | STATIC_ASSERT | 237616 | STATIC_ASSERT |
| 16 | 540720 | STATIC_ASSERT | 311344 | STATIC_ASSERT |

**Verdict: exactly §14's expected discriminating outcome** — COMPILES at 1/2/4,
STATIC_ASSERT at 8. Pass size is **4**, and the cherry-pick is load-bearing: it moves the
largest admissible `MAX_TOKENS` from 2 to 4.

The arena numbers are read out of the compiler (an incomplete class template
instantiated on the size, so nvcc prints the integer), and the runner cross-checks that
the mirrored arena model predicts the same COMPILES/STATIC_ASSERT boundary the real
kernel produced — it does, with zero mismatches on both trees.

## P4 — partial-RoPE column permutation (gates §4.4's zero-kernel-change route)

Qwen3.5 rotates only dims `[0:64]`, pairing `(j, j+32)`; MPK's kernel rotates the full
256 pairing `(i, i+128)`. The load-time permutation maps Qwen's rotated pairs onto MPK's
and pads the cos/sin table with `cos=1, sin=0` elsewhere
(`demo/qwen3_5/rope_permutation.py`).

Measured on synthetic tensors **and on the real oracle dumps** (decode + prefill):

| check | result |
|---|---|
| algebra, fp32 (shared permutation-invariant RMS scale) | **max abs diff 0.0, bit-exact, all 5 cases** |
| as-run fp32 (each path does its own reduction) | 0.0 |
| bf16, kernel rounding order | **0 ULP** |
| rebuilt cos/sin table vs HF's dumped table | bit-exact in bf16 |
| padding slots are exactly `cos=1`, `sin=0` | true |

**Verdict: `PERMUTATION_ROUTE_GO`** — the kernel is untouched for RoPE; the `ROTARY_DIM`
template fallback is not needed.

Recovered positions confirm the setup independently: prefill `0..7`, decode `8`, matching
the oracle's own description (θ = 1e7, `rotary_dim = 64`).

## Per-intermediate numerics vs the HF oracle

Layer 3 (first full-attention layer), 8-token prefill + the following decode step.
Full table in `oracle_result.json`.

| intermediate | prefill | decode |
|---|---|---|
| kv cache **v** (raw, no norm/rope) | **bit-exact** 0/4096 | **bit-exact** 0/4608 |
| k norm+rope vs kernel's own arithmetic reference | **bit-exact** 0/4096 | **bit-exact** 0/512 |
| k norm+rope vs oracle | 1131/4096, max 6.25e-2 | 149/512, max 3.13e-2 |
| q norm+rope vs oracle (all 16 heads) | 9280/32768, max 6.25e-2 | — |
| **attention out from pre-roped q/k** (norm+rope bypassed) | **bit-exact 0/32768** | **bit-exact 0/4096** |
| attention out (UNGATED, kernel does norm+rope) | 15234/32768, max 2.34e-2 | 1586/4096, max 1.56e-2 |
| **gate epilogue in isolation** | **bit-exact 0/32768** | **bit-exact 0/4096** |
| gated out vs oracle | max **1.95e-3**, mean 9.1e-6 | max **9.77e-4**, mean 4.6e-6 |

### The residual is fully attributed, not merely bounded

Feeding HF's own post-rope q/k makes the kernel's attention **bit-exact** against the
oracle, so the attention math itself carries no error. The remaining q/k difference was
chased to its mechanism by swapping one factor at a time:

| reference | vs oracle |
|---|---|
| MPK norm order + MPK rope rounding | 1131/4096 |
| HF norm order + MPK rope rounding | 1131/4096 (norm association changes nothing) |
| HF norm order + HF rope rounding | 1090/4096 |
| **fp32 norm weight + HF rope rounding** | **bit-exact 0/4096** |

So the entire residual is (a) MPK taking the folded Gemma norm weight as **bf16** —
`k_norm_weight_ptr` is `T const *` for every model, not a Qwen3.5 choice — and (b) HF
rounding each RoPE product to bf16 before summing while MPK keeps both in fp32 and rounds
once. Nothing is unexplained. Whether to carry the norm weight in fp32 is a
representation change affecting every model and belongs to M2-I9's full-model AC-3 gate,
not to this kernel wave.

### Counterfactuals asserted to MISS

* **Output-gate rounding.** HF evaluates `torch.sigmoid(gate)` on a bf16 tensor, so the
  sigmoid is rounded to bf16 *before* the multiply. The kernel does the same and is
  bit-exact. The "more accurate" fp32-folded variant misses on **1084/4096** (decode) and
  **8974/32768** (prefill) elements — implementing it would have been wrong.
* **QKVG packing.** `q_proj` packs `[q|gate]` *per head*; the plausible
  "first 4096 = q, second 4096 = gate" block split does not reproduce `attn.q_split`.
* **Norm width.** RMSNorm is over the full 256 dims; normalising over `rotary_dim = 64`
  is off by 4.4–4.5 absolute.

## Q-loop equivalence (`qloop_result.json`)

At the Qwen3.5 shape no unsplit reference exists (P3: `MAX_TOKENS=8` does not fit), so
the equivalence is proved at 4 Q / 1 KV head, head_dim 128, where a single 8-row pass
does exist:

* split runs (`2x4`, `4x2`, and the production `arena 4 / 2x4` form) are **bit-identical**
  to the unsplit 8-row reference for output, KV-k and KV-v, across T = 8/5/1 and
  seq = 8/40/37, gated and ungated;
* at the Qwen3.5 shape, pass sizes 4/2/1 agree bit-for-bit with each other;
* **cross-pass causal coupling** holds — a query in pass 1 still attends keys contributed
  by pass 0 — and the truncated-to-pass-1 counterfactual differs (1.02e+00), so the test
  can actually discriminate a per-pass causal bug.

Test mode adds the load-bearing counterfactual end to end: at `mbt = 8` the graph
**fails to build without** `max_tokens_per_pass` (the kernel's smem assert) and builds
**with** it.

## Byte-identity of existing models' codegen

`qwen3_8b_codegen_byte_identity.json`: the Qwen3-8B graph emitted by a pristine
`892f466` build and by the M2-I6 build are byte-identical —
`test_rank0.cu` sha256 `357c7151…53cb` and `task_graph_rank0.json` sha256
`d7c74138…6fd0` on both sides.

Positive control (so the identity cannot be a stale-binary artifact): `strings` on the
rebuilt `core.so` finds **both** emission format strings — the legacy 11-template-argument
one and the new 13-argument one — where a pristine build contains only one. Test mode
shows the two branches directly:

```
default : ...task_impl<bfloat16, 8, 1, 512, 5120, 4096, 256, 64, 64, 0, 0, 4>
gated   : ...task_impl<bfloat16, 8, 1, 512, 9216, 4096, 256, 64, 64, 0, 0, 4, 1, 4>
```

The Qwen3-8B CI is green at this state: token-equality `1 passed`, MPK **3.886 ms/token**
vs torch 17.371 (4.47x) — `qwen3_8b_ci_run.log`. The *final* CI proof remains M2-I9's.

## Latent head_dim-256 bugs found and fixed

Both were pre-existing, silent (wrong numbers, no crash, fully deterministic), and could not
fire below head_dim 128 — Qwen3.5 is the first model on this kernel with head_dim 256.

1. **Fused norm+RoPE read the wrong partner column.** `rms_norm_sm100` interleaved the
   rotation into the normalisation loop. NeoX pairs column `i` with `i ± HEAD_DIM/2`, so
   when `HEAD_DIM > NUM_THREADS` (256 > 128) each thread owns two columns and the loop runs
   twice: the first trip read a partner that had not been normalised yet, the second read a
   partner that had already been rotated. Fixed by normalising the whole head first, then
   staging the rotated values in registers between two barriers. The same pattern in
   `rotary_embedding_hopper` (instantiated unconditionally by this task) is fixed the same
   way. `rotary_embedding_sm100.cuh` has the identical shape but **no callers**, so it was
   left alone.

2. **`s_o_buffer` per-thread stride was hard-coded to 64 floats** while an m-tile's
   accumulator is `(HEAD_DIM/16)*8` floats — exactly 64 at head_dim 128, but **128** at 256,
   so every thread overran its neighbour's slice. Widening the buffer would have cost 32 KiB
   and pushed even `MAX_TOKENS=1` over budget, so the reduction is now chunked over output
   n-tiles as well as m-tiles, keeping the slice at 64 floats for any head_dim.

Both fixes are the same sequence of operations at head_dim ≤ 128 (single trip / single
n-chunk), which is why the arena in the P3 table is unchanged and Qwen3-8B stays exact.

Follow-up for M3, not a correctness issue: at head_dim 256 with `MAX_TOKENS=4` the register
accumulator is `MMA_ITERS_M(2) × 16 × 8 = 256` floats per thread, which will spill. The
Q-loop makes the pass size a free tuning parameter (1/2/4 all compile), so the right value
is a measurement, not a constant.

## Reproducing

```bash
# P3 (host-side nvcc only, no GPU)
export PATH=/usr/local/cuda-12.8/bin:$PATH
python3 run_p3.py --tree <mirage> --label post_pick --out p3_smem_sweep_post_pick.json

# P4 (torch, CPU)
python3 p4_rope_perm.py --oracle-dir ~/mpk-qwen35/oracle-work/dumps

# kernel tests (GPU; claim a lock first, see resources.md)
cd tests/runtime_python/blackwell/sm100_attention_qwen35
CUDA_HOME=/usr/local/cuda-13.0 python setup.py build_ext --inplace   # must match torch's CUDA
python test_attention_qwen35_oracle.py --oracle-dir ~/mpk-qwen35/oracle-work/dumps
python test_attention_qwen35_qloop.py
python test_attention_qwen35_testmode.py
```
