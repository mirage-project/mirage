# M4-I2 — the ferret dense-FP8 winner, integrated

Task 279 `TASK_LINEAR_FP8_BLOCKSCALE_SM100` now runs the ferret
`dense-fp8-blockscale` winner from `~/mpk-qwen35/ferret/workspace4` tag **v011**
(`fed45b8`), which measured `min_ratio` **1.011** over 30 shape/M configs against
vLLM's `cutlass_3x_gemm_fp8_blockwise` — worst `outproj_M2` at 101.1%, best
`gdnz_M4` at 128.1%.

**The workspace worktree was +36 lines past v011** (an unfinished `mbarrier`
probe, plus a `.chain_stop`). This port takes the **v011 tag blob**, not the
worktree file. `git show v011:kernel.cu` → 945 lines, md5
`78189574291323b27ca199a5e0a2af28`.

## What landed, and the one structural difference

The pre-M4-I2 implementation is preserved **byte-for-byte** as
`linear_fp8_blockscale_task_impl_golden` — verified programmatically, 8459 bytes
identical — and `linear_fp8_blockscale_task_impl` became a compile-time
dispatcher over `fast_path_ok()`. The fast path is the v011 body, verbatim modulo
two mechanical renames (v011's vendored `load_smem`/`load_smem_with_predict`,
which carry the `.cg` hint Mirage's shared helpers do not, are called here as
`load_smem_cg`/`load_smem_cg_predict`; the `.cg` helpers are vendored into the
blockscale namespace so no other task's codegen moves).

What the fast path changes: the activation tile is staged once for the whole K
extent; the fp32 scale panels and the bf16 residual move into shared memory in
A's commit group (the golden path issued *dependent global* scale loads inside
the K loop); B streams through a deep **per-warp** cp.async ring, up to full
prefetch, with no block-wide barrier anywhere in the K loop; fragments load via
`ldmatrix`; K tiles are processed in pairs with interleaved MMA streams and
double-buffered fragment registers.

### The structural difference: a per-task N slice finer than the scale block

v011's win needs `OUTPUT_SIZE` **16/32/64** per shape rather than 128. This is
not a micro-optimisation and it is not optional:

| ferret tag | change | `min_ratio` |
|---|---|---|
| v007 | pre-slicing (slice 128) | 0.727 |
| v008 | per-shape N-slicing lands | 0.862 |
| v010/v011 | deep rings, which only *fit* smem at a narrow slice | 1.009 / 1.011 |

The mechanism is MPK-specific: one persistent worker CTA per SM, one task at a
time, so a projection dispatched as N/128 tasks occupies N/128 of the 148 SMs —
**16** of them for the N=2048 `out_proj`/`o_proj` pair. Integrating at slice 128
would have banked roughly the v007 number and left the kernel ~1.3x off vLLM.

**How it was resolved.** MPK splits an input by *integer division*
(`runtime.cc`: `block_size = dim[input_map.x] / grid_dim.x`), so a
`[N/128, K/128]` weight scale under a grid of N/16 gives `block_size == 0` and
**every task silently reads scale row 0** — wrong numbers, no error. So the
builder attaches `weight_scale` **row-replicated** to one row per task
(`Qwen35Builder._fp8_block_scale`): `repeat_interleave` copies the same fp32
values, so the promoted numbers stay bit-identical to the checkpoint's, and the
ordinary grid split then hands each task its containing block row. The whole
dense scale set is ~340 KiB before replication, so the cost is noise.

Three fail-closed guards went in with it, because the failure mode is a silent
wrong answer:

- `linear_fp8_blockscale_layer` asserts `weight_scale.dim(0) == grid.x` and that
  the derived slice is legal. This closes the silent-row-0 hole for **every**
  caller, not just this one.
- `task_register.cc`'s `output_size % 128 == 0` is relaxed to admit a `>= 16`
  sub-multiple, and no further.
- The dispatcher `static_assert`s that *every* instantiation has an admissible
  path — if the fast path rejects it (batch > 16, odd K-tile count, a ring that
  does not fit smem) the golden path must be able to run it, and the golden path
  needs whole 128-row blocks. A builder that asks for a sub-block slice at batch
  64 fails the **build**, not a numeric check at runtime.

Slices live in `builder.py`'s `FP8_DENSE_N_SLICE`, keyed on `(N, K)` — exactly the
six shipped call sites the ferret run benchmarked. Anything else, and any
`max_num_batched_tokens > 16` (where the fast path's per-warp B ring is
inadmissible because it assumes `TILE_M == 16`), falls back to slice 128 + the
golden path. Verified: `mbt=132 -> slice 128` for all six shapes.

| call site | N | K | residual | slice | tasks/launch (was N/128) |
|---|---|---|---|---|---|
| GDN `in_proj_qkv` | 8192 | 2048 | no | 64 | 128 (was 64) |
| GDN `in_proj_z` | 4096 | 2048 | no | 32 | 128 (was 32) |
| attn `qkv(g)_proj` | 9216 | 2048 | no | 64 | 144 (was 72) |
| GDN `out_proj` / attn `o_proj` | 2048 | 4096 | **yes** | 16 | 128 (was 16) |
| shared `gate_up` | 1024 | 2048 | no | 32 | 32 (was 8) |
| shared `down` | 2048 | 512 | no | 64 | 32 (was 16) |

### The A/B arm

`MPK_FP8_DENSE_BASELINE=1` pins slice 128 in the builder **and** passes
`-DMPK_FP8_DENSE_BASELINE=1` to the JIT, making `fast_path_ok()` false. Both arms
therefore come from one tree and interleave inside a single GPU claim. It is a
faithful stand-in, not a strawman — see the HEAD control below.

## Provenance

Isolated clone `~/mpk-qwen35/mirage-m4i2` with its own freshly built C++
extension (STALE-EXTENSION TRAP). `git bundle` was unavailable to the authoring
session, so the commits arrived as a `format-patch` series applied with `git am`;
`git am` rewrites committer dates, so the SHAs differ from the authoring clone's
and the **tree hash** is the content link:
`5e1d0fece281359b5742b1f3314bf4455ccb1cca`, asserted equal at setup.

## Gate 1 — bit-exactness, both nvcc flag lanes

The ferret task's own bar was bit-exactness against the frozen current kernel, so
a faithful port should be bit-exact. It is, and it was proven rather than assumed:
each shipped projection is computed twice from identical inputs — once as N/128
**golden** tasks, once as N/slice **fast** tasks — and the bf16 outputs compared
bitwise, with a `0xEE` poison fill so an unwritten slice cannot pass by matching
a zeroed buffer.

Two data regimes, because they falsify different things. **E** is
exact-by-construction (the ferret harness's own scheme: small-integer fp8, n/8
scales), so bit-exactness is independent of FMA contraction and within-tile
summation order and a mismatch means a real port defect. **R** is deliberately
inexact, so bit-exactness additionally requires identical compiler contraction —
a compiler property, not part of the numerics contract, hence reported rather
than required.

| harness | nvcc | lane | regime E | regime R |
|---|---|---|---|---|
| standalone (torch-free) | **12.8**, the shipped JIT toolchain | no fast-math | 30/30 bit-exact | 0 differing elements, max \|ULP\| 0 |
| standalone | **12.8** | `-use_fast_math` (**what ships**) | 30/30 bit-exact | 0 differing elements, max \|ULP\| 0 |
| pybind | 13.0 (torch-matched) | no fast-math | 30/30 bit-exact | — |
| pybind | 13.0 | `-use_fast_math` | 30/30 bit-exact | — |

30 = 6 shipped shapes x 5 decode batch sizes. Regime R coming out at **zero**
differing elements is stronger than the contract requires.

The pre-existing single-task test also passes in both lanes, including its
scale-consumption negative control, and now covers the dispatcher on both sides
(batch <= 16 takes the fast path, batch 64/256 the golden one).

**Why a second, torch-free harness exists.** The megakernel JIT resolves nvcc off
`PATH` and every driver on this box pins 12.8, but the box's torch is
`2.13.0+cu130` and `torch.utils.cpp_extension` refuses a CUDA-major mismatch — so
the pybind harness can *only* build under 13.0 and cannot certify the shipped
compiler. `scripts/bitexact_standalone.cu` has no torch dependency and closes
that gap.

### Two instrument defects this gate exposed

1. **The pybind harness was compiling against the wrong shared-memory budget.**
   `MPK_TARGET_CC` was undefined, which preprocesses as 0, so `runtime_header.h`
   selected the **163 KiB** fallback instead of B200's real **207 KiB**. Invisible
   while the only path here needed 41 KiB; the fast path needs up to 198 KiB, so
   admissible slices failed to build. Fixed the way
   `sm100_attention_qwen35/setup.py` already had to (`MPK_TARGET_CC=100` +
   `MODE_OFFLINE`) — **12 of the 14 blackwell test setups still lack these**, so
   any other kernel guarding an arena with that constant is being validated
   against a budget it will not get. Left as a note for the coordinator; out of
   this issue's scope to change 12 unrelated harnesses.
2. **Relying on `if constexpr` branch-discarding to suppress a `static_assert` is
   not portable.** nvcc kept parsing the discarded branch while recovering from
   an earlier diagnostic, and surfaced the golden path's `OUTPUT_SIZE % 128`
   assert for a slice the golden path was never going to run. Fixed structurally:
   `golden_output_size()` sanitises the argument so a spuriously-instantiated
   branch still compiles, and the *safety* now rests on the dispatcher-level
   reachability `static_assert`, which no compiler's instantiation eagerness can
   affect. `--expt-relaxed-constexpr` is also stated explicitly rather than
   inherited from torch's defaults.

## Gate 2 — `-Xptxas -v` on the generated megakernel TU

MPK inlines every task into ONE `__global__`, so ptxas allocates a **single**
register budget and stack frame for all of them: a dense-fp8 win that raised
register pressure would tax every other stage, which is how M3-I6a's finding
worked in reverse. The fast path adds live state (paired-tile partials,
double-buffered `a_frag`/`b_frag`), exactly the shape of change that does this.

`persistent_kernel` entry, bs1 TU, shipped JIT flags:

| TU | registers | stack frame | spill stores | spill loads | barriers | smem |
|---|---|---|---|---|---|---|
| pre-M4-I2 HEAD (`22c3f24c`) | 238 | 144 B | **0** | **0** | 16 | 5824 B |
| arm A (`MPK_FP8_DENSE_BASELINE=1`) | 238 | 144 B | **0** | **0** | 16 | 5824 B |
| arm B (**shipped**) | 238 | 144 B | **0** | **0** | 16 | 5824 B |

**No change, and no spill introduced.** The dense-fp8 fast path's own frame
(~40–56 registers) sits well under the megakernel's shared ceiling, which some
other task sets.

The HEAD and arm-A TUs are **byte-identical** — same sha256
`808fbec567e51ba208415c63cdbaf5b1f5a395bc37c4bbdc4123046eeee91cef` — so arm A's
generated code *is* the pre-M4-I2 code, which is what licenses using it as the
e2e baseline.

## Gate 4 — per-bs e2e A/B

Geometry B (the AC-4-shaped primary): synthetic 256-token prompts, `msl=353`, 96
decode steps, `mbt=16`. Three reps per (arm, bs), arms **interleaved per (bs, rep)
inside one GPU claim** so drift or a co-tenant hits both equally. A kernel dir per
(arm, bs) — the slice is a compile-time template argument, and two arms sharing a
`--kernel-dir` under `--reuse-kernel` would run one binary and report themselves
identical (M3-I7 defect 3).

30 runs, all on GPU 2. **0 dirty, 0 unauditable**; observed pinned-device floors
5–7 MiB. The audit derives the pinned device from each run's own recorded
`cuda_visible_devices` + its own `gpu_before`, never from the guard's candidate
list — that substitution is exactly M3-I7's phantom-dirty-rep bug.

| bs | A base (per-rep ms) | B new (per-rep ms) | A med | B med | speedup |
|---|---|---|---|---|---|
| 1 | 1206.8 / 1207.8 / 1207.0 | 1096.1 / 1097.1 / 1096.9 | 1207.0 | 1096.9 | **1.100x** |
| 2 | 1391.0 / 1391.3 / 1393.6 | 1264.9 / 1265.1 / 1270.7 | 1391.3 | 1265.1 | **1.100x** |
| 4 | 1746.3 / 1754.8 / 1753.6 | 1616.8 / 1612.1 / 1623.2 | 1753.6 | 1616.8 | **1.085x** |
| 8 | 2683.3 / 2684.1 / 2681.6 | 2504.1 / 2503.5 / 2495.3 | 2683.3 | 2503.5 | **1.072x** |
| 16 | 4124.6 / 4129.9 / 4144.5 | 3806.8 / 3815.5 / 3832.4 | 4129.9 | 3815.5 | **1.082x** |

Per-rep ranges are 1.0–25.6 ms, i.e. well under 1%.

**HEAD control.** Three reps of the pre-M4-I2 worktree at bs1: 1206.6 / 1207.7 /
1206.9, median **1206.9** vs arm A's **1207.0** — 0.01% apart. The baseline arm
is not a strawman.

**A second correctness signal, for free.** `tokens_sha256` from the two arms
matched in **all 15 (bs, rep) pairs** — the kernel-level gate proved the paths
bit-exact on synthetic tensors, and this is the same property through 40 real
layers on real checkpoint weights.

## Files

```
scripts/bitexact_standalone.cu   Gate 1a: torch-free, shipped nvcc 12.8
scripts/gate1.sh                 Gate 1 driver, both flag lanes, both harnesses
scripts/mk_ptxas_m4i2.sh         Gate 2: head vs armA vs armB, compile-only
scripts/sweep_fp8.sh             Gate 4: the interleaved A/B
scripts/head_control.sh          the pre-M4-I2 worktree control + its TU
scripts/gate_ac3_m4i2.sh         Gate 3: full AC-3, five bs
scripts/stage_wallspan.sh        profiled runs -> concurrency.py
scripts/tables_m4i2.py           A/B tables + the per-run gpu_before audit
scripts/stage_tables.py          stage wallspan + the width residual
tables/                          the generated tables (txt/json/csv)
logs/, raw/                      per-gate logs and reports
```
