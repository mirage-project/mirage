# Exemplar merged PRs (the empirical basis for the norms)

All PRs below are squash-merged commits on `mirage-project/mpk`, sampled from
`git log --before=2026-04-01` across the five change categories (plus a few clean newer
bugfixes). Footprints are `git show --stat` dir-buckets. Find the row closest to your change
and mirror its shape. Each PR's number is the `(#NNN)` in its subject.

## Category A — Model support / demo (the tightest norm)

| PR | Commit | Footprint | What it did NOT touch |
|---|---|---|---|
| #535 Add Llama3 family demo | `e36b734d` | `demo/llama3/{demo.py, models/*.py}` (4 new) + `demo/qwen3/demo.py` (3 lines) + 2 kernel `.cuh` (4 lines each, reuse plumbing) + 4 tests | persistent_kernel.py, persistent_kernel.cuh, multigpu.py, src/kernel, runtime.cc |
| #632 Dynamically Sharded Weight Loader | `3fc06d86` | `demo/qwen3` (2) + `python/mirage/mpk` (1) | everything else |
| #705 clamp TMA box dims (deadlock fix, model-adjacent) | `b293bb6e` | `tma.cuh` + `models/qwen3/builder.py` + `demo/qwen3` | shared runtime/python |

**Norm shown:** a model comes up through `demo/<model>/` + `models/<model>/builder.py`
composing existing generic ops. Kernel reuse is a few-line plumbing touch, not a shared-file
rewrite. (Modern split: builder in `python/mirage/mpk/models/<model>/`, demo + HF reference in
`demo/<model>/`. The `models/<model>/` builder dir was introduced by `#537`.)

### The one model PR that touched `persistent_kernel.py` — and why it's still on-norm
- **#563 Qwen3-30B-A3B MoE demo** (`62239a87`) added ~196 lines to `persistent_kernel.py`, but
  every added method is a **generic op** named by operation: `moe_w13_linear_layer`,
  `moe_silu_mul_layer`, `moe_w2_linear_layer`, `moe_mul_sum_add_layer`, `splitk_linear_layer`,
  `tensor_init_layer`, `moe_topk_softmax_routing_layer`. Zero `qwen3_*` methods. It touched the
  shared file **only** because it introduced genuinely-new reusable task types (with their
  registration + kernels), and it exposed them generically. That is the allowed way to touch the
  shared surface; a `<model>_*`-named method is not.
- Corroboration: at merge-base, all ~60 public `*_layer` methods in `persistent_kernel.py` are
  operation-named (`rmsnorm_layer`, `mla_decode_layer`, `allreduce_layer`,
  `mla_mtp_decode_tp8_layer`, …). None is `qwen3_*`/`deepseek_*`/`unified_*`.

## Category B — New task / kernel

| PR | Commit | Footprint | Note |
|---|---|---|---|
| #605 Add MLA kernel for Blackwell | `f2a7f139` | `tasks/blackwell/*.cuh` + its unit test + `runtime_kernel_wrapper` | Cleanest possible: kernel + test + wrapper, nothing else |
| #546 Blackwell MoE Task + tests | `d3b1fbb5` | `tasks/{blackwell,ampere,hopper}` + tests | Kernel-only; no python runtime surface |
| #514 MPK SM100 Linear Task added | `5162bfbb` | `tasks/` + tests + `src/kernel` (3 registration files) + `tma.cuh` | A **new task type** DOES touch C++ registration + `runtime_header.h` — that is its home, not a violation |

**Norm shown:** a kernel = `.cuh` + unit test + wrapper. If it's a *new task type*, add the
C++ registration coherently (`runtime_header.h` + `src/kernel/{task_register,graph,runtime}.cc`
+ `tma.cuh`), and expose it via a generic `persistent_kernel.py` op — separately/minimally.

## Category C — Runtime / scheduler change (its own PR)

| PR | Commit | Footprint |
|---|---|---|
| #411 Split persistent kernel | `8eeacf22` | **exactly 2 files**: `runtime_header.h` + `persistent_kernel.cuh` |
| #470 Fix acquire/release between workers and schedulers | `09aa22fd` | `utils.cuh` + `persistent_kernel.cuh` + a few tasks + test |
| #557 schedulers: allow calling persistent kernel multiple times | `5e75bb45` | scheduler/runtime surface |
| #612 Refactoring allreduce + perf | `92f0d7b1` | `tasks` + `src/kernel` + `persistent_kernel.py` + `.cuh` + `runtime_header` (single-topic refactor) |

**Norm shown:** runtime changes are **their own** coherent PRs, not smuggled into a model or
kernel diff. `#411` is the archetype — 2 files, one idea. `multigpu.py` is likewise
runtime-owned: the only pre-April PR to touch it is `#614` (NVSHMEM tile-based allreduce), so
model/experiment residue there is off-norm.

## Category D — Perf

| PR | Commit | Footprint |
|---|---|---|
| #720 flatten MLA kv-cache gather + 4-way ILP (~2.3x) | `dd8729b3` | the gather kernel(s) — single-topic |
| #536 Optimize multitoken_paged_attention_hopper + linear_swapAB_hopper | `6cdb8f3e` | the two named kernels |
| #612 allreduce refactor + perf | `92f0d7b1` | (also Cat C) |

**Norm shown:** a perf PR names its target in the subject and touches only that kernel/path. No
env-var lever ships — the win is hard-wired.

## Category E — Bugfix

| PR | Commit | Footprint |
|---|---|---|
| #719 per-token FP8 quantize must not derive batch row from blockIdx.x under MPK | `8b19538b` | **ONE file** (`tasks/blackwell`) |
| #572 Fix misalignment in attention kernel | `16dcdfb3` | the arch-mirrored kernels + 1 test + 1 src |
| #705 clamp TMA box dims (deadlock) | `b293bb6e` | `tma.cuh` + qwen3 builder/demo |
| #722 / #723 fix MoE residual / grid_dim under TP | `8de1cfc2` / `22457652` | the MoE kernel + grid wiring |

**Norm shown:** a bugfix is scoped to its root cause — often one file, plus a regression test.
No opportunistic refactor rides along.

## Env-var baseline (Norm 3 anchor)
`git show 5715c6f2:python/mirage/mpk/persistent_kernel.py | grep os.environ` → **9 call sites, 5
unique vars, all build-path/infra**: `MIRAGE_HOME`, `NVSHMEM_INC_PATH`, `NVSHMEM_LIB_PATH`,
`MPI_INC_PATH`, `MPI_LIB_PATH`. Any `MPK_*` toggle in a landed diff to this file is residue to
purge (hard-wire the win, delete the debug knob). `multigpu.py` legitimately reads the NVSHMEM
build-path vars — that is infra, not an experiment lever.

## Comment-density baseline (Norm 6 anchor)
`mla_sm100_2sm.cuh` (from #605) ≈ 5.1% comment lines; `models/qwen3/builder.py` ≈ 10.4%.
Sparse, functional, no narrative.

## How to reproduce these footprints
```bash
git show -s --format='%h %ci %s' <commit>
git show --stat --format='' <commit> | grep '|'   # per-file diffstat
```
