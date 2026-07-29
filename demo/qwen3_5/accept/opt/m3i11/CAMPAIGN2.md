# M3-I11 campaign 2 — the TMA store-visibility fix, and the discriminating experiment

Predictions were written and committed in `PREREG.md` (commit `0cdd52f0`) **before**
the first GPU run. This file reports what happened.

**Headline: the fix is right and is landed; it is not the cause of the census
nondeterminism.** The pre-registered discriminator ran 124 cold-compile reps
(plus a 10-rep negative control and 8 warm perf runs per arm) across six GPUs and
found the divergence at statistically identical rates with and without the fence:
CTRL 7/55, FIX 6/59, Fisher exact p = 0.77. Under `PREREG.md` prediction 2 the
fence is **EXONERATED**. The phenomenon is now cheaply reproducible, which is the
useful part: it went from "1 event in 27 runs, cause unknown" to 10–16% per cold
rep on a device that happens to be in the bad state.

## 1. The defect and the correct primitive

`linear_sm100_mpk.cuh` ended every task with `cute::tma_store_wait<0>()`, which
is `cp.async.bulk.wait_group.read 0`
(`deps/cutlass/include/cute/arch/copy_sm90_tma.hpp:1245-1258`).

PTX ISA 9.0 **§9.7.9.25.6.2** (`cp.async.bulk.wait_group`) defines the two forms:

> By default, `cp.async.bulk.wait_group` instruction will cause the executing
> thread to wait until completion of all the bulk async operations in the
> specified bulk async-group. A bulk async operation includes the following:
> ▶ Optionally, reading from the tensormap. ▶ Reading from the source locations.
> ▶ **Writing to their respective destination locations.** ▶ **Writes being made
> visible to the executing thread.**
> The optional `.read` modifier indicates that the waiting has to be done until
> all the bulk async operations in the specified bulk async-group have
> completed: 1. reading from the tensormap 2. the reading from their source
> locations.

So `.read` retires when the source smem may be recycled — by construction before
the destination global write. That is the right wait for the in-loop
`tma_store_wait<NUM_C_STAGE-1>()` that guards stage reuse, and the wrong wait for
the end of a task.

It is load-bearing here because the CTA does not exit at the end of an MPK task.
It release-increments the trigger event (`persistent_kernel.cuh:1063`,
`atom_add_release_gpu_u64`) and a consumer CTA acquires that counter
(`persistent_kernel.cuh:1004`, `ld_acquire_sys_u64`) and reads the output with
ordinary loads. TMA stores are performed in the **async proxy**;
`atom.add.release.gpu` and `__syncthreads()` order generic-proxy accesses only.

No separate proxy fence is needed once completion is observed, per PTX ISA 9.0
**§9.7.9.25.2 (Async Proxy)**:

> The `cp{.reduce}.async.bulk` operations are performed in the asynchronous proxy
> (or async proxy). Accessing the same memory location across multiple proxies
> needs a cross-proxy fence. For the async proxy, `fence.proxy.async` should be
> used to synchronize memory between generic proxy and the async proxy.
> **The completion of a `cp{.reduce}.async.bulk` operation is followed by an
> implicit generic-async proxy fence. So the result of the asynchronous operation
> is made visible to the generic proxy as soon as its completion is observed.**

Hence the narrowest correct primitive is `cp.async.bulk.wait_group 0` — the
default (non-`.read`) form — and **not** an added `fence.proxy.async.global`.

**In-repo precedent settles it.** Every Hopper MPK task that ends on a TMA store
already uses that form via `kernel::tma::store_async_wait<0>()`
(`hopper/tma.cuh:33-36`): `linear_hopper.cuh:360`, `norm_linear_hopper.cuh:363`,
`linear_swapAB_hopper.cuh:383`, `matmul_demo_hopper.cuh:262`. The Blackwell
kernel, added later against CUTLASS's cute helpers (#514), is the only one that
used the `.read` helper at the task-terminal wait. The fix restores the repo's
own idiom; the helper it calls was already in the tree and already included by
this file.

Reaches Qwen3.5 through `linear_layer` only: lm_head (`builder.py:477`), the MoE
router projection (`:625`), GDN `in_proj_ba` (`:530`) and the bf16 qkvg
projection (`:582`). The dense FP8 path (`linear_fp8_blockscale_sm100.cuh`) uses
plain global stores and is unaffected.

**Two siblings carry the same defect class and are NOT touched here** (neither is
reachable from Qwen3.5): `linear_fp8_1d2d_sm100.cuh:716` has the identical
`tma_store_wait<0>()` task-terminal wait, and `linear_fp8_sm100.cuh` is worse —
its task ends after `cute::tma_store_arrive()` (`:723`) with **no terminal wait at
all**. Both should be fixed with the same one-liner when something makes them
reachable.

## 2. The fix

`kernel::tma::store_async_wait<0>()` in place of `cute::tma_store_wait<0>()` at
the task-terminal wait. Commit `0cdd52f0`. One kernel line, plus the citation
comment and the `utils.py` TODO resolution.

## 3. Gates

| gate | result |
|---|---|
| AC-3 sweep, all five bs, per-case byte-diff vs committed `results/dumps_final` | **50/50 identical** (`bs1..bs16`, 10 prompts each) |
| AC-3 verdict profile | unchanged: the same 5 pre-existing `p06-poem @ pos 60` failures (exact tie, `margin=0.0`, `ref_top1=31000` vs `engine=40581`). Byte-identical dumps make this necessary, not merely observed. |
| warm-run bit-exactness, bs4 + bs16, 8 runs per arm | every run hashes to the census consensus `be346b6d868ef8e7e46980426e662722` |
| perf, interleaved same-window, 3 reps/arm | bs4 **−0.01%** median (rep-paired −0.01%); bs16 **+0.05%** median (rep-paired +0.08%). Both inside run-to-run noise. |

Perf detail (ms per decode step, CTRL vs FIX alternating on one GPU):

```
bs=4    ctrl 10.9648 10.9651 10.9636   median 10.9648
        fix  10.9662 10.9618 10.9636   median 10.9636    -0.01%
bs=16   ctrl 17.4885 17.4784 17.5002   median 17.4885
        fix  17.4860 17.4965 17.5133   median 17.4965    +0.05%
```

A fence adds ordering, not arithmetic, and only warp 0's elected lane blocks on
it once per task — the measurement matches that expectation.

**Unit instruments: unavailable at this SHA, in BOTH arms.** The standalone
kernel-wrapper tests (`tests/runtime_python/blackwell/sm100_linear/`
`test_matmul_mpk.py`, `test_matmul_splitk.py`) hang inside the kernel launch on
the **unpatched** clone, and both `test_mode` graphs that drive `linear_layer`
(`test_diamond_fork_join_testmode.py`, `test_qwen3_mlp_testmode.py`) hang at
"Finished Launching Persistent Kernel (Async)" on the **unpatched** clone
(rc=124 at 600–900 s). Pre-existing, unrelated to this change, reported as its
own finding. Their build also needs a bypass: those `setup.py` files pin nvcc
12.8 while this venv's torch is built against CUDA 13.0, so torch's advisory
`_check_cuda_version` aborts the build — the shipped MPK JIT already mixes
exactly those two versions, so neutralising the check reproduces the shipped
toolchain rather than deviating from it.

In-runtime coverage of the changed kernel therefore rests on the AC-3 sweep
(5 bs × 10 prompts through lm_head, router, `in_proj_ba`, qkvg) and the 95+
1024-token census runs, which is stronger coverage than the unit tests would
have given.

## 4. The discriminator

Protocol: `e4_full.py`, bs4, msl 1280, 1024 new tokens, ten reference prompts, a
fresh kernel dir compiled in-process every rep, KV/GDN wave-boundary
fingerprints attached — the M3-I5c S6 arm's protocol exactly.

Design change vs the brief, made before running: **CTRL and FIX alternate
rep-by-rep on the same GPU** rather than running as blocks. The census anomalies
were first seen in a heavily contended window, so box load is a plausible
modulator; interleaving makes load drift unable to masquerade as an arm effect.
A same-window CTRL arm was added for the same reason — the 1/10 baseline came
from an earlier window, so "0 under the fix" alone would have been confounded
with "the window changed".

The primary instrument is the **fingerprint**, not the token md5. That choice
decided the result: read by md5 alone the FIX arm looks clean, and the campaign
would have reported a false confirmation.

### Result

| block / GPU | CTRL reps | CTRL state-div | CTRL token-div | FIX reps | FIX state-div | FIX token-div |
|---|---:|---:|---:|---:|---:|---:|
| `g0` — GPU0 (block cut short: foreign job joined) | 3 | 1 | 1 | 2 | 1 | 0 |
| `g1` — GPU1 | 10 | 1 | 1 | 15 | 1 | 0 |
| `k1` — GPU1, second block | 10 | 1 | 1 | 10 | 1 | 0 |
| `k2` — GPU2 | 9 | 4 | 3 | 8 | 3 | 2 |
| `h3` — GPU3 | 14 | 0 | 0 | 14 | 0 | 0 |
| `g7` — GPU7 (the pre-registered device) | 9 | 0 | 0 | 10 | 0 | 0 |
| `g6fix` — GPU6, negative control, FIX only | — | — | — | 10 | 0 | 0 |
| **pooled (excl. negative control)** | **55** | **7 (12.7%)** | **6 (10.9%)** | **59** | **6 (10.2%)** | **2 (3.4%)** |

One further CTRL rep (`g7_ctrl_c4`) is excluded as a run error, not a result: I
pruned the census kernel directory for disk space while that compile was in
flight and it died on a missing `.so`. My fault, not the kernel's.

- state-level (any wave-boundary KV/GDN fingerprint key deviates from the set
  consensus): Fisher exact **p = 0.77** — no detectable effect.
- token-level: 6/55 vs 2/59, Fisher exact **p = 0.15** — a nominal 3× reduction
  that is not significant, and undercut by the fact that the single most severe
  event of the whole campaign is in the **FIX** arm (`k2_fix_c8`: all three
  waves, all ten prompts diverging, 101–953 of 1024 tokens each).

GPU7 did free up eventually (after ~17 h occupied) and the pre-registered arm ran
there: **0/9 CTRL and 0/10 FIX, byte-clean**. That is a null result for the
discriminator on that device, and it is the single most interesting number in the
table — see below.

**Verdict: EXONERATED**, per `PREREG.md` prediction 2. The fix lands anyway as
the correctness repair it is.

### What the campaign did establish

1. **The divergence is cheap to reproduce.** 13–16% per cold rep on a
   divergence-prone GPU, ~2 min per rep. It is no longer a needle in a haystack;
   it is bisectable.
2. **Device-conditionality is real and sharp, but it is a device *state*, not a
   device identity.** In this campaign GPUs 0, 1 and 2 diverged while GPUs 3, 6
   and **7** were byte-clean (28, 20 and 19 reps). GPU7 is the device that
   produced the M3-I5c S6 event and is on the historical anomaly list (1/4/7);
   tonight it was clean. So "GPU-conditional" as previously recorded is too
   strong — the prone set moves. Whatever the mechanism is, it is something a
   device can enter and leave, which rules out fixed per-device hardware
   properties (binning, a bad row) and points at state left behind by workload
   history: clocks/power, page/allocation history, or driver-side state.
3. **The "one wave only" signature is a property of mild events, not of the
   phenomenon.** Mild events touch one wave (`g1_ctrl_c1`: w1 only, w0/w2 clean —
   the exact I5c S6 shape). Severe ones touch all three (`k2_ctrl_c9`,
   `k2_fix_c8`). Reporting the wave-scoping as diagnostic of the mechanism was
   an over-read of a single observation.
4. **Severity varies enormously**, from 1.4% of one wave's K/V entries with no
   token effect, up to every prompt in every wave.
5. **GPU2 also started hanging** — `k2_fix_c9` tripped the engine watchdog
   ("no progress for 115 s" in wave-1 prefill). Same GPU, same window as its
   4/9 + 3/8 divergence rate. Divergence and hangs co-locate on one device.
6. **Position within the interleaved block does not predict divergence**
   (events at positions 1, 2, 4, 5, 9, 13, 14, 15, 16, 17) — first-touch /
   allocator-warmup, a candidate from `PREREG.md`, is not supported.

### Next discriminator, recommended

Device state now dominates every other correlate, and it tracks the *device's
recent history*, not the code and not the device identity. Two observations to
build the next experiment on, both uncontrolled so far:

- GPU2 was the dirtiest device of the campaign (4/9 CTRL, 3/8 FIX, plus an engine
  watchdog hang) and I claimed it **immediately** after a ~145 GB foreign job
  released it.
- GPU7 was clean (0/19) and I claimed it after it had been running one long
  steady 165 GB job for ~17 h and then sat idle.

So the discriminator to run next is a paired census on **one** device: arm 1
immediately after a large foreign allocation is freed, arm 2 after the same
device has been idle (or after `nvidia-smi -r` / a fresh CUDA context with the
allocator caches dropped). If arm 1 diverges and arm 2 does not, the mechanism is
allocation/page history and the search moves to first-touch state on
`cudaMalloc`ed megakernel intermediates — which campaign 1 tested only by
dirtying host-side patterns (`e3_churn.py`), never by controlling the *previous
tenant's* footprint.

After that, the per-wave fingerprint should be pushed down to per-layer
snapshots within a single diverging run to localise the first diverging tensor —
now affordable because the event rate on a prone device is ~1 in 8 rather than
1 in 27.

## 5. The `models/utils.py` TODO

Partially resolved; the comment is updated in place.

The TODO ("both MPK ptx and cutlass version will output unexpected result (not
same out put for same prompt) if the OUTPUT_SIZE is too big") dates to
2025-12-20 (#537). On Blackwell the "cutlass version" is `linear_sm100_mpk.cuh`
and this issue found and fixed one concrete mechanism for exactly that symptom
there; exposure grows with per-task `OUTPUT_SIZE` because more and larger store
atoms are in flight at the terminal wait, which matches the size dependence the
cap works around.

It is **not** the whole story: the ptx-based Hopper linear has used the correct
`store_async_wait<0>()` since #459 (2025-08-31), i.e. before the TODO was
written, and the Ampere linear writes its output with plain generic-proxy
stores. Any residual size-dependent nondeterminism on those two paths has a
different cause and was not re-measured here. The 256 cap stays — relaxing it
changes the task graph and per-task tile shape and needs its own run.

## 6. Two side findings

**The "HEAD is broken" alarm was a stale compiled extension, not a source
regression.** `build_annotated_graph: bgraph inputs/outputs count mismatch` at
`c80ebd68` came from a `core.so` older than `b0920b28`, whose `src/kernel/`
changes altered the graph the Python side emits. There is **no `src/` change
anywhere in `170ab325..c80ebd68`** — only `builder.py` — so a fresh build at HEAD
must behave like a fresh build at `170ab325`, and it does: fresh clone, fresh
`build_ext`, graph assembled in 12.3 s, 43928 tasks, all five AC-3 batch sizes
green. Rebuild the extension after any commit that touches `src/`.

**The one AC-3 case that differed at HEAD was the nondeterminism, not a
regression.** `bs2/p08-science` diverged from baseline at position 28 in a single
cold arm — on GPU0, one of the divergence-prone GPUs, one prompt of fifty cases,
a mid-sequence trajectory fork. A 2×2 replication on clean GPU3, 3 reps per arm,
attributes it cleanly:

| arm | fence | attention pass-size 4→2 | bs2 result, 3 reps |
|---|---|---|---|
| A = HEAD `c80ebd68` | yes | yes | baseline, all 10 prompts, all reps |
| B = HEAD − fence | no | yes | baseline, all 10 prompts, all reps |
| C = `0cdd52f0` tree | yes | no | baseline, all 10 prompts, all reps |
| D = `170ab325` | no | no | baseline, all 10 prompts, all reps |

120 trajectories, all identical to the committed baseline. The fence is
exonerated, `a86b1eb1` is exonerated, and so is any interaction between them.
The cause is the nondeterminism this issue is about.

## 7. What this means for M4

M4's final gate runs 1024-token workloads. At 10–16% divergence per cold run on a
device in the bad state, **a 1024-token gate is not trustworthy on this box
today**, and the ≥2-rep protocol amendment is necessary but not sufficient — two
reps disagree about a quarter of the time at that rate.

Worse, the prone set moves between sessions (GPU7 was prone in the I5c window and
clean in this one), so a device cannot be certified clean once and trusted later.
Until the mechanism is found the gate needs one of:

1. the fingerprint detector attached to the gate run itself — `e4_full.py`
   already snapshots KV/GDN state at every wave boundary, it is ~100% sensitive
   where token ids are ~2%, and it costs nothing on top of a run the gate is
   doing anyway. ≥2 reps compared *by fingerprint* rather than by token md5 turns
   a silent wrong answer into a loud one; or
2. an in-window clean demonstration on the gate device, recorded as part of the
   gate evidence — not a device allow-list carried over from a previous session.

Option 1 is the one to build: it is cheap, it composes with the existing ≥2-rep
rule, and it fails loudly instead of quietly.

## 8. Files

- `PREREG.md` — the pre-registration, committed before any run.
- `scripts/i11b_census_ab.sh` — the interleaved CTRL/FIX cold-compile census.
- `scripts/i11b_fpdiff.py` — per-set fingerprint divergence + wave-scope report.
- `scripts/i11b_pool.py` — the pooled discriminator table and the tests above.
- `scripts/i11b_ac3.sh`, `scripts/i11b_perf.sh` — the gates.
- `scripts/bs2_attrib.sh` — the 2×2 attribution of the HEAD AC-3 case.
- `scripts/head_verify.sh` — fresh-clone/fresh-build verification at a given SHA.
- `results/campaign2_pooled.txt` — the pooled table as produced.
- Raw fingerprints and per-rep metadata: `/home/catalyst/mpk-artifacts/m3i11/campaign2/`.
