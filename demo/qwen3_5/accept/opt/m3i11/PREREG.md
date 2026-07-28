# M3-I11 campaign 2 — pre-registration (written and committed BEFORE any run)

Registered 2026-07-28, before the first GPU run of this campaign. Everything below
is a prediction, not a result.

## Intervention (single change under test)

`include/mirage/persistent_kernel/tasks/blackwell/linear_sm100_mpk.cuh`, the
task-terminal TMA store wait: `cute::tma_store_wait<0>()`
(= `cp.async.bulk.wait_group.read 0`, source-read completion) →
`kernel::tma::store_async_wait<0>()` (= `cp.async.bulk.wait_group 0`,
destination-write completion + visibility to the executing thread).

Nothing else changes. `python/mirage/mpk/models/utils.py` gets a comment-only
edit (the TODO resolution), no behaviour change.

## Correct-primitive argument (decided by inspection, before measuring)

PTX ISA 9.7.9.25.6.2 (`cp.async.bulk.wait_group`): the default form waits for
the bulk async operation including "Writing to their respective destination
locations" and "Writes being made visible to the executing thread"; the optional
`.read` modifier waits only for "reading from the tensormap" and "the reading
from their source locations". PTX ISA 9.7.9.25.2 (Async Proxy): "The completion
of a `cp{.reduce}.async.bulk` operation is followed by an implicit generic-async
proxy fence. So the result of the asynchronous operation is made visible to the
generic proxy as soon as its completion is observed." Therefore the non-`.read`
wait is sufficient and no separate `fence.proxy.async{.global}` is required.
In-repo precedent: every Hopper MPK task that ends on a TMA store already uses
the non-`.read` form (`hopper/linear_hopper.cuh:360`,
`hopper/norm_linear_hopper.cuh:363`, `hopper/linear_swapAB_hopper.cuh:383`,
`hopper/matmul_demo_hopper.cuh:262`); the Blackwell kernel is the only one that
used cute's `.read` helper at the task-terminal wait.

## Discriminator

Cold-compile census on **GPU7**, `e4_full.py`, bs4, msl 1280, 1024 new tokens,
10 reference prompts, KV/GDN wave-boundary fingerprints attached — the exact
protocol of the M3-I5c S6 arm (`run_s6_census.sh`), which produced **1 diverging
rep in 10 cold reps** (rep `g7_bs4_c3`: wave1-only, 4 prompts, ~46% of that
wave's KV/GDN entries, 58–838 of 1024 tokens per prompt; waves 0 and 2
byte-clean).

Three arms, same window, same box, same clone:

| arm | code | GPU | reps | role |
|---|---|---|---|---|
| CTRL | HEAD, no fix | 7 | 10 | same-window reproduction of the 1/10 baseline |
| FIX | HEAD + fix | 7 | ≥15 | the discriminator |
| NEG | HEAD + fix | 5 or 6 | 10 | negative control, keeps GPU-conditionality readable |

CTRL is added beyond the brief because the baseline rate was measured in an
earlier window; without it, "0 divergences under the fix" is confounded with
"the window changed".

## Predictions

1. **If the fence is the mechanism:** FIX = 0 diverging reps out of ≥15, and the
   wave-scoped signature (one whole wave's KV/GDN entries perturbed, neighbouring
   waves byte-clean) does not appear at all. CTRL is expected to show ≥1
   divergence in 10; if CTRL shows 0, the discriminator is UNDERPOWERED and the
   result must be reported as inconclusive rather than as confirmation.
2. **If the fence is not the mechanism:** FIX shows divergence at a rate
   statistically indistinguishable from 1/10, with the same wave-scoped
   signature → the fence is EXONERATED as the cause of the census divergence.
   The fix still lands as a correctness repair. Next hypothesis to carry
   forward: first-touch page/allocation state or a JIT-timing-sensitive race
   outside this kernel (the wave1-only + cold-compile + GPU-conditional
   signature).
3. **NEG** is predicted 0/10 either way; a divergence there would break the
   GPU-conditional correlate and is itself a finding.

Power note: with a true rate of 0.1 per rep, 15 clean reps give
P(0 | rate unchanged) = 0.9^15 = 0.21. That is weak on its own, which is why
CTRL runs in the same window and why the *signature* (not just the count) is
part of the criterion. If CTRL diverges and FIX does not, the paired result is
the evidence; the absolute rate is not.

## Gates (predictions)

- Bit-exactness: a fence adds ordering, not arithmetic. Predicted byte-identical
  outputs vs the pre-fix baseline at every AC-3 case (5 batch sizes) and in the
  unit/oracle instruments for the affected linear tasks (lm_head, MoE router
  projection, GDN `in_proj_ba`, bf16 qkvg).
  Falsifier: any per-case byte diff. If the fix *changes* bytes, the pre-fix
  build was reading stale data somewhere reproducibly, which would itself be the
  finding.
- Perf: the wait is once per linear task, and only warp 0's elected lane blocks
  on it. Predicted median end-to-end regression **≤1%**; ≥3 reps per arm,
  same-window interleaved control. `lm_head` / router / `in_proj_ba` are hot, so
  a real regression is plausible and will be reported as measured either way.
