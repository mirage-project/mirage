# M3-I9 — the bs16 admission protocol, and the matched-geometry re-measure

M3-I1 ranked this backlog #4: "fix the mbt=16 admission policy, +44% wave-level at bs16", plus
rank 11, the measurement debt that every MPK number is at the AC-3 geometry while the vLLM
baseline is at 256/1024. This issue owns both.

Prepared in **prep mode**. No GPU: all eight B200s are contended, M3-I2b owns the next window
and M3-I8 the one after, so `plan_m3i9.sh` is written and **not armed** — it refuses without
`M3I9_ARMED=1`, refuses while either prior lock or plan pid is alive, and refuses if the shared
clone still has a prior window's arm staged.

## Three results

**1. Backlog #4's +44% does not reproduce, and its mechanism was wrong.**
The delta_basis was `36 prefill iters @ 25.5 ms + 107 decode iters @ 22.0 ms = 3272 ms`. The
25.5 ms is the *mean of the 108 measured mixed iterations*, and 95% of those are starved steps
moving one token per slot — the cheap kind. An iteration that actually delivers 16 prefill
tokens delivers them to **one** slot, which the fitted cost law prices at 36.9 ms. The same
143-iteration schedule, correctly priced, is **+26.8%**, and 143 iterations is neither the floor
(131 is) nor reachable by any admission *order*. The reachable optimum is **+61.5%**. Full
arithmetic: `python3 rank_policies.py`, or `predictions.md` §2.

The mechanism matters more than the number. "`mbt=16` equals the batch size so decode saturates
admission" reads as *raise `mbt`* — which needs M3-I5b's router row loop and *costs* step time
(`mbt=32` alone is +15.5% central and negative at the pessimistic end of its band). The actual
defect is that **one slot may take the whole token budget**. `prepare_next_batch` walks slots in
order and gives each prefilling request `min(remaining, mbt − used)`, so the `j`-th slot to
finish prefill only ever gets `mbt − j` tokens per iteration — a harmonic blow-up that turns
36 iterations of prefill work into 108.

**2. Iteration cost tracks the LARGEST slot chunk, not the token total.** Fitting M3-I1's 638
measured iterations against the schedule replay gives, at bs16,

```
iter_us = 18917 + 982.4 * max_chunk + 142.7 * n_live       R2 = 0.960
```

against `R2 = 0.510` for the token total. The mechanism is `grid_dim=(v_heads, mbr)` on the GDN
recurrent task — one block per slot, walking that slot's tokens in order — so the iteration's
critical path is one slot's sequential work. Sixteen tokens in one slot cost **1.66×** the same
sixteen tokens spread one per slot. The model reproduces all five measured profiled wave wall
times to ≤0.04% (`python3 cost_model.py`).

**3. HAZARD-COMPACTION is not dormant — it is firing in the shipped waves, and it explains
the standing `identical: false` signal.** `mpk_engine_run.py` says the hazard "only fires when a
request retires while another is still active — i.e. rolling admission". The trigger is right;
the `i.e.` is not. Retirement-while-others-are-active happens *inside* a single wave, because
`max_seq_length` retires on the global step while slot-order-greedy admission advances the low
slots first. The replay gives **1 / 12 / 69 live-slot migrations at bs 4 / 8 / 16**. Most are
harmless — the request has already written its reported 64 tokens — but the ones that are not
are exactly the duplicate padding slots, and those are exactly the slots reporting
`identical: false` in every committed dump: **14 of 14, no exceptions**. The competing
explanation (a differently *chopped* prefill gives a different answer) is refuted by the
committed report itself: 9 of 10 prompts are prefilled with 2–4 different chunkings across batch
sizes in placements that never migrate, and all 25 (prompt, bs) sequences are byte-identical.
`python3 compaction_audit.py`.

At the pinned 256/1024 geometry this stops being a footnote: **15 of 16 requests** have their
reported window written across a migration, and there are no duplicate slots to absorb it —
every one of them is reported. The fix is a correctness prerequisite for M4, not an
optimisation.

## The policy ranking (predicted)

Ranked by predicted bs16 wave time. `runtime` = needs a `prepare_next_batch` change;
`py/adapter` = shippable in the protocol layer today. Full tables incl. 256/1024 and the mbt
sensitivity band: `python3 rank_policies.py`.

| rank | policy | lane | iters | vs today | migrations | straddling |
|---:|---|---|---:|---:|---:|---:|
| 1 | **per-request token cap = 1** | runtime | **131** | **1.615×** | **0** | **0** |
| 2 | hold-decode until all prefilled | runtime | 143 | 1.318× | 36 | 0 |
| 3 | mbt=32 + cap=2 | runtime + I5b | 119 | 1.313× | 32 | 0 |
| 4 | mbt=32 | runtime + I5b | 123 | 1.155× | 50 | 6 |
| 5 | slot order = ascending-padded | **py/adapter** | 179 | 1.114× | 91 | 4 |
| 9 | *today* | shipped | 203 | 1.000× | 69 | 6 |
| 10 | mbt=64 + cap=4 | runtime + I5b | 113 | 0.912× | 32 | 0 |

Length-sorted admission is worth +11.4% at AC-3 and **exactly nothing** at 256/1024, where every
prompt is the same length — and it *raises* compaction exposure. Staggered admission and drain
refill are **OUT**: both move live slots by construction, which is what
`assert_no_rolling_admission` refuses. They stay out until GDN state migrates with the slot.
`mbt=64` is a net loss at the central extrapolation.

The winner is also the only candidate that needs no extrapolation: every one of its 131
iterations is the directly measured 16-slot 1-token step (22003 µs, n=18, spread 0.49%).

## The runtime knob — implemented (local tree only, unbuilt)

The winning policy is **not** pure Python: `prepare_next_batch` is device code. It is now landed
in this tree with a default that is byte-identical to today. **Not built** (B200s contended) and
**not propagated** to the box clone — M3-I2b owns that clone's window. CPU gate:
`python3 test_admission_policy.py`.

**1. `include/mirage/persistent_kernel/admission_policy.h`** (new) — the default define and the
one function both call sites use:

```cuda
#ifndef MPK_MAX_TOKENS_PER_REQUEST
#define MPK_MAX_TOKENS_PER_REQUEST MPK_MAX_NUM_BATCHED_TOKENS
#endif

MPK_ADMISSION_FN int
    admission_prefill_tokens(int remaining, int budget_left, int cap) {
  int k = (remaining < budget_left) ? remaining : budget_left;
  return (k < cap) ? k : cap;
}
```

It is `__host__ __device__` under nvcc and plain `inline` otherwise, which is what lets the CPU
test exercise the *real* function rather than a copy. Two `static_assert`s reject `cap < 1`
(would stall every prefill) and `cap > mbt` at compile time.

**Why the default is provably a no-op, from the code.** Both call sites pass
`budget_left = MPK_MAX_NUM_BATCHED_TOKENS - num_tokens`, and `num_tokens` starts at 0 and only
ever increases by a non-negative `num_new_tokens`, so `0 ≤ budget_left ≤ MPK_MAX_NUM_BATCHED_TOKENS`.
The default `cap` **is** `MPK_MAX_NUM_BATCHED_TOKENS`. Hence
`min(remaining, budget_left) ≤ budget_left ≤ cap`, so the second clamp is the identity for every
reachable state and the expression equals the pre-M3-I9 `min(remaining, budget_left)` exactly.
Check B of `test_admission_policy.py` confirms it empirically: chunk-vector-identical schedules
at all five batch sizes, i.e. M3-I1's validated 109/109/109/111/203.

**2. `include/mirage/persistent_kernel/persistent_kernel.cuh`** — the two MODE_OFFLINE prefill
sites (step 3 and the admission loop) call the helper. The decode branch is untouched:
`min(1, budget)` is a correctness constraint (token *n+1* depends on token *n*), not a scheduling
choice, and it is already ≤ any cap ≥ 1. MODE_ONLINE / MODE_ONLINE_PINNED /
MODE_ONLINE_NOTOKEN are untouched (check A asserts it).

**3. `python/mirage/mpk/persistent_kernel.py`** — `max_tokens_per_request: int = None` on
`PersistentKernel.__init__`, range-checked and refused outside `mode="offline"`. `None` emits no
`-D` at all, so the compile command and the generated graph are unchanged byte for byte.

**4. `demo/qwen3_5/accept/mpk_engine_run.py`** — default off:
`--per-request-token-cap {auto,N}`, `auto = max(1, mbt // batch_size)`. It raises a
`NotImplementedError` naming this spec if the installed `PersistentKernel` has no
`max_tokens_per_request` parameter, rather than silently measuring the unmodified schedule under
a flag that claims otherwise — which is exactly what happens against the box's current build
until it is rebuilt.

Why `auto` and not a global 1: the cap must be an equal *share* of the budget. A global 1 is a
19% regression at bs1. Bound to `bs`, the cost at bs 2/4/8 is 0.6–2.7% and it buys zero live-slot
migrations at every batch size — take the small loss (`predictions.md` §1).

Not needed: I5b's router row loop, a larger `mbt`, or GDN state migration. The cap *removes*
every live-slot move rather than adding more.

## What is implemented vs specced

| | where | state |
|---|---|---|
| policy simulator (compaction + rolling admission + per-slot chunks) | `protocol_sim.py` | implemented; `--self-check` reproduces M3-I1's 109/109/109/111/203 |
| per-iteration cost model + closure gate | `cost_model.py` | implemented; ≤0.04% on all five waves; `--fit` re-derives |
| policy ranking, backlog #4 re-derivation, mbt band | `rank_policies.py` | implemented |
| compaction / isolation-mismatch audit | `compaction_audit.py` | implemented |
| `--slot-order` arm (default `wave` = shipped) | `mpk_engine_run.py` | implemented, default-off |
| per-wave compaction exposure in the timings artifact | `mpk_engine_run.py` | implemented, diagnostic only |
| `first_divergence` on duplicate-slot checks | `mpk_engine_run.py` | implemented — this is the free falsifier |
| corrected HAZARD-COMPACTION docstring | `mpk_engine_run.py` | implemented |
| `--per-request-token-cap` adapter flag | `mpk_engine_run.py` | implemented, refuses without the runtime knob |
| the runtime knob itself | `admission_policy.h` (new) + `persistent_kernel.cuh` + `persistent_kernel.py` | **implemented, local tree only, NOT BUILT** — clang-format clean |
| CPU gate on the admission arithmetic | `test_admission_policy.{cpp,py}` | implemented; 4 checks, 30 chunk-exact replays, PASS |
| matched-geometry re-measure design | `remeasure-protocol.md` | implemented (design + generator + guards) |
| pinned-geometry prompt generator | `make_synthetic_prompts.py` | implemented; `--verify` cross-checks the seed formula |
| the capture | `plan_m3i9.sh` + `analyze_m3i9.py` | written, **not armed** |

## Running the offline analysis now

```bash
python3 protocol_sim.py --self-check     # reproduces M3-I1's validated iteration counts
python3 cost_model.py                    # closure gate: all five measured waves, <=0.04%
python3 cost_model.py --fit              # re-derive the coefficients from opt/tables/
python3 rank_policies.py                 # the ranking + the +44% re-derivation
python3 compaction_audit.py              # the hazard audit + the H1/H2 discrimination
python3 test_admission_policy.py         # the C++ knob: static + no-op + cap + range (needs g++)
```

No GPU, no B200 artifacts beyond `opt/meta/` and (for `--fit`) `opt/tables/`.

## What is queued for the window

Value order, and the first three stages need **no source change at all** — the central mechanism
claim is settled before anyone compiles a modified runtime.

0. shipped baseline at bs16 + the **free falsifier**: `first_divergence` must clear the
   registered per-slot bounds 60/54/46/35/19 and be strictly decreasing across slots 10–14
1. negative control at `msl=212` (zero straddles predicted) — the six duplicate pairs must come
   back `identical: true`
2. cost-law check: `--slot-order sorted-padded`, predicted 179 iterations / 4214 ms — the only
   cheap test of the cost law *off* its fit set. A miss here stops the runtime change.
3. build the runtime knob (already landed in-tree, CPU-gated) and prove codegen identity for
   default-off — one rebuild of the box clone, which is why it cannot be prepped further here
4. AC-3 gate under the cap: bs16 first, then the sweep + per-case byte diff vs `e51cb86`
5. perf bs16, ≥3 reps: predicted 4566.5 → 2825 ms unprofiled
6. perf bs 1/2/4/8: the predicted **small losses** are part of the claim
7. matched geometry 256/1024, base and cap, all five bs — feeds M4
8. analysis + `backlog.json` update

Arm with `M3I9_ARMED=1` once M3-I2b's and M3-I8's windows have released.
