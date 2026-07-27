# F1 root cause — the gate is correct; the oracle's long/short threshold is not

**Verdict: F1 did not fire.** `gate_padding_rows` achieves exactly what it claimed. At bs1 the
activated expert count is **8.000 in every one of the 32 steady-decode iterations, in every one of
the 40 layers, with zero variance** (cap 8). At bs2 it is **15.731 mean, range 14.875–16.000**
(cap 16), which is the top-8 collision structure of two tokens, not an excess.

The reported 10.07 / 18.37 came from `trace_lib.per_task_table`'s hard-coded 1 µs split between
"dispatched but empty" and "did real work". For task 241 that threshold sits **inside** the empty
task's own latency tail, not in the gap between the two populations. Moving it anywhere into the
real gap (2–48 µs) removes the entire excess and nothing else.

All numbers below are re-derived offline from the same raw captures the falsifier used
(`/home/catalyst/mpk-artifacts/m3i8-f1-raw/raw_bs{1,2}_rep0.npz`, originally
`~/mpk-qwen35/m3i9x/f1/`). No GPU. Scripts: `f1_threshold_layer_probe.py`,
`f1_rootcause_probe.py`; outputs `f1_data/probe_bs1.json`, `f1_data/rootcause_bs{1,2}.json`.

---

## 1. The mechanism

`TASK_MOE_W13_FP8_BLOCKSCALE_SM100` (task 241) is dispatched on a **static grid**: exactly
**10240 launches per iteration** at bs1 and bs2 (min = max over every decode iteration) = 40 layers
× 256 slots, where 256 = 128 group slots × `moe_n_splits` = 2. 128 is the ceiling on distinct
experts at `mbt=16, top-8`. A slot whose group index is ≥ `mpk_active_expert_ids[NUM_EXPERTS]` pops
the queue, checks its dependency, writes two profiler stores and exits.

The oracle counts a launch as "activated" when its duration ≥ 1 µs. The actual duration histogram
for task 241 over the whole bs1 capture (1 116 160 launches = 10240 × 109 iterations,
`probe_bs1.json`):

| duration bucket | launches | |
|---|---:|---|
| 250 ns – 500 ns | 299 446 | empty |
| 500 ns – 750 ns | 610 944 | empty |
| 750 ns – 1 µs | 113 025 | empty |
| **1 µs – 1.5 µs** | **16 611** | **empty, counted as "long"** |
| **1.5 µs – 2 µs** | **74** | **empty, counted as "long"** |
| 2 µs – 48 µs | **0** | — the real gap |
| 48 µs – 64 µs | 75 905 | real tile |
| 64 µs – 128 µs | 155 | real tile |

The empty population is one distribution with a decaying tail — 299k / 611k / 113k / 16.6k / 74 /
0 — that crosses 1 µs for **1.60 % of empty launches** and terminates by 2 µs. Real tiles start at
48 µs. The 1 µs cut therefore slices the empty population; a 2–48 µs cut separates the two cleanly.

Task 242 (`moe_w2`) has the same shape with a thinner tail (671 launches in 1–1.5 µs, 26 in
1.5–2 µs, zero in 2–24 µs, real tiles 24–48 µs).

### Activated count vs. threshold, decode-only iterations (bs1, `probe_bs1.json`)

| threshold | task 241 mean | min | max | task 242 mean |
|---|---:|---:|---:|---:|
| ≥ 1 µs | **10.0734** | 8.85 | 11.91 | 8.0754 |
| ≥ 2 µs | **8.0000** | 8.00 | 8.00 | 8.0000 |
| ≥ 4 / 8 / 16 / 32 µs | 8.0000 | 8.00 | 8.00 | 8.0000 |

10.0734 reproduces `f1_bs1.json`'s reported `tail_mean` 10.0734 exactly, so this is the same
computation, only the threshold moved.

### The excess is exactly the mis-classified population

At bs1, decode-only: launches in [1 µs, 16 µs) = **165.875 per iteration**.
165.875 / (40 × 2) = **2.0734** = 10.0734 − 8.0000, to four decimals. There is nothing else to
explain. At bs2 the same quantity is 216.55 / 80 = 2.7069 = 18.4379 − 15.7310.

### Two consumers of one list must agree — and only do at the corrected threshold

`moe_w13` (241) and `moe_w2` (242) both read the same `mpk_active_expert_ids`, so their inferred
activation must be identical. Full-run averages:

| | ≥ 1 µs | ≥ 4 µs |
|---|---:|---:|
| bs1 task 241 | 10.6472 | **8.7292** |
| bs1 task 242 | 8.8094 | **8.7292** |
| bs2 task 241 | 18.7330 | **16.2618** |
| bs2 task 242 | 16.3571 | **16.2618** |

They disagree by 21 % at 1 µs and agree to the last digit at 4 µs. That is an internal consistency
proof independent of any assumption about what the gate should do.

---

## 2. What the corrected oracle says the gate achieves

Exact per-layer decomposition (task 241 emits exactly 256 launches per layer and layers are
strictly sequential in the task graph, so sorting an iteration's launches and chunking by 256
recovers layer identity with no gap heuristic — `f1_rootcause_probe.py`):

**bs1, 32/32 steady-decode iterations, separator ≥ 16 µs**

- long launches per layer: the set of observed values is `{16}` — every layer, every iteration.
- 16 = 8 experts × 2 splits ⇒ **activated = 8.000 exactly, variance zero.**
- per-iteration activated: min 8.0000, max 8.0000, mean 8.0000.

**bs2, 29/29 steady-decode iterations (`n_live = 2`)**

- long launches per layer ∈ `{26, 28, 30, 32}` = 13/14/15/16 experts × 2 splits.
- per-iteration activated: min 14.875, max 16.000, mean **15.731**, never above the cap of 16.
- Independent-uniform collision model for two top-8 draws from 256 experts predicts
  16 − 64/256 = 15.75. Measured 15.731.

---

## 3. Hypotheses, adjudicated

| | verdict | evidence |
|---|---|---|
| H1 stale / one-iteration-lagged live count | **refuted** | the count is exactly right (8·live), not merely close; a lagged count cannot produce the correct value in 32/32 iterations |
| H2 marks accumulate across iterations | **refuted** | at the corrected threshold the bs1 series is a flat constant 8.000 with zero variance — no growth, no history dependence |
| H3 count slot holds something other than live tokens | **refuted** | empirically (8·live exactly). Source: `task_register.cc:2637` emits `runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]`; `persistent_kernel.cuh:318/378` sets that slot to `num_tokens` (the packed live-token count for the iteration) and `:411-412` back-fills the unused tail with the same value |
| H4 another writer of `mpk_active_expert_ids` | **refuted** | only `topk_softmax_task_impl` writes it (init to −1 at `topk_softmax_sm100.cuh:126-128`, mark at `:339`, compact at `:372-382`); refuted empirically anyway |
| H5 per-layer heterogeneity | **confirmed — but only in the contamination** | activation is dead-uniform (all 40 layers = 16 long launches at bs1). The [1 µs, 16 µs) population is strongly layer-dependent: cv 0.98, layer 8 carries 16.2/iteration, layer 0 carries 0.03. That is queueing structure inside the wave, not marking |
| H6 wrong denominator for the v1 graph | **refuted** | 40 × `moe_n_splits`=2 = 80 is right: total launches 10240/iteration = 40 × 256, real tiles 640 = 40 × 8 × 2. A wrong denominator could not land on an exact integer 8 |
| **H7 (added) instrument threshold artifact** | **confirmed, and complete** | accounts for the excess to four decimals at both batch sizes; see §1 |

Supporting detail for H7 being jitter rather than a second work-doing population: the [1 µs, 16 µs)
launches have durations 1024 / 1056 / 1088 / 1248 ns (p0/p50/p75/p95), max 1792 ns — quantised to
the 32 ns profiler tick and starting at the first tick above 1000 ns — and are spread across **all
128 worker blocks**. A distinct "real expert, zero rows" population would sit at a characteristic
duration and would show up equally in task 242, which it does not (165.9/iteration vs 6.0/iteration
at bs1).

---

## 4. Corrected mechanism statement (the claim that should be pinned)

> With `gate_padding_rows` ON, in any steady-decode iteration the number of activated expert
> groups per layer equals `|⋃_{r < live} top8(r)|` — the union of the live rows' top-8 sets,
> with padding rows contributing nothing. At `live = 1` that is exactly 8. At `live = k > 1` it is
> `8k` minus router collisions, hence `≤ 8·live` with equality only when no two live tokens share
> an expert.

The earlier claim ("activation == unique(live rows' top-8)") was **correct**. What was wrong is the
stricter reading that got attached to it at bs2 — `8·live` is a bound, not an identity, once
`live > 1` — and the instrument used to check it.

**Restated falsifier, for a fresh oracle run:** bucket task-241 launches by `BEGIN_TASK_GRAPH`
iteration using a **≥ 4 µs** long/short separator; over the final 32 steady-decode iterations at
any batch size, `activated > 8·n_live` in any single iteration ⇒ F1 fires. Additionally at bs1,
`activated != 8.000` in any iteration ⇒ F1 fires (this arm has zero measured variance, so it is a
sharp test).

---

## 5. Re-enabling the default

**Justified.** `gate_padding_rows` should go back to default-ON.

1. The pre-registered falsifier was answered by an instrument defect, now root-caused with byte
   evidence drawn from the very captures the falsifier used. Under the corrected instrument the
   pre-registered prediction — "bs1 = 8.0 exactly at every steady iteration; bs2 ≤ 16" — is met
   exactly, including the sharp bs1 arm.
2. Nothing that carried the *decision* moves. AC-3 is byte-identical to the committed M2 dumps in
   both arms at all five batch sizes (`../bytediff_v1.json`), the bs16 inert control is
   byte-identical (`../bytediff_v1_bs16.json`), and the step-time wins (+9.7 / +15.3 / +7.1 /
   +25.1 / −0.0 %) are wall-clock measurements that never touched the oracle.
3. The safety argument in VALIDATION.md still holds and is now redundant: gating only removes
   padding-row marks, and a live row's own top-k comes from its own 256 logits either way.

### What the re-validation must run

No new GPU capture is required to justify the re-enable — the closure is complete on captures
already taken. What should ride the next window, in cost order:

1. **(free, offline) Republish the F1 row.** Re-run `f1_per_iteration.py` with
   `--long-threshold-ns 4000` over the existing `raw_bs{1,2}_rep0.npz` and replace the
   VALIDATION.md F1 table's bs1/bs2 entries.
2. **(~2 GPU-min) Extend to bs4/8/16.** The I8 window saved `--save-raw` only at rep 0 of each arm
   (`run_m3i8.sh:73`); if those `.npz` survive on the box the whole table is re-derivable offline.
   Otherwise one profiled `profile_wave.py` capture per batch size on the v1 arm, then the
   restated falsifier in §4.
3. **(already green) AC-3 byte-diff** on the re-enabled default — it is the gate that actually
   binds, and it passed in the I8 window at every batch size.

### Tooling change this implies (coordinator call — it moves published numbers)

`trace_lib.py:278` hard-codes `short = dd < 1000.0`, and the comment at `:271-277` asserts
"empirically those land under 1 us while any real tile is >= 4 us". The second half is true; the
first is false for task 241 by 1.6 % of launches. Raising the split to 4 µs — or, better, deriving
it per task type from the observed bimodal gap — is the fix. **Blast radius:** every published
"activated groups per layer" number in M3-I1 and M3-I8 is inflated by the same artifact
(M3-I1 base bs1 4508.9 long/iteration → 56.36 activated, `opt/pertask_by_bs.csv`; the contamination
scales with the number of *empty* slots, so it is roughly +1.2 at base and +2.1 at v1). The
conclusions those tables supported — wave counts, the worker-depth cost model — are unaffected,
because they turn on the real-tile counts, which do not move.

---

## 6. Latent hazard found while reading the source (did not fire; not the F1 mechanism)

`topk_softmax_sm100.cuh:372-382` compacts the mark array **in place**:

```cpp
int const mark = mpk_active_expert_ids[local_expert];   // read slot t
if (mark >= 0) {
  int const pos = atomicAdd(mpk_active_expert_ids + NUM_EXPERTS, 1);
  mpk_active_expert_ids[pos] = expert;                  // write slot pos < n_active
}
```

There is no synchronisation between the read and the writes, and the writes land in
`[0, n_active)` — the same slots threads `0 … n_active-1` are about to read. A thread whose mark
slot is overwritten first would read a value ≥ 0 and add a spurious expert.

It does not fire today, and the data proves it: activation is exactly `8·live` with zero variance
over 40 layers × 108 iterations × 2 batch sizes. The reason is structural — for Qwen3.5
`blockDim.x` = 256 = `NUM_EXPERTS`, so every thread handles exactly one expert in a single pass and
all 256 loads are issued before any store round-trip completes. It becomes reachable if
`blockDim.x < NUM_EXPERTS` (the loop then makes multiple passes and a later pass reads slots an
earlier pass may have written) **and** `n_active > blockDim.x`. Worth a comment or a
compact-into-a-separate-buffer fix the next time that file is touched; not urgent.
