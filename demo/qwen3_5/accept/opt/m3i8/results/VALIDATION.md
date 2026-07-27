# M3-I8 validation — window run 2026-07-27 (B200), coordinator close report

Pre-registered plan: `../predictions.md` (committed 46872ad, BEFORE any measurement) +
`../plan_m3i8.sh`. Raw evidence in this directory; analyzer output `analyze_final2.log`.
The change under test: `gate_padding_rows` — the MoE router marks experts for LIVE rows only
(v1, default ON in the qwen3_5 builder since 46872ad; base arm = `MOE_GATE_PADDING_ROWS=False`).

## Step-time A/B (median, warmup + >=3 reps, no-profiler lane)

| bs | base step µs | v1 step µs | Δ | pre-registered band |
|---:|---:|---:|---:|---|
| 1  | 11715 | 10676 | **+9.7%**  | +3% (0..+9) |
| 2  | 12474 | 10823 | **+15.3%** | +5% (0..+12) |
| 4  | 11868 | 11080 | **+7.1%**  | +3% (0..+9) |
| 8  | 15610 | 12479 | **+25.1%** | +17% (+12..+24) |
| 16 | 18736 | 18739 | **-0.0%**  | +1% (0..+2, inert) |

4/5 at or above the central prediction; bs2/bs8 land just past the optimistic bound; bs16
inert exactly as predicted (nothing to remove at live=16). F2 (bs8 >= 10%) does not fire.
Mechanism detail (from `analyze_final2.log`): the win is wave-count + tail — moe_w13 at bs8
goes 2 waves -> 1 (117.3 -> 66.2 µs/layer), moe_w2 likewise (62.6 -> 36.3 µs/layer),
confirming the corrected worker-DEPTH cost model from the prep, not the group-count model.

## Correctness (the binding gate): AC-3 PASS

- v1 arm: per-case byte-diff vs the committed M2 dumps (`results/dumps_final`) — **identical,
  10/10 prompts at all five batch sizes** (`bytediff_v1.json`).
- bs16 inert control: byte-identical (`bytediff_v1_bs16.json`).
- Qwen3-8B CI smoke: OK (3.9 ms/tok; smoke status per the M3-I2a ruling).
- The one pre-existing M2-era adjudicated reference-side tie (p06-poem, pos 60) recurs
  UNCHANGED at every bs — present in base and v1 alike, not introduced by I8.

## F1 mechanism oracle — instrument honesty

The pre-registered primary instrument `mask_probe.py` is BROKEN (its offline single-shot
admission never reaches a correct live-row count at any bs — even bs16 where gating is a
no-op; `--new-tokens 64` reproduces identically, ruling out "needs more iterations").
Substitute instrument: M3-I1's profiler-inference (activated = nlong / (40 x moe_n_splits)),
calibrated against I1 on the base arm to within ±0.3 (56.6/59.6/60.5/69.9/87.0 vs I1's
56.4/59.4/60.2/70.1/86.7).

| bs | base activated | v1 activated | pure-decode cap 8·live | strict check |
|---:|---:|---:|---:|---|
| 1  | 56.6 | 9.6  | 8   | over by 1.6 |
| 2  | 59.6 | 16.9 | 16  | over by 0.9 |
| 4  | 60.5 | 30.3 | 32  | under |
| 8  | 69.9 | 50.4 | 64  | under |
| 16 | 87.0 | 87.0 | 128 | n/a (inert) |

The analyzer's strict-cap lines print `FALSIFIED (F1)` at bs1/bs2 (and v2a bs1). Read
carefully, that label overstates: the gate plainly took effect (56.6 -> 9.6 at bs1, an 83%
reduction landing near the predicted 8.0). The substitute instrument averages over the WHOLE
profiled run, and prefill iterations run at live = chunk (up to 16) with ~87 activated — so a
run-average is biased ABOVE the pure-decode cap at small bs by construction (bs1: two prefill
iterations at ~87 mixed into 64 decode iterations at 8 already yields ~10.4). The bias
vanishes where prefill and decode activate similarly (bs16: 87.0 == 87.0). The excess is
therefore attributed to instrument granularity, not to the gate leaking marks; and the
decisive question — does gating ever corrupt a live row — is answered directly by the AC-3
byte-diff above. A pure-decode-window isolation re-run can ride the next GPU window if the
reviewer wants the strict oracle closed; it does not gate this issue's close.

## v2 arms (grid widen — STAGED, not applied; patch in `../v2-moe-grid-widen.patch`)

- bs1 composition v1+v2a / v1+v2b: **+24.6% / +24.7%** vs base (beats the predicted
  +10.4..+22.8 band).
- bs8 v1+v2a: **+19.6%** — WORSE than v1 alone (+25.1%): widening splits the grid and pushes
  bs8 back from 1 wave to 2, exactly the wave-cost model's warning. v2 is therefore
  bs1-conditional; adopting it as a default would need per-bs codegen or a bs-conditional
  grid rule. Disposition deferred to the I7 re-rank (recorded in `../../backlog.json`).
- v2a's own AC-3 was not run (blocked by the same mask_probe bug; v2 ships nothing, so this
  blocks nothing).

## Window hygiene

Zero hangs (the bs2 host-spin watch item did not recur, 0/many). Plan gaps found and
backfilled by the runner: venv missing pip (ensurepip), missing `hatchling`, stage0 missing
`MPK_ACCEPT_DIR`, stage1 missing `mkdir masks/` (cost 3 wasted base compiles); 2 transient
GPU-guard refusals on the contended box (rc=97), both backfilled. Repo restored to clean
d19c8d88; no lingering processes.

## Implied decode throughput after v1 (steady-decode step rate, bs·1e6/step_µs)

93.7 / 184.8 / 361.0 / 641.1 tok/s at bs 1/2/4/8 (bs16 stays 274.7 e2e — wave degeneracy,
owned by the I9 cap policy). Gaps to vLLM: 3.05x / 2.87x / 2.59x / 2.64x / 11.0x.
End-to-end confirmation lands at the I7 integration gate.
