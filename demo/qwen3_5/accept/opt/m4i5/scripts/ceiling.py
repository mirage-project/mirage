#!/usr/bin/env python3
"""M4-I5 -- the packed-to-128 ceiling, as a bound with its assumptions.

Two independent bounds on the decode step, both computed from the compiled task
graph plus measured per-task times, so neither is a wish:

A. WORK BOUND.  total task-us / 128.  Unreachable in practice (it assumes zero
   dependency structure) but it says how far the step is from arithmetic.

B. LATENCY BOUND (critical path).  `critpath.py` walks the compiled DAG -- task
   T waits on event `T.dependent_event` and on finishing triggers
   `T.trigger_event` -- and returns the longest weighted chain, which is a floor
   no amount of width removes.  Reported beside this model, not folded into it.

C. WAVE-DEPTH STAGE MODEL.  Between the two bounds sits the thing a split
   actually changes.  MPK hands the tasks of one event to workers in order --
   task at position j of the launch goes to worker `j % 128`, and each worker
   drains its queue in order (persistent_kernel.cuh EVENT_LAUNCH_DEPENDENT_TASKS
   / M3-I8 model_moe_wall.py C3) -- and live tasks are a contiguous prefix of a
   grouped-GEMM call site.  So a level costs
       max_w [ live_w * T_live + dead_w * T_dead ]
   and splitting a stage k ways multiplies both the emitted and the live count
   by k while dividing T_live by k.  That is why k=4 was NEUTRAL at bs8 in
   M3-I8 (it doubles the depth and halves the task) and why k=8 need not be.
   The model is calibrated per stage on the MEASURED span at the current k, and
   the calibration factor is printed, so a stage whose model does not fit is
   visible rather than silently extrapolated.

Split admissibility is source-derived, not assumed -- see SPLITS below.
"""
from __future__ import annotations

import argparse
import json

NW = 128

# task_type -> (knob, current_k, max_k, needs_merge, fixed_us, limit)
#
# `fixed_us` is the part of a task's latency that does NOT divide by the split
# factor.  It is a MEASURED number or 0 with a citation -- never a guess:
#   * MoE grouped GEMMs: 0.  M3-I8 fitted T = 0.93 us * (N_tile/128) * (K/128)
#     across a 1.55x range in live-task count on both stages to 5% with no
#     intercept (model_moe_wall.py C2), and grid.y divides N_tile exactly.
#   * attention: M3-I6a measured the intercept directly -- per-task latency is
#     `fixed + per_KV_token * ctx` with R^2 >= 0.998, fixed = 29.63 us at bs1
#     and 26.91 us at bs8 (pass size 2, the shipped value).  Only the ctx term
#     divides by a KV split, which is exactly what its recorded model says
#     (29.6 + 0.0536*ctx/k + merge).
#   * the small per-row stages: the dead-task cost, i.e. the queue pop +
#     dependency check + two profiler stores that every task pays whatever it
#     does.  Taken from this run's own measured t_dead.
#   * the router: NOT measured.  Reported at fixed = 0 and flagged, because its
#     epilogue (the cross-row expert compaction) is the unmeasured part.
# Every entry is a source citation, not an estimate.
SPLITS = {
    241: ("moe_n_splits (grid.y)", 2, 8, False, 0.0,
          "moe_fp8_blockscale_sm100.cuh static_asserts OUTPUT_SIZE % 128 == 0 "
          "(one whole 128-row scale block); w13 OUTPUT_SIZE = 2*inter = 1024, "
          "so 1024/8 = 128 is the finest legal slice. grid.y partitions OUTPUT "
          "COLUMNS -> disjoint outputs, K reduction stays inside one task, no "
          "cross-task reduction exists, so NO merge and no barrier."),
    242: ("moe_n_splits (grid.y)", 2, 16, False, 0.0,
          "same static_assert; w2 OUTPUT_SIZE = hidden = 2048 -> 2048/16 = 128. "
          "Shares the knob with 241, so a single knob caps at 8."),
    279: ("fp8_grid(N) = N/128 (grid.x)", 1, 1, True, 0.0,
          "ALREADY at the finest legal N split: the per-task N slice must be a "
          "whole number of 128-row scale blocks (linear_fp8_blockscale_sm100"
          ".cuh:120) and grid.x splits weight rows and scale dim 0 together "
          "(persistent_kernel.py:2059-2060). Widening further means splitting "
          "K, which is a cross-task REDUCTION -> needs the atomic "
          "last-arriving-task merge (M3-I3 idiom), a real kernel change."),
    253: ("grid_for_rmsnorm_linear_layer", 1, 1, True, 0.0,
          "grid is a hardcoded 96/64, or size//256 above the 400*96 threshold, "
          "in a util SHARED with every other MPK model (models/utils.py:3). The "
          "256 cap is a deliberate workaround for size-dependent "
          "nondeterminism (M3-I11 partly root-caused it); relaxing it changes "
          "the per-task tile shape and needs its own bit-exactness run."),
    260: ("grid_dim=(1,1,1)", 1, 16, True, 0.0,
          "ONE task per layer. Splittable over the mbt=16 router rows (each row"
          "'s top-k is a per-row reduction over its own 256 logits), but the "
          "`routing`/`moe_mask` outputs are a COMPACTION across experts over "
          "all rows, so the last-arriving task must run it -> needs the M3-I3 "
          "atomic epilogue. M3-I5c already fixed a compaction race here."),
    257: ("grid_dim=(mbr, kv_heads)", 1, 8, True, "ATTN_FIXED",
          "split-KV over the KV range; TASK_PAGED_ATTENTION_SPLIT_KV_SM100 is "
          "the in-tree precedent and M3-I6a recorded the model "
          "(29.6 + 0.0536*ctx/k + merge). Needs a partial-output merge -> the "
          "M3-I3 atomic last-block epilogue."),
    275: ("grid_dim=(mbt,1,1) + QUANTIZE_ROW_SPLIT", 1, 8, False, "T_DEAD",
          "one task per token row already (M3-I8 v1). Finer means splitting the "
          "128-element scale groups across tasks -- legal (a group's fp8 bytes "
          "and its fp32 block scale come from that group alone) but the tasks "
          "are already 4.2 us, so per-task fixed cost dominates."),
    261: ("grid_dim=(mbt,1,1)", 1, 8, False, "T_DEAD",
          "one task per token row; the topk-weighted sum over `topk` slots and "
          "the residual add are per-row, so the hidden axis can be split with "
          "disjoint outputs and no merge."),
    238: ("grid_dim=(mbt,1,1)", 1, 8, False, "T_DEAD",
          "one task per token row; sigmoid-gate * mul + add is elementwise over "
          "the hidden axis -> splittable with disjoint outputs, no merge."),
    154: ("grid_for_rmsnorm_linear_layer", 1, 1, True, 0.0,
          "RMS norm needs the whole row's sum of squares; splitting the row is "
          "a cross-task reduction."),
    234: ("gdn_conv_channel_blocks", 8, 16, False, "T_DEAD",
          "grid (mbr, channel_blocks); channels are independent in a causal "
          "conv1d, so raising the block count is merge-free. conv_dim/blocks "
          "must stay a whole tile."),
    237: ("gdn_split (grid.z)", 4, 4, True, 0.0,
          "ALREADY split 4x by M3-I3 with the atomic last-block epilogue. "
          "Further splitting needs another scratch input and MAX_INPUTS_PER_TASK "
          "is 7 with 6 inputs + 1 output already used (the o-partials and the "
          "arrival counter already share one buffer for this reason)."),
}


def level_cost(E, n, t_live, t_dead, k=1, fixed=0.0):
    """Cost of one dependency level with split factor k, from MPK's dispatch:
    position j of the launch goes to worker j % 128, workers drain in order,
    live tasks are a contiguous prefix.  `fixed` is the part of t_live that a
    split does NOT divide; it is floored at t_dead because no task can cost
    less than a queue pop plus a dependency check."""
    tk = t_live if k == 1 else max(fixed, t_dead) + (t_live - fixed) / k
    Ek, nk = E * k, n * k
    worst = 0.0
    for w in range(NW):
        tot = (Ek - w + NW - 1) // NW if Ek > w else 0
        live = (nk - w + NW - 1) // NW if nk > w else 0
        worst = max(worst, live * tk + max(tot - live, 0) * t_dead)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("width_json", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    report = {}
    for wp in a.width_json:
        w = json.load(open(wp))
        bs = w["batch_size"]
        step = w["step_us"]
        stages = {r["task_type"]: r for r in w["stages"]}

        # per-level dispatch overhead, measured on the stages that are already
        # packed (live_per_level >= 128): span/levels - T_live.
        oh = []
        for r in w["stages"]:
            if (r["levels"] or 0) and (r["live_per_level"] or 0) >= NW:
                oh.append(r["span_us"] / r["levels"]
                          - level_cost(r["emitted_per_level"], r["live_per_level"],
                                       r["t_live_us"], r["t_dead_us"]))
        # NOT a measured dispatch latency -- it is the residual by which the
        # wave-depth model under-reads an already-packed stage, reported so the
        # model's error is visible instead of folded into a ceiling.
        per_level_model_residual = round(sum(oh) / len(oh), 3) if oh else None

        rows = []
        attn_fixed = 29.63 if bs == 1 else 26.91   # M3-I6a, pass size 2
        for tt, (knob, cur_k, max_k, merge, fx, limit) in SPLITS.items():
            r = stages.get(tt)
            if not r or not r["levels"]:
                continue
            E = r["emitted_per_level"]
            n = r["live_per_level"]
            tl, td = r["t_live_us"] or r["t_all_us"], r["t_dead_us"]
            L = r["levels"]
            fixed = (attn_fixed if fx == "ATTN_FIXED"
                     else (td if fx == "T_DEAD" else float(fx)))
            model1 = L * level_cost(E, n, tl, td)
            calib = r["span_us"] / model1 if model1 else 0.0
            ks, kk = [], 1
            while cur_k * kk <= max_k:
                ks.append(kk)
                kk *= 2
            ks = ks or [1]
            preds = {}
            for kk in ks:
                if kk < 1:
                    continue
                preds[cur_k * kk] = round(
                    calib * L * level_cost(E, n, tl, td, kk, fixed), 1)
            best_k = min(preds, key=lambda x: preds[x])
            dspan = r["span_us"] - preds[best_k]
            rows.append(dict(
                task_type=tt, name=r["name"], knob=knob,
                levels=L, emitted_per_level=E, live_per_level=n,
                t_live_us=tl, t_dead_us=td,
                span_measured_us=r["span_us"], span_model_us=round(model1, 1),
                model_calibration=round(calib, 3),
                current_k=cur_k, max_k=max_k, needs_merge=merge,
                fixed_us=round(fixed, 3),
                fixed_source=("M3-I6a measured intercept" if fx == "ATTN_FIXED"
                              else "measured dead-task cost" if fx == "T_DEAD"
                              else "0 (M3-I8 fit has no intercept)"
                              if tt in (241, 242) else "0 -- UNMEASURED"),
                limited_by=limit,
                span_at_k=preds, best_k=best_k,
                d_span_us=round(dspan, 1),
                sole_fraction=round(r["sole_us"] / r["span_us"], 3)
                if r["span_us"] else 0.0,
                d_step_lower_us=round(dspan * (r["sole_us"] / r["span_us"]), 1)
                if r["span_us"] else 0.0,
                d_step_upper_us=round(dspan, 1),
                sole_idle_us=r["sole_idle_us"]))
        rows.sort(key=lambda x: -x["d_step_lower_us"])

        nomerge = [r for r in rows if not r["needs_merge"]]
        allr = rows
        def tot(rs, key):
            return round(sum(x[key] for x in rs), 1)

        report[str(bs)] = dict(
            step_us=step,
            work_bound_us=w["machine"]["work_bound_us"],
            occupancy=w["machine"]["occupancy"],
            us_at_conc_le_16=round(w["machine"]["us_by_band"]["0-0"]
                                   + w["machine"]["us_by_band"]["1-16"], 1),
            total_sole_us=round(sum(r["sole_us"] for r in w["stages"]), 1),
            total_sole_idle_us=round(sum(r["sole_idle_us"] for r in w["stages"]), 1),
            per_level_model_residual_us=per_level_model_residual,
            n_dependency_levels=sum(r["levels"] or 0 for r in w["stages"]),
            merge_free_ceiling=dict(
                d_step_lower_us=tot(nomerge, "d_step_lower_us"),
                d_step_upper_us=tot(nomerge, "d_step_upper_us"),
                step_lower=round(step - tot(nomerge, "d_step_upper_us"), 1),
                step_upper=round(step - tot(nomerge, "d_step_lower_us"), 1)),
            all_splits_ceiling=dict(
                d_step_lower_us=tot(allr, "d_step_lower_us"),
                d_step_upper_us=tot(allr, "d_step_upper_us"),
                step_lower=round(step - tot(allr, "d_step_upper_us"), 1),
                step_upper=round(step - tot(allr, "d_step_lower_us"), 1)),
            stages=rows)

    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)

    for bs, r in report.items():
        print("=" * 100)
        print(f"bs{bs}  step {r['step_us']} us   occupancy {r['occupancy']}   "
              f"work bound {r['work_bound_us']} us   "
              f"time at conc<=16: {r['us_at_conc_le_16']} us")
        print(f"  one-stage-only time {r['total_sole_us']} us, of which "
              f"{r['total_sole_idle_us']} us is idle machine;  "
              f"{r['n_dependency_levels']} dependency levels; "
              f"model residual {r['per_level_model_residual_us']} us/level")
        print(f"  {'stage':30s}{'lvl':>4s}{'E/lvl':>7s}{'live':>7s}{'T':>7s}"
              f"{'span':>8s}{'calib':>6s}{'bestk':>6s}{'span@k':>8s}"
              f"{'dstep_lo':>9s}{'fix':>6s}{'merge':>6s}")
        for s in r["stages"]:
            print(f"  {s['name'][:30]:30s}{s['levels']:4d}{s['emitted_per_level']:7.1f}"
                  f"{s['live_per_level']:7.1f}{s['t_live_us']:7.2f}"
                  f"{s['span_measured_us']:8.1f}{s['model_calibration']:6.2f}"
                  f"{s['best_k']:6d}{s['span_at_k'][s['best_k']]:8.1f}"
                  f"{s['d_step_lower_us']:9.1f}{s['fixed_us']:6.1f}"
                  f"{'Y' if s['needs_merge'] else 'n':>6s}")
        m, al = r["merge_free_ceiling"], r["all_splits_ceiling"]
        print(f"  MERGE-FREE ceiling: step {r['step_us']} -> "
              f"{m['step_lower']}..{m['step_upper']} us  "
              f"(x{r['step_us']/m['step_upper']:.3f}..x{r['step_us']/m['step_lower']:.3f})")
        print(f"  ALL-SPLITS ceiling: step {r['step_us']} -> "
              f"{al['step_lower']}..{al['step_upper']} us  "
              f"(x{r['step_us']/al['step_upper']:.3f}..x{r['step_us']/al['step_lower']:.3f})")


if __name__ == "__main__":
    main()
