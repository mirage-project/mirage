#!/usr/bin/env python3
"""M3-I10 remeasure closure (codex c2 finding F1): regenerate ferret_targets.json
so PRIMARY fields carry the matched-geometry (current HEAD) numbers, not M3-I1's.

Why this exists (not another extend_ferret.py append): the committed file's
consumers read `mpk_us_per_step` / `ratio_mpk_over_vllm` / `rank` directly --
appending a `remeasure_update_*` annotation string next to a stale primary
field doesn't change what a machine-reading consumer sees. This script instead:

  1. Reads the CURRENT ferret_targets.json (any hand-authored prose, shape
     descriptions, roofline physics -- all vLLM/architecture facts that do NOT
     depend on MPK's remeasure -- are preserved verbatim).
  2. Reads the regenerated matched-geometry join
     (opt/m3i10/remeasure/armA_m3i10/tables/comparison_by_stage.csv) as the
     ONE source of truth for new mpk_us_per_step / vllm_us_per_step / ratio /
     abs_gap, per stage per bs.
  3. Reads the per-call-site splits (opt/m3i10/remeasure/qc/armA_bs*_qc.json)
     for task 253 and task 279's shape-level detail.
  4. Rewrites every one of the 15 stages: PRIMARY fields = new numbers;
     everything from the old primary fields moves verbatim into a
     `history_m3i1` sub-object (nothing is lost, nothing is silently stale).
  5. Recomputes rank (targets only, sorted by NEW bs1 abs_gap_us_step,
     descending -- documented in `basis.rank_rule` since comparison_by_stage.csv
     itself carries no rank column).
  6. quantize moves from `targets` to `dispositions` (disposition
     "resolved-by-I2b"): its gap collapsed to ~0/negative, so it is no longer a
     viable ferret target, with the code-delta-isolated collapse evidence
     attached.
  7. shared-expert-gate's bs8 disposition text is updated from "flagged,
     re-measure will settle it" to "confirmed capture artifact, does not
     reproduce" with the new monotonic numbers.
  8. Coverage assertion re-run against the SAME 15-stage set (unchanged: this
     remeasure did not add or remove a stage from the vLLM-side table).

Usage: python3 regenerate_ferret_v2.py
    (lives beside opt/m3i10/scripts/extend_ferret.py; paths below are relative
    to opt/m3i10/ -- i.e. this file's grandparent -- hardcoded, not
    parameterised, since this is meant to be re-run from a clean checkout)
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # .../opt/m3i10 (script lives in scripts/)
FT_PATH = HERE / "ferret_targets.json"
CMP_PATH = HERE / "remeasure" / "armA_m3i10" / "tables" / "comparison_by_stage.csv"
QC_DIR = HERE / "remeasure" / "qc"
LATECTX_DIR = HERE / "remeasure" / "armAlate" / "tables"
BSES = ["1", "8", "16"]
GENERATED_UTC = "2026-07-27"
GENERATOR = "demo/qwen3_5/accept/opt/m3i10/scripts/regenerate_ferret_v2.py"

# mpk_task string -> matching mpk task_type int(s), for pulling the right
# comparison_by_stage.csv row and (for 253/279) the call-site split.
STAGE_TASK_TYPES = {
    "GDN recurrent (delta rule)": [237],
    "quantize / fp8 casts": [275],
    "MoE routed GEMM w13 (gate_up)": [241],
    "dense projections (fp8 blockscale)": [279],
    "MoE routed GEMM w2 (down)": [242],
    "full attention": [257],
    "MoE router top-k/softmax": [260],
    "GDN conv1d": [234],
    "dense projections (bf16 small + lm_head)": [253],
    "sampling / argmax": [259, 258],
    "embedding": [101],
    "MoE combine (weighted sum + residual)": [261],
    "MoE/shared SiLU-mul": [118],
    "norms / RoPE / glue": [154],
    "shared-expert gate (sigmoid*shared+residual)": [238],
}
# reverse: old target's mpk_task string prefix -> stage name in comparison_by_stage.csv
TARGET_STAGE_BY_TASK_PREFIX = {
    "237": "GDN recurrent (delta rule)",
    "275": "quantize / fp8 casts",
    "241": "MoE routed GEMM w13 (gate_up)",
    "279": "dense projections (fp8 blockscale)",
    "242": "MoE routed GEMM w2 (down)",
    "257": "full attention",
    "260": "MoE router top-k/softmax",
    "234": "GDN conv1d",
    "253": "dense projections (bf16 small + lm_head)",
}
TARGET_RATIO = 0.75  # single documented factor (spec: 0.70-0.80); see basis.target_rule


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_comparison():
    rows = {r["stage"]: r for r in csv.DictReader(open(CMP_PATH))}
    return rows


LATECTX_REGIME = {  # from opt_fixed/schedule_sim.py's steady_window() on the
                    # msl=897 capture -- see remeasure_spec.md sec 4(d)
    "1": dict(iteration=335, n_live=1, window="[560,656)", context_approx="~801-896"),
    "8": dict(iteration=413, n_live=8, window="[560,656)", context_approx="~801-896"),
    "16": dict(iteration=726, n_live=12, window="[720,733)",
              context_approx=("263-890 PER-SLOT, NOT a tight high-context band "
                              "(exact per-slot step+1 from schedule_sim.simulate: "
                              "[890,867,842,813,781,745,702,651,587,502,374,263], "
                              "the 4 already-retired slots omitted) -- bs16's "
                              "staggered admission means its 12 concurrent "
                              "survivors are never all at similar context "
                              "simultaneously, unlike bs1 (single request) or "
                              "bs8 (full 8-wide window, uniform ~801-896). No "
                              "prefill-free full-bs16 regime exists at ANY "
                              "context (structural, same reason M3-I1 documented "
                              "for the original bs16 steady-state). Treat bs16's "
                              "late-context number as directionally consistent "
                              "but NOT a clean single-context measurement.")),
}
LATECTX_TASK_NAME = {237: "TASK_GDN_RECURRENT_SM100", 279: "TASK_LINEAR_FP8_BLOCKSCALE_SM100",
                      241: "TASK_MOE_W13_FP8_BLOCKSCALE_SM100",
                      257: "TASK_ATTN_SM100", 275: "TASK_QUANTIZE_FP8_SM100"}


def load_latectx_wallspans():
    """{task_type: {bs: wall_span_us}} from the msl=897 (context ~556-896)
    closure capture's concurrency.json (its regime detection needed no
    warm/steady override -- verified against opt_fixed/schedule_sim.py's raw
    steady_window() per bs, see LATECTX_REGIME)."""
    out = {tt: {} for tt in LATECTX_TASK_NAME}
    for bs in BSES:
        p = LATECTX_DIR / f"bs{bs}_concurrency.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        pcc = d.get("per_task_concurrency", {})
        for tt, name in LATECTX_TASK_NAME.items():
            if name in pcc:
                out[tt][bs] = pcc[name]["wall_span_us"]
    return out


def load_site_splits():
    """{task_type: {bs: [{site, sum_us_per_iter, wallspan_us_per_iter, mean_us, n_per_iter}]}}"""
    out = {253: {}, 279: {}}
    for bs in BSES:
        p = QC_DIR / f"armA_bs{bs}_rep0_qc.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for tt_str, blob in d.get("call_site_split", {}).items():
            tt = int(tt_str)
            if tt not in out or "sites" not in blob:
                continue
            out[tt][bs] = blob["sites"]
    return out


def new_numbers(stage_name, cmp_rows):
    row = cmp_rows[stage_name]
    mpk = {bs: f(row[f"mpk_us_step_bs{bs}"]) for bs in BSES}
    vllm = {bs: f(row[f"vllm_us_step_bs{bs}"]) for bs in BSES}
    ratio = {bs: f(row[f"ratio_bs{bs}"]) for bs in BSES}
    gap = {bs: f(row[f"abs_gap_us_step_bs{bs}"]) for bs in BSES}
    return mpk, vllm, ratio, gap


def build_history(old_row, keys):
    """Extract M3-I1 history from `old_row`. Idempotent: if `old_row` already
    carries a `history_m3i1` (i.e. this script is being re-run against its own
    prior output, where old_row's top-level fields are already the matched-
    geometry numbers, not M3-I1's), return that preserved sub-object as-is
    rather than re-deriving history from the now-current primary fields --
    otherwise a second run would silently overwrite real M3-I1 history with a
    duplicate of the matched-geometry numbers."""
    if "history_m3i1" in old_row:
        return old_row["history_m3i1"]
    return {k: old_row[k] for k in keys if k in old_row}


def make_call_site_block(sites_by_bs):
    """sites_by_bs: {bs: [site rows]} -> {site_name: {bs: {sum_us_per_step,
    wallspan_us_per_step, mean_us_per_call}}}"""
    by_site = {}
    for bs, sites in sites_by_bs.items():
        for s in sites:
            e = by_site.setdefault(s["site"], {})
            e[bs] = dict(sum_us_per_step=s["sum_us_per_iter"],
                        wallspan_us_per_step=s["wallspan_us_per_iter"],
                        mean_us_per_call=s["mean_us"], n_per_step=s["n_per_iter"])
    return by_site


def main():
    old = json.loads(FT_PATH.read_text())
    cmp_rows = load_comparison()
    site_splits = load_site_splits()
    latectx = load_latectx_wallspans()
    old_targets_by_stage = {TARGET_STAGE_BY_TASK_PREFIX[t["mpk_task"].split()[0]]: t
                            for t in old["targets"]}
    old_disp_by_stage = {d["stage"]: d for d in old["dispositions"]}

    HIST_TARGET_KEYS = ["mpk_us_per_step", "ratio_mpk_over_vllm", "target_us_per_step",
                        "step_gain_if_met_us", "rank", "vllm_sub_split_bs1_us"]
    HIST_DISP_KEYS = ["mpk_us_per_step", "vllm_us_per_step", "ratio_mpk_over_vllm",
                      "step_gain_if_met_us"]

    new_targets = []
    new_dispositions = []

    for stage_name in STAGE_TASK_TYPES:
        mpk, vllm, ratio, gap = new_numbers(stage_name, cmp_rows)
        is_old_target = stage_name in old_targets_by_stage
        is_old_disp = stage_name in old_disp_by_stage
        assert is_old_target != is_old_disp, stage_name

        # --- decide the NEW category ------------------------------------
        if stage_name == "quantize / fp8 casts":
            # special-cased regardless of old category: collapsed by I2b.
            # Idempotent re-run support: on a first run this stage is still in
            # `targets` (M3-I1-era file); on a re-run it is already in
            # `dispositions` with its M3-I1 numbers preserved under
            # history_m3i1 -- read whichever is present so this script can be
            # re-run safely against its own prior output.
            if stage_name in old_targets_by_stage:
                old_row = old_targets_by_stage[stage_name]
                hist = build_history(old_row, HIST_TARGET_KEYS)
            else:
                old_row = dict(old_disp_by_stage[stage_name])
                hist = old_row.get("history_m3i1", {})
            row = {k: old_row[k] for k in
                   ("mpk_task", "vllm_kernel", "shape", "sites_per_step", "roofline",
                    "roofline_reading") if k in old_row}
            row.update(dict(
                stage=stage_name,
                disposition="resolved-by-I2b",
                vllm_us_per_step={bs: vllm[bs] for bs in BSES},
                mpk_us_per_step={bs: mpk[bs] for bs in BSES},
                ratio_mpk_over_vllm={bs: ratio[bs] for bs in BSES},
                step_gain_if_met_us={bs: gap[bs] for bs in BSES},
                reason=("RESOLVED by M3-I2b's quantize-redundancy fix, confirmed at "
                        "matched geometry / current HEAD (anchor-QC exact, "
                        "max_frac_err=0.0000 at every bs). Wallspan collapsed "
                        f"{hist['mpk_us_per_step']['bs1']:.0f}->{mpk['1']:.1f} us/step "
                        "at bs1 (ratio 8.12x->1.00x); code-delta isolation "
                        "(committed M3-I1 vs this remeasure's arm B, SAME msl=132 "
                        "geometry, current HEAD) attributes the full move to the code "
                        "change -- geometry delta (arm B -> arm A) on top of that is "
                        "negligible (562->560 us/step at bs1). No ferret dispatch: "
                        "there is nothing left to buy. See opt/m3i10/remeasure/ "
                        "(pertask_by_bs.csv code-delta columns) for the arm-B-vs-M3-I1 "
                        "isolation."),
                history_m3i1=hist,
            ))
            new_dispositions.append(row)
            continue

        if is_old_target:
            old_row = old_targets_by_stage[stage_name]
            hist = build_history(old_row, HIST_TARGET_KEYS)
            row = {k: old_row[k] for k in
                   ("mpk_task", "vllm_kernel", "why", "shape", "sites_per_step",
                    "vllm_us_per_call", "note", "roofline", "roofline_reading")
                   if k in old_row}
            row.pop("remeasure_update_20260727", None)

            # I10-fix-1 (c3): attention's PRIMARY basis is late-context
            # (opt/m3i10/remeasure/armAlate/, ctx ~801-896) -- that is what
            # this file's own late_context_verdict already documented as
            # correct, so the primary fields must actually use it, not just
            # say so in a side note. Every other target keeps the
            # matched-geometry (ctx 257-352) basis as primary.
            tt0 = STAGE_TASK_TYPES[stage_name][0]
            is_attention = (stage_name == "full attention")
            if is_attention and latectx.get(tt0):
                primary_mpk = dict(latectx[tt0])
                primary_ratio = {bs: round(primary_mpk[bs] / vllm[bs], 3)
                                 for bs in BSES if vllm.get(bs)}
                primary_gap = {bs: round(primary_mpk[bs] - vllm[bs], 1)
                               for bs in BSES if vllm.get(bs)}
            else:
                primary_mpk, primary_ratio, primary_gap = mpk, ratio, gap

            row.update(dict(
                stage=stage_name,
                vllm_us_per_step={bs: vllm[bs] for bs in BSES},
                mpk_us_per_step={bs: primary_mpk[bs] for bs in BSES},
                ratio_mpk_over_vllm={bs: primary_ratio[bs] for bs in BSES},
                target_us_per_step={bs: round(vllm[bs] * TARGET_RATIO, 1) for bs in BSES},
                step_gain_if_met_us={bs: round(primary_mpk[bs] - vllm[bs] * TARGET_RATIO, 1)
                                    for bs in BSES},
                history_m3i1=hist,
            ))

            if is_attention:
                row["context_band"] = {
                    "primary_basis": "late-context (opt/m3i10/remeasure/armAlate/), "
                                     "ctx ~801-896 at bs1/bs8",
                    "bs1": "801-896 (single request, clean)",
                    "bs8": "801-896 (full 8-concurrent decode_full window, clean)",
                    "bs16": "263-890 (STAGGERED: 12/16 concurrent survivors spread "
                            "across this range at the chosen snapshot, not a tight "
                            "band -- no full-bs16 prefill-free window exists at ANY "
                            "context, structural, same reason M3-I1 documented for "
                            "the original bs16 steady-state; per-slot context array "
                            "in matched_window.late_context_regime.16). Treat bs16's "
                            "primary numbers here as directionally right, not as "
                            "precise as bs1/bs8.",
                    "why_primary": "this is the context band the vLLM reference table "
                                   "itself was sampled at (556-896) -- comparing MPK's "
                                   "OWN matched-geometry window (257-352) against "
                                   "vLLM's 556-896 window was exactly the F2 mismatch; "
                                   "late-context is the apples-to-apples basis.",
                }
                row["matched_window"] = dict(
                    description=("arm A's own matched-geometry window, ctx 257-352 -- "
                                "PREVIOUS primary basis, kept for continuity, not used "
                                "for ranking/target math above (see context_band)."),
                    mpk_us_per_step=mpk, ratio_mpk_over_vllm=ratio,
                    step_gain_if_met_us=gap,
                    late_context_regime=LATECTX_REGIME,
                    pct_change_matched_to_late={
                        bs: round(100 * (primary_mpk[bs] - mpk[bs]) / mpk[bs], 1)
                        for bs in BSES if mpk.get(bs)},
                )
                row["late_context_verdict"] = (
                    "CONFIRMED real and larger than the old single-FMHA-kernel +8.3% "
                    "correction implied, and now the PRIMARY basis above (not just a "
                    "side note): wallspan grew "
                    + ", ".join(f"{bs}={row['matched_window']['pct_change_matched_to_late'].get(bs, '?')}%"
                               for bs in BSES) +
                    " moving from arm A's own context (~257-352, now in "
                    "matched_window) to the vLLM reference table's own sampled band "
                    "(556-896, this row's primary fields at ~801-896). bs1/bs8 are "
                    "clean, single-context (bs1) or uniform full-8-concurrent (bs8) "
                    "measurements. bs16 carries a real caveat -- see context_band. See "
                    "remeasure_spec.md sec 4(d) and opt/m3i10/remeasure/armAlate/.")
            else:
                # F2 closure: late-context spot check for every OTHER target
                # task that has one -- informational only, primary basis
                # unchanged (matched-geometry).
                if latectx.get(tt0):
                    lc = latectx[tt0]
                    row["late_context_check_msl897"] = dict(
                        wallspan_us_per_step=lc,
                        pct_change_vs_matched_geometry={
                            bs: round(100 * (lc[bs] - mpk[bs]) / mpk[bs], 1)
                            for bs in lc if mpk.get(bs)},
                        regime=LATECTX_REGIME,
                    )
            # per-call-site detail for 253 / 279
            tt = STAGE_TASK_TYPES[stage_name][0]
            if tt in (253, 279) and site_splits.get(tt):
                row["call_site_split_matched_geometry"] = make_call_site_block(
                    site_splits[tt])
                if tt == 253:
                    # the specific closure the HELD status needed: per-site
                    # ratio against vLLM's own bs1 sub-split.
                    vsub = old_row.get("vllm_sub_split_bs1_us", {})
                    site_bs1 = {s["site"]: s for s in site_splits[253].get("1", [])}
                    ratios = {}
                    name_map = {"lm_head": "lm_head", "moe_router_gate": "MoE router gate",
                               "in_proj_ba": "in_proj_ba"}
                    for k, vname in name_map.items():
                        if k in site_bs1 and vname in vsub and vsub[vname]:
                            ratios[k] = round(
                                site_bs1[k]["wallspan_us_per_iter"] / vsub[vname], 3)
                    row["call_site_ratio_vs_vllm_bs1"] = ratios
                    row["held_status_resolution"] = (
                        "RESOLVED. lm_head ratio " + str(ratios.get("lm_head")) +
                        "x (near parity -- both sides roofline-bound, SKIP as a ferret "
                        "target); MoE router-gate " + str(ratios.get("moe_router_gate")) +
                        "x and in_proj_ba " + str(ratios.get("in_proj_ba")) +
                        "x carry essentially all of the family's gap and all of the "
                        "headroom (matching the vLLM-side note that the 458.6 us of "
                        "small bf16 GEMMs + splitK reduce is where the room is). "
                        "Split extraction method: deterministic worker-rotation "
                        "positional matching (graph-index order vs trace time order), "
                        "empirically validated per bs (>=98.4% of 128 workers matched "
                        "at the discovered rotation offset) before trusting the split -- "
                        "see opt/m3i10/remeasure/scripts/anchor_qc.py + "
                        "opt/m3i10/remeasure/qc/armA_bs*_rep0_qc.json.")
            new_targets.append(row)
            continue

        # --- disposition rows (unchanged category, refreshed numbers) -----
        old_row = old_disp_by_stage[stage_name]
        hist = build_history(old_row, HIST_DISP_KEYS)
        row = {k: old_row[k] for k in ("mpk_task", "vllm_kernel", "disposition",
                                       "sites_per_step") if k in old_row}
        row.update(dict(
            stage=stage_name,
            vllm_us_per_step={bs: vllm[bs] for bs in BSES},
            mpk_us_per_step={bs: mpk[bs] for bs in BSES},
            ratio_mpk_over_vllm={bs: ratio[bs] for bs in BSES},
            step_gain_if_met_us={bs: gap[bs] for bs in BSES},
            history_m3i1=hist,
        ))
        if stage_name == "shared-expert gate (sigmoid*shared+residual)":
            row["reason"] = (
                "CONFIRMED at matched geometry / current HEAD: the bs8 anomaly does "
                "NOT reproduce. Old wallspan 428/596/432 us (bs1/8/16, a clear spike at "
                "bs8); new wallspan 293/360/363 us -- smooth and monotonic. Code-delta "
                "isolation (arm B, same msl=132, vs the committed M3-I1 capture) shows "
                "the I8 gate alone already mostly resolves it (428->295, 596->373, "
                "432->379); geometry delta (arm B->arm A) finishes the smoothing. Now "
                "mpk-ahead at all three batch sizes (ratio " + str(ratio["1"]) + "x / " +
                str(ratio["8"]) + "x / " + str(ratio["16"]) +
                "x). No ferret dispatch. See opt/m3i10/remeasure/pertask_by_bs.csv "
                "(task 238) for the code-delta/geometry-delta table.")
        elif stage_name == "MoE/shared SiLU-mul":
            # was "below-threshold" (slower-but-cheap) because of the bs16
            # point specifically; that point flips too at matched geometry, so
            # the category itself must move, not just the prose.
            row["disposition"] = "mpk-ahead"
            row["reason"] = (
                "Was slower only at bs16 (1.23x, category 'below-threshold') in the "
                "M3-I1 capture; at matched geometry / current HEAD it is mpk-ahead at "
                f"ALL three batch sizes (ratio {ratio['1']}x / {ratio['8']}x / "
                f"{ratio['16']}x) -- category updated to mpk-ahead accordingly. Still "
                "the one APPROXIMATE mapping in the table (vLLM's "
                "activationDeepSeekKernel also re-quantises to fp8, billed to MPK task "
                "275 instead) -- read the flip as a direction, not a precise "
                "measurement. No ferret dispatch.")
        else:
            row["reason"] = old_row.get("reason", "")
        new_dispositions.append(row)

    # rank: targets only, sorted by NEW bs1 abs gap, descending
    new_targets.sort(key=lambda r: -r["step_gain_if_met_us"]["1"])
    for i, r in enumerate(new_targets):
        r["rank"] = i + 1

    # ---------------- coverage -------------------------------------------
    all_stages = [s for s in cmp_rows if s != "TOTAL (step)"]
    target_names = [r["stage"] for r in new_targets]
    disp_names = [r["stage"] for r in new_dispositions]
    covered = set(target_names) | set(disp_names)
    missing = sorted(set(all_stages) - covered)
    extra = sorted(covered - set(all_stages))
    assert not missing, f"stages with no row: {missing}"
    assert not extra, f"rows with no stage: {extra}"
    assert len(set(target_names)) == len(target_names), "duplicate target stage"
    assert len(set(disp_names)) == len(disp_names), "duplicate disposition stage"

    def slower_at(bs):
        return sorted(s for s in all_stages
                     if f(cmp_rows[s].get(f"ratio_bs{bs}")) and
                     f(cmp_rows[s][f"ratio_bs{bs}"]) > 1.0)
    slower_any = sorted({s for bs in BSES for s in slower_at(bs)})
    slower_all = sorted(s for s in all_stages
                        if all(f(cmp_rows[s].get(f"ratio_bs{bs}")) and
                              f(cmp_rows[s][f"ratio_bs{bs}"]) > 1.0 for bs in BSES))

    out = dict(old)  # start from old to keep ncu_status / sglang_cross_check / etc.
    out["schema_version"] = "2.0"
    out["generated_utc"] = GENERATED_UTC
    out["generator"] = GENERATOR
    out["basis"] = {
        "vllm": old["basis"]["vllm"],
        "mpk": ("matched-geometry re-measure at current HEAD (msl=353 = "
                "256-token synthetic prompt + 96 decode steps + 1; gate_padding_rows "
                "ON, post-M3-I2b), opt/m3i10/remeasure/armA/pertask_by_bs.csv -- "
                "SUPERSEDES the M3-I1 AC-3-geometry capture this file used before. "
                "Anchor-QC (integer per-step task-type counts vs the compiled graph) "
                "exact at every bs: max_frac_err=0.0000."),
        "target_rule": f"beat the corresponding vLLM kernel by {int((1 - TARGET_RATIO) * 100)} % => target = {TARGET_RATIO}x the vLLM number",
        "rank_rule": ("targets ranked by NEW step_gain_if_met_us at bs1, descending "
                      "(comparison_by_stage.csv carries no rank column; this is the "
                      "one place this file imposes an ordering choice, stated here "
                      "so it is reproducible)."),
        "caveat": ("Matched geometry closes the AC-3-vs-256/1024 gap for every stage "
                  "EXCEPT attention's decode CONTEXT: this table's MPK side still "
                  "samples decode context ~257-352 (a 256-token prompt, steady window "
                  "8 steps in) against the vLLM reference's ~556-896. See "
                  "`late_context_addendum` for the closure capture and corrected "
                  "attention row."),
        "history": ("Every row's pre-remeasure (M3-I1, AC-3 geometry, pre-I8/I2b) "
                   "numbers are preserved verbatim in that row's `history_m3i1`."),
    }
    out["targets"] = new_targets
    out["dispositions"] = new_dispositions
    out["coverage"] = {
        "source": "opt/m3i10/remeasure/armA_m3i10/tables/comparison_by_stage.csv",
        "n_stages_total": len(all_stages),
        "n_stages_slower_at_every_batch_size": len(slower_all),
        "n_stages_slower_at_some_batch_size": len(slower_any),
        "n_stages_slower_per_bs": {f"bs{bs}": len(slower_at(bs)) for bs in BSES},
        "n_real_target_specs": len(new_targets),
        "n_disposition_rows": len(new_dispositions),
        "assertion": ("targets + dispositions enumerate every stage in "
                      "comparison_by_stage.csv exactly once; verified by "
                      "regenerate_ferret_v2.py at generation time"),
        "stage_index": {s: ("target" if s in target_names else
                            next(d["disposition"] for d in new_dispositions
                                if d["stage"] == s))
                       for s in sorted(all_stages)},
        "slower_stages_any_bs": slower_any,
        "slower_stages_every_bs": slower_all,
    }
    out.pop("pending_remeasure", None)
    gdn_lc, dense_lc, moe_lc = latectx.get(237, {}), latectx.get(279, {}), latectx.get(241, {})
    out["late_context_addendum"] = {
        "why": ("codex c2 finding F2: this file's matched-geometry MPK numbers still sampled "
                "decode context ~257-352 while the vLLM reference table sampled ~556-896 -- "
                "exactly the stage the attention finding is about. remeasure_spec.md sec 4(d) "
                "closes it with a dedicated msl=897 capture (context ~801-896)."),
        "attention_wallspan_us_matched_vs_late": {
            bs: {"matched_ctx_257_352": cmp_rows["full attention"][f"mpk_us_step_bs{bs}"],
                "late_ctx_801_896": latectx.get(257, {}).get(bs)}
            for bs in BSES},
        "spot_check_context_flat_stages": {
            "GDN recurrent (task 237)": {
                bs: dict(matched=cmp_rows["GDN recurrent (delta rule)"][f"mpk_us_step_bs{bs}"],
                        late=gdn_lc.get(bs),
                        pct_change=(round(100 * (gdn_lc[bs] - float(cmp_rows["GDN recurrent (delta rule)"][f"mpk_us_step_bs{bs}"])) / float(cmp_rows["GDN recurrent (delta rule)"][f"mpk_us_step_bs{bs}"]), 2) if gdn_lc.get(bs) else None))
                for bs in BSES},
            "dense fp8 (task 279)": {
                bs: dict(matched=cmp_rows["dense projections (fp8 blockscale)"][f"mpk_us_step_bs{bs}"],
                        late=dense_lc.get(bs))
                for bs in BSES},
            "verdict": ("GDN recurrent and dense-fp8 stay flat within ~1% moving from context "
                       "257-352 to 801-896 (GDN: +0.2%/+0.02%/-0.01% at bs1/8/16; dense-fp8: "
                       "+0.2%/-0.6%/-0.5%) -- confirms these are genuinely context-insensitive, "
                       "not an artifact of the un-recentred window. MoE w13 is flatter at "
                       "bs1/bs8 (-0.3%/+3.9%) but shows +18.8% at bs16 -- read with the same "
                       "reduced-concurrency caveat as the attention bs16 point (only 12/16 "
                       "live, 13-iteration window), not yet attributed cleanly to context vs. "
                       "concurrency."),
        },
    }
    out["remeasure_executed"] = {
        "spec": "remeasure_spec.md",
        "status": "EXECUTED " + GENERATED_UTC,
        "tiers_run": "tier 1 (arm A, matched geometry) + tier 2 (arm B, continuity) "
                    "+ late-context addendum (see remeasure_spec.md)",
        "artifacts": "opt/m3i10/remeasure/ (rsync target; raw npz pointers to "
                    "/home/catalyst/mpk-artifacts/m3i10-remeasure/)",
    }

    FT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {FT_PATH}")
    print(f"targets: {len(new_targets)}  dispositions: {len(new_dispositions)}")
    print("\nnew rank table (bs1 step_gain_if_met_us, descending):")
    for r in new_targets:
        print(f"  {r['rank']:2d}  {r['stage']:45s} "
              f"ratio(1/8/16)={r['ratio_mpk_over_vllm']['1']:.2f}/"
              f"{r['ratio_mpk_over_vllm']['8']:.2f}/{r['ratio_mpk_over_vllm']['16']:.2f} "
              f"gap(1/8/16)={r['step_gain_if_met_us']['1']:.0f}/"
              f"{r['step_gain_if_met_us']['8']:.0f}/{r['step_gain_if_met_us']['16']:.0f}")
    print("\ndispositions:")
    for r in new_dispositions:
        print(f"  {r['stage']:45s} -> {r['disposition']}")


if __name__ == "__main__":
    main()
