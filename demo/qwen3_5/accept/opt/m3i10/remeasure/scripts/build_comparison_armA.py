#!/usr/bin/env python3
"""M3-I10 remeasure fork of opt/m3i10/scripts/build_comparison.py.

SAME join logic (MAP table, columns, roofline-adjacent prose) as the
committed script. Three deltas, all just WHERE things are read/written from,
none of the arithmetic:

  1. REPO points at this re-measure's own armA/pertask_by_bs.csv (matched
     geometry, current HEAD) instead of the live opt/pertask_by_bs.csv
     (M3-I1's msl=132 capture) -- that live file is exactly what this issue
     found stale and is NOT touched by this run.
  2. vLLM-side bs<N>_stages.csv/json are read from the EXISTING committed
     m3i10/tables/ (VOUT_READ) -- unchanged, this issue never re-measures
     vLLM; only the MPK column was stale.
  3. Output goes to VOUT_WRITE (this re-measure's own tables/ dir), never
     overwriting the committed m3i10/tables/comparison_by_stage.csv, so the
     coordinator can diff old vs new before landing anything.

MPK_STEP is no longer a hardcoded snapshot (the spec explicitly flags the
old hardcoded M3-I1 constants as the one thing that must be updated) -- it is
read directly out of the regenerated armA/attribution.csv's step_us column,
so it can never silently drift from the pertask table it is the denominator
for.
"""
import csv
import json
from pathlib import Path

import os

ARM = "armA"
REMEASURE = Path(os.environ.get("M3I10RM_DIR",
                                str(Path.home() / "mpk-qwen35" / "m3i10-remeasure")))
REPO = REMEASURE / ARM                                   # pertask_by_bs.csv, attribution.csv
# committed vLLM-side bs*_stages.{csv,json} -- read-only, the isolated clone's own copy
# (unchanged by this remeasure; only the MPK column was stale).
VOUT_READ = Path(os.environ.get(
    "M3I10RM_VLLM_TABLES",
    str(Path.home() / "mpk-qwen35" / "mirage-rm" / "demo" / "qwen3_5" / "accept"
        / "opt" / "m3i10" / "tables")))
VOUT_WRITE = REMEASURE / f"{ARM}_m3i10" / "tables"
VOUT_WRITE.mkdir(parents=True, exist_ok=True)
BSES = ["1", "8", "16"]

# stage -> (vLLM stage keys to sum, MPK task_type(s) to sum, n_layer_sites, note)
MAP = [
    ("dense projections (fp8 blockscale)",
     ["dense_fp8_attn_gdn_proj", "dense_fp8_shared_expert"], [279], 40,
     "160 GEMM sites both sides: qkvz x30, gdn out_proj x30, qkvg x10, o_proj x10, "
     "shared gate_up x40, shared down x40"),
    ("dense projections (bf16 small + lm_head)",
     ["gdn_in_proj_ba_bf16", "moe_router_gemm_bf16", "bf16_small_gemm_splitk_reduce",
      "lm_head_bf16"], [253], 71,
     "71 sites both sides: in_proj_ba x30, MoE router gate x40, lm_head x1"),
    ("MoE routed GEMM w13 (gate_up)", ["moe_grouped_gemm_w13"], [241], 40, ""),
    ("MoE routed GEMM w2 (down)", ["moe_grouped_gemm_w2"], [242], 40, ""),
    ("MoE router top-k/softmax", ["moe_router_topk"], [260], 40,
     "vLLM routing kernel only; the router GEMM is counted in the bf16 dense row"),
    ("MoE combine (weighted sum + residual)", ["moe_finalize_combine"], [261], 40, ""),
    ("MoE/shared SiLU-mul", ["moe_act_silu_requant", "shared_expert_silu_mul"], [118], 80,
     "APPROX: vLLM activationDeepSeekKernel also re-quantises the intermediate to fp8, which "
     "on the MPK side lives in task 275 - this row flatters vLLM's 118-equivalent"),
    ("shared-expert gate (sigmoid*shared+residual)", ["shared_expert_gate"], [238], 40,
     "vLLM = gate GEMM + sigmoid + mul (+splitK reduce at bs8/16), 3-4 kernels; MPK fuses all "
     "of it into one task. vLLM side runs on the overlapped side stream"),
    ("quantize / fp8 casts", ["quantize_fp8_main", "quantize_fp8_shared"], [275], 200, ""),
    ("GDN conv1d", ["gdn_conv1d"], [234], 30, ""),
    ("GDN recurrent (delta rule)", ["gdn_recurrent"], [237], 30, ""),
    ("full attention", ["full_attention", "attention_kv_write"], [257], 10,
     "Arm A is matched-geometry (256-token prompt, msl=353 = 256+96 decode steps+1): the "
     "M3-I1 caveat about AC-3 context (132) no longer applies to this column. This row's "
     "PRIMARY basis in ferret_targets.json is actually the late-context closure capture "
     "(msl=897, ctx~801-896, opt/m3i10/remeasure/armAlate/), not this matched-geometry "
     "(ctx 257-352) number directly -- see that file's context_band/matched_window fields."),
    ("norms / RoPE / glue", ["norms_rope_glue"], [154], 81,
     "NOT LIKE-FOR-LIKE: MPK fuses most norms/RoPE/L2-norm into its GEMM, attention and "
     "recurrent tasks; task 154 is only the standalone RMSNorm"),
    ("sampling / argmax", ["sampling_argmax"], [259, 258], 1, ""),
    ("embedding", ["embedding"], [101], 1, ""),
]


def load_vllm():
    out = {}
    for bs in BSES:
        rows = list(csv.DictReader(open(VOUT_READ / f"bs{bs}_stages.csv")))
        out[bs] = {r["stage"]: r for r in rows}
    return out


def load_mpk():
    rows = list(csv.DictReader(open(REPO / "pertask_by_bs.csv")))
    return {int(r["task_type"]): r for r in rows}


def load_mpk_step():
    """MPK_STEP[bs] = regenerated attribution.csv's step_us -- read, not
    hardcoded, so it cannot go stale the way the M3-I1 snapshot did."""
    rows = list(csv.DictReader(open(REPO / "attribution.csv")))
    return {str(r["batch_size"]): float(r["step_us"]) for r in rows}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    V, M = load_vllm(), load_mpk()
    MPK_STEP = load_mpk_step()
    recs = []
    for name, vkeys, mtasks, nlayers, note in MAP:
        rec = {"stage": name, "vllm_kernels": vkeys, "mpk_tasks": mtasks,
               "n_layer_sites": nlayers, "note": note}
        for bs in BSES:
            v = sum(f(V[bs][k]["sum_us_per_step"]) for k in vkeys if k in V[bs])
            vu = sum(f(V[bs][k]["union_us_per_step"]) for k in vkeys if k in V[bs])
            m = sum(f(M[t][f"wallspan_us_bs{bs}"]) for t in mtasks if t in M)
            mw = sum(f(M[t][f"us_bs{bs}"]) for t in mtasks if t in M)
            rec[f"vllm_us_step_bs{bs}"] = round(v, 2)
            rec[f"vllm_union_us_step_bs{bs}"] = round(vu, 2)
            rec[f"mpk_us_step_bs{bs}"] = round(m, 1)
            rec[f"mpk_workertime_us_step_bs{bs}"] = round(mw, 1)
            rec[f"vllm_us_layer_bs{bs}"] = round(v / nlayers, 3)
            rec[f"mpk_us_layer_bs{bs}"] = round(m / nlayers, 3)
            rec[f"ratio_bs{bs}"] = round(m / v, 3) if v else None
            rec[f"abs_gap_us_step_bs{bs}"] = round(m - v, 1)
        recs.append(rec)

    tot = {"stage": "TOTAL (step)", "n_layer_sites": None, "note": "vLLM: union over all kernels"}
    for bs in BSES:
        vsum = sum(r[f"vllm_us_step_bs{bs}"] for r in recs)
        j = json.loads((VOUT_READ / f"bs{bs}_stages.json").read_text())
        tot[f"vllm_us_step_bs{bs}"] = round(vsum, 1)
        tot[f"vllm_union_us_step_bs{bs}"] = round(j["step_union_us_median"], 1)
        tot[f"mpk_us_step_bs{bs}"] = MPK_STEP[bs]
        tot[f"ratio_bs{bs}"] = round(MPK_STEP[bs] / j["step_union_us_median"], 3)
        tot[f"abs_gap_us_step_bs{bs}"] = round(MPK_STEP[bs] - j["step_union_us_median"], 1)

    (VOUT_WRITE / "comparison.json").write_text(json.dumps({"stages": recs, "total": tot}, indent=2))
    cols = (["stage", "n_layer_sites"]
            + [c for bs in BSES for c in
               (f"vllm_us_step_bs{bs}", f"mpk_us_step_bs{bs}", f"ratio_bs{bs}",
                f"abs_gap_us_step_bs{bs}", f"vllm_us_layer_bs{bs}", f"mpk_us_layer_bs{bs}")]
            + ["note"])
    with open(VOUT_WRITE / "comparison_by_stage.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs + [tot]:
            w.writerow(r)

    for bs in BSES:
        print(f"\n{'=' * 118}\n=== bs{bs}  (MPK step {MPK_STEP[bs]:.0f} us, "
              f"vLLM step {tot[f'vllm_union_us_step_bs{bs}']:.0f} us, "
              f"overall {tot[f'ratio_bs{bs}']:.2f}x)")
        print(f"{'vLLM us/step':>13}{'MPK us/step':>13}{'ratio':>8}{'gap us':>9}"
              f"{'vLLM us/lyr':>12}{'MPK us/lyr':>11}  stage")
        for r in sorted(recs, key=lambda x: -x[f"abs_gap_us_step_bs{bs}"]):
            flag = " <== MPK SLOWER" if (r[f"ratio_bs{bs}"] or 0) > 1.0 else ""
            print(f"{r[f'vllm_us_step_bs{bs}']:13.1f}{r[f'mpk_us_step_bs{bs}']:13.1f}"
                  f"{r[f'ratio_bs{bs}'] or 0:8.2f}{r[f'abs_gap_us_step_bs{bs}']:9.0f}"
                  f"{r[f'vllm_us_layer_bs{bs}']:12.2f}{r[f'mpk_us_layer_bs{bs}']:11.2f}  "
                  f"{r['stage']}{flag}")
    print(f"\nwrote {VOUT_WRITE / 'comparison_by_stage.csv'}")


if __name__ == "__main__":
    main()
