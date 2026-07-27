#!/usr/bin/env python3
"""Join the vLLM per-stage decode tables (this issue) with the committed MPK per-task tables
(M3-I1 / M3-I8) into the M3-I10 correspondence + gap table.

MPK side: `wallspan_us_bs*` from demo/qwen3_5/accept/opt/pertask_by_bs.csv - the union of the
time a task family is executing inside the persistent kernel.  M3-I1 established that summing
task-family wall spans accounts for 109-114 % of the step, so a family's wall span is the fair
estimate of the step time it costs.

vLLM side: sum of that stage's CUDA kernel durations per decode step.  For every stage except
the shared expert, stage-union == stage-sum in the traces, so the two conventions agree; the
shared-expert stage runs on an overlapped side stream and is flagged.
"""
import csv
import json
from pathlib import Path

REPO = Path("/home/catalyst/project/demo/qwen3_5/accept/opt")
VOUT = Path(__file__).resolve().parent.parent / "tables"
BSES = ["1", "8", "16"]

# MPK step time (us) from M3-I1 attribution.csv / README
MPK_STEP = {"1": 15264.0, "8": 18618.0, "16": 22005.0}

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
     "NOT LIKE-FOR-LIKE: vLLM measured at KV length 556-896 (256+1024 workload), MPK at the "
     "AC-3 geometry (max_seq_length 132). The MPK/vLLM ratio here is a LOWER bound"),
    ("norms / RoPE / glue", ["norms_rope_glue"], [154], 81,
     "NOT LIKE-FOR-LIKE: MPK fuses most norms/RoPE/L2-norm into its GEMM, attention and "
     "recurrent tasks; task 154 is only the standalone RMSNorm"),
    ("sampling / argmax", ["sampling_argmax"], [259, 258], 1, ""),
    ("embedding", ["embedding"], [101], 1, ""),
]


def load_vllm():
    out = {}
    for bs in BSES:
        rows = list(csv.DictReader(open(VOUT / f"bs{bs}_stages.csv")))
        out[bs] = {r["stage"]: r for r in rows}
    return out


def load_mpk():
    rows = list(csv.DictReader(open(REPO / "pertask_by_bs.csv")))
    return {int(r["task_type"]): r for r in rows}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def main():
    V, M = load_vllm(), load_mpk()
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
        j = json.loads((VOUT / f"bs{bs}_stages.json").read_text())
        tot[f"vllm_us_step_bs{bs}"] = round(vsum, 1)
        tot[f"vllm_union_us_step_bs{bs}"] = round(j["step_union_us_median"], 1)
        tot[f"mpk_us_step_bs{bs}"] = MPK_STEP[bs]
        tot[f"ratio_bs{bs}"] = round(MPK_STEP[bs] / j["step_union_us_median"], 3)
        tot[f"abs_gap_us_step_bs{bs}"] = round(MPK_STEP[bs] - j["step_union_us_median"], 1)

    Path(VOUT / "comparison.json").write_text(json.dumps({"stages": recs, "total": tot}, indent=2))
    cols = (["stage", "n_layer_sites"]
            + [c for bs in BSES for c in
               (f"vllm_us_step_bs{bs}", f"mpk_us_step_bs{bs}", f"ratio_bs{bs}",
                f"abs_gap_us_step_bs{bs}", f"vllm_us_layer_bs{bs}", f"mpk_us_layer_bs{bs}")]
            + ["note"])
    with open(VOUT / "comparison_by_stage.csv", "w", newline="") as fh:
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
    print(f"\nwrote {VOUT / 'comparison_by_stage.csv'}")


if __name__ == "__main__":
    main()
