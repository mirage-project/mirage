#!/usr/bin/env python3
"""Roll the per-kernel decode tables up into LAYER STAGES that are comparable to MPK task
families, computing both the sum of kernel durations and the union of their intervals.

The stage map is explicit and auditable: (substring, stream-class) -> stage.  stream-class is
"main" (the primary compute stream) or "side" (the shared-expert branch vLLM overlaps).
"""
import argparse
import gc
import json
import statistics
from collections import defaultdict
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

# (substring, stream_class or None=any) -> (stage, n_sites_per_step)
# n_sites: how many model sites the stage covers per decode step (used for us/layer).
RULES = [
    ("cutlass_3x_gemm_fp8_blockwise", "side", "dense_fp8_shared_expert", 80),
    ("cutlass_3x_gemm_fp8_blockwise", "main", "dense_fp8_attn_gdn_proj", 80),
    ("per_token_group_quant_8bit_kernel", "side", "quantize_fp8_shared", 80),
    ("per_token_group_quant_8bit_kernel", "main", "quantize_fp8_main", 120),
    ("bmm_E4m3_E4m3E4m3", None, "moe_grouped_gemm_w13", 40),
    ("bmm_Bfloat16_E4m3E4m3", None, "moe_grouped_gemm_w2", 40),
    ("activationDeepSeekKernel", None, "moe_act_silu_requant", 40),
    ("routingIndices", None, "moe_router_topk", 40),
    ("finalizeKernel", None, "moe_finalize_combine", 40),
    ("nvjet_sm100_tst_32x64_64x16_4x1_v_bz_splitK_TNN", None, "moe_router_gemm_bf16", 40),
    ("nvjet_sm100_tst_32x64_64x16_2x1_v_bz_splitK_TNN", None, "gdn_in_proj_ba_bf16", 30),
    ("splitKreduce_kernel", "side", "shared_expert_gate", 40),
    ("splitKreduce_kernel", "main", "bf16_small_gemm_splitk_reduce", 70),
    ("gemv2N_kernel", None, "shared_expert_gate", 40),
    ("nvjet_sm100_tst_64x8_64x16_1x1_h_bz_splitK_TNT", None, "shared_expert_gate", 40),
    ("nvjet_sm100_tst_8x64_64x16_1x1_h_bz_splitK_TNN", None, "shared_expert_gate", 40),
    ("sigmoid_kernel_cuda", None, "shared_expert_gate", 40),
    ("BinaryFunctor", "side", "shared_expert_gate", 40),
    ("triton_poi_fused_mul_silu_slice_0", None, "shared_expert_silu_mul", 40),
    ("_causal_conv1d_update_kernel", None, "gdn_conv1d", 30),
    ("fused_recurrent_gated_delta_rule", None, "gdn_recurrent", 30),
    ("fmhaSm100fKernel", None, "full_attention", 10),
    ("reshape_and_cache_flash", None, "attention_kv_write", 10),
    ("_compute_slot_mapping_kernel", None, "attention_kv_write", 4),
    ("nvjet_sm100_tst_192x", None, "lm_head_bf16", 1),
    ("ArgMaxOps", None, "sampling_argmax", 1),
    ("indexSelectSmallIndex", None, "embedding", 1),
    # everything triton_* / elementwise that is left = fused norms / RoPE / L2norm / gating glue
    ("triton_", None, "norms_rope_glue", None),
    ("elementwise_kernel", None, "norms_rope_glue", None),
    ("reduce_kernel", None, "norms_rope_glue", None),
    ("index_elementwise_kernel", None, "norms_rope_glue", None),
    ("Memcpy", None, "memcpy_memset", None),
    ("Memset", None, "memcpy_memset", None),
]


def classify(name, stream, main_stream):
    for sub, sc, stage, sites in RULES:
        if sub not in name:
            continue
        if sc == "main" and stream != main_stream:
            continue
        if sc == "side" and stream == main_stream:
            continue
        return stage, sites
    return "unmapped", None


def union_us(ivs):
    ivs.sort()
    tot, cs, ce = 0.0, None, None
    for a, b in ivs:
        if cs is None:
            cs, ce = a, b
        elif a <= ce:
            ce = max(ce, b)
        else:
            tot += ce - cs
            cs, ce = a, b
    if cs is not None:
        tot += ce - cs
    return tot


def parse(path, anchor):
    with open(path) as f:
        data = json.load(f)
    gpu = []
    for e in data.get("traceEvents", []):
        if e.get("cat") not in GPU_CATS:
            continue
        a = e.get("args") or {}
        gpu.append((e.get("ts", 0) or 0, e.get("dur", 0) or 0, e.get("name", ""), str(a.get("stream"))))
    del data
    gc.collect()
    gpu.sort()
    ats = [ts for ts, _, n, _ in gpu if n == anchor]
    t0, t1, nst = ats[0], ats[-1], len(ats) - 1
    # main stream = the stream carrying the most GPU time
    st_tot = defaultdict(float)
    for ts, d, n, s in gpu:
        if t0 <= ts < t1:
            st_tot[s] += d
    main = max(st_tot, key=st_tot.get)

    stage_iv = defaultdict(list)
    stage_calls = defaultdict(int)
    stage_sites = {}
    unmapped = defaultdict(float)
    for ts, d, n, s in gpu:
        if not (t0 <= ts < t1):
            continue
        stage, sites = classify(n, s, main)
        if stage == "unmapped":
            unmapped[n] += d
        stage_iv[stage].append((ts, ts + d))
        stage_calls[stage] += 1
        if sites:
            stage_sites[stage] = sites
    out = {}
    for stage, ivs in stage_iv.items():
        out[stage] = {
            "calls_per_step": stage_calls[stage] / nst,
            "sum_us_per_step": sum(b - a for a, b in ivs) / nst,
            "union_us_per_step": union_us(ivs) / nst,
            "sites_per_step": stage_sites.get(stage),
        }
    all_iv = [iv for ivs in stage_iv.values() for iv in ivs]
    meta = {
        "trace": Path(path).name, "n_steps": nst, "main_stream": main,
        "streams_us_per_step": {k: v / nst for k, v in st_tot.items()},
        "total_sum_us_per_step": sum(b - a for a, b in all_iv) / nst,
        "total_union_us_per_step": union_us(all_iv) / nst,
        "step_wall_us": (t1 - t0) / nst,
        "unmapped_us_per_step": {k: v / nst for k, v in sorted(unmapped.items(), key=lambda x: -x[1])[:12]},
    }
    del gpu
    gc.collect()
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    pers, metas = [], []
    for t in args.traces:
        o, m = parse(t, args.anchor)
        pers.append(o)
        metas.append(m)
        print(f"[stage] {m['trace']}: steps={m['n_steps']} main_stream={m['main_stream']} "
              f"sum={m['total_sum_us_per_step']:.1f} union={m['total_union_us_per_step']:.1f} "
              f"wall={m['step_wall_us']:.1f}", flush=True)

    stages = sorted({s for p in pers for s in p})
    rows = []
    for s in stages:
        sums = [p.get(s, {}).get("sum_us_per_step", 0.0) for p in pers]
        unis = [p.get(s, {}).get("union_us_per_step", 0.0) for p in pers]
        calls = [p.get(s, {}).get("calls_per_step", 0.0) for p in pers]
        sites = next((p[s]["sites_per_step"] for p in pers if s in p and p[s]["sites_per_step"]), None)
        med = statistics.median(sums)
        rows.append({
            "stage": s, "calls_per_step": statistics.median(calls),
            "sites_per_step": sites,
            "sum_us_per_step_median": med,
            "sum_us_per_step_min": min(sums), "sum_us_per_step_max": max(sums),
            "range_pct": (max(sums) - min(sums)) / med * 100 if med else 0.0,
            "union_us_per_step_median": statistics.median(unis),
            "us_per_site": med / sites if sites else None,
        })
    rows.sort(key=lambda r: -r["sum_us_per_step_median"])
    res = {"label": args.label, "n_windows": len(pers), "per_trace_meta": metas, "stages": rows,
           "step_sum_us_median": statistics.median([m["total_sum_us_per_step"] for m in metas]),
           "step_union_us_median": statistics.median([m["total_union_us_per_step"] for m in metas])}
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"{args.label}_stages.json").write_text(json.dumps(res, indent=2))
    with open(od / f"{args.label}_stages.csv", "w") as f:
        f.write("stage,calls_per_step,sites_per_step,sum_us_per_step,sum_min,sum_max,range_pct,"
                "union_us_per_step,us_per_site\n")
        for r in rows:
            f.write(f'{r["stage"]},{r["calls_per_step"]:.2f},{r["sites_per_step"] or ""},'
                    f'{r["sum_us_per_step_median"]:.2f},{r["sum_us_per_step_min"]:.2f},'
                    f'{r["sum_us_per_step_max"]:.2f},{r["range_pct"]:.2f},'
                    f'{r["union_us_per_step_median"]:.2f},'
                    f'{r["us_per_site"]:.3f}\n' if r["us_per_site"] else
                    f'{r["stage"]},{r["calls_per_step"]:.2f},,{r["sum_us_per_step_median"]:.2f},'
                    f'{r["sum_us_per_step_min"]:.2f},{r["sum_us_per_step_max"]:.2f},'
                    f'{r["range_pct"]:.2f},{r["union_us_per_step_median"]:.2f},\n')
    print(f"\n=== {args.label}: step sum={res['step_sum_us_median']:.1f} "
          f"union={res['step_union_us_median']:.1f} us ===")
    print(f"{'sum us/step':>12} {'union':>9} {'calls':>7} {'sites':>6} {'us/site':>8}  stage")
    for r in rows:
        ups = f"{r['us_per_site']:.3f}" if r['us_per_site'] else "-"
        print(f"{r['sum_us_per_step_median']:12.2f} {r['union_us_per_step_median']:9.2f} "
              f"{r['calls_per_step']:7.1f} {str(r['sites_per_step'] or '-'):>6} {ups:>8}  {r['stage']}")
    print("\nunmapped (first window):", json.dumps(metas[0]["unmapped_us_per_step"], indent=1)[:900])


if __name__ == "__main__":
    main()
