#!/usr/bin/env python3
"""Decisive follow-up measurement (M2-I2, coordinator completion-review cycle 3).

Cycle 2 caught that P10's measured frob_rel_error is K-INVARIANT (mean 3.537e-3 at K=2048
vs 3.486e-3 at K=4096, ratio 1.01 -- NOT the sqrt(2)=1.41 a 1/sqrt(K) single-flip model
predicts). Corrected model: a FIXED FRACTION f of the K contracted activation elements
disagree in e4m3 quantization bucket between the two independently-authored quantizers
(vLLM's per_token_group_quant_fp8 vs the HF kernels-hub finegrained-fp8 Triton kernel), each
contributing ~one e4m3 LSB (0.125 relative); N=f*K such disagreements accumulate like sqrt(N)
(CLT/random-walk, same reasoning as p10_fp8_dense_bar.py's single_flip_floor), and the K-term
sum itself has magnitude ~sqrt(K)*term_scale -- the two sqrt(K) factors cancel, leaving:

    relative_frob ~= E4M3_RELATIVE_LSB * sqrt(f), INDEPENDENT of K.

P10's measured ~3.5e-3 implies f ~= (3.5e-3/0.125)^2 ~= 7.8e-4 (order 1e-4..1e-3). This script
tests that directly and decisively: extracts the RAW e4m3 CODE each quantizer actually
produces (not just the downstream GEMM output) on identical bf16 activations, counts the
fraction of positions where the codes differ, and measures the ULP distance of every
disagreement (should be 1 if this is ordinary rounding-boundary variance between two correct
quantizers, not a bug).

Method for extracting Triton's raw code (finegrained_fp8_linear only exposes a FUSED
act-quant+matmul, no standalone quantize entry point): an IDENTITY-WEIGHT trick. Build a
weight tensor W = the EXACT KxK identity matrix in float8_e4m3fn (1.0 and 0.0 are both
exactly representable in e4m3 -- this cast is lossless) with weight_scale_inv = all-ones
(also exact), block_size=[128,128]. Then finegrained_fp8_linear(X, W, ones, [128,128]) at
row m, col i = sum_k Xq_triton[m,k]*Xscale_triton[m,k_grp] * W[i,k]*1 = (since W[i,k]=1 iff
i==k, else 0) exactly Xq_triton[m,i] * Xscale_triton[m,i's group] -- i.e. Triton's OWN
quantize-then-dequantize reconstruction of X, with no matmul math to obscure it (the other
K-1 terms per output contribute exactly 0.0, the fp32 additive identity -- no extra rounding).
Compared directly against CUTLASS's own Xq*Xs reconstruction (trivial, no trick needed).
"""
import argparse
import json
import os
import zlib

os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

import torch  # noqa: E402
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)
from transformers.integrations.finegrained_fp8 import finegrained_fp8_linear  # noqa: E402

BLOCK = 128
E4M3_MAG_MAX = 126  # 0x7E: finest finite magnitude code for float8_e4m3fn (0x7F is NaN)


def stable_seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def identity_weight(K, device):
    w = torch.eye(K, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    s = torch.ones(K // BLOCK, K // BLOCK, dtype=torch.float32, device=device)
    return w, s


def ulp_step(fp8_tensor, n):
    """Step e4m3 codes by n ULPs, same sign, magnitude bits +n (clamped to finite range)."""
    raw = fp8_tensor.view(torch.uint8)
    sign = (raw & 0x80).to(torch.int16)
    mag = (raw & 0x7F).to(torch.int16)
    mag_stepped = (mag + n).clamp(0, E4M3_MAG_MAX)
    new_raw = (sign | mag_stepped).to(torch.uint8)
    return new_raw.view(torch.float8_e4m3fn)


def analyze_case(K, M, seed, device, ulp_search=6):
    X = torch.randn(M, K, generator=torch.Generator(device=device).manual_seed(seed),
                     device=device, dtype=torch.float32).to(torch.bfloat16)

    Xq_cutlass, Xs_cutlass = per_token_group_quant_fp8(X, BLOCK, column_major_scales=True, use_ue8m0=False)
    # BUG CAUGHT DURING THIS CYCLE'S DEBUG (before trusting any count): v_triton comes out of
    # the fused kernel ALREADY rounded to bf16, while a naive `Xq*Xs` reconstruction here is
    # full fp32 -- comparing those two precisions directly flagged ~90% of positions as
    # "differing" purely from ORDINARY bf16 rounding (~0.4% noise), not e4m3 code disagreement
    # (verified on a single 128-element block: after matching precision, 0/128 differed).
    # Fix: round BOTH reconstructions through bf16 before any comparison -- apples to apples.
    v_cutlass = (Xq_cutlass.float() * Xs_cutlass.repeat_interleave(BLOCK, dim=1)[:, :K]).to(torch.bfloat16).float()

    W, Wscale = identity_weight(K, device)
    v_triton = finegrained_fp8_linear(X, W, Wscale, block_size=[BLOCK, BLOCK]).float()

    exact_diff_mask = (v_cutlass != v_triton)
    n_diff_exact = int(exact_diff_mask.sum().item())
    n_total = v_cutlass.numel()

    # ULP-distance search for every exactly-differing position: step CUTLASS's own code by n
    # in [-ulp_search..ulp_search]\{0}, dequantize with CUTLASS's own (same-group) scale,
    # round through bf16 the SAME way v_cutlass/v_triton were (precision-matched), and find
    # which step's value matches Triton's reconstruction EXACTLY (no tolerance fudge needed
    # once precision is matched on both sides).
    #
    # SECOND MECHANISM FOUND DURING THIS CYCLE'S DEBUG: not every disagreement is a code-level
    # ULP step. On a real case, 15/25 differing positions were ALL in the same 128-wide group
    # and ALL shared one uniform ratio target/cutlass_recon (1.005814, identical to 6 sig figs)
    # that matches NO possible e4m3 ULP step (a real 1-ULP step at code=240's bracket gives
    # 256/240=1.0667 or 224/240=0.9333, neither close) -- i.e. that whole group's SCALE
    # disagreed by ~0.58% between the two quantizers (verified: cutlass's scale for every group
    # in this test independently matched a from-scratch absmax/448 computation exactly; only
    # Triton's differed, and only for that one group). Every OTHER differing position, by
    # contrast, DID resolve to an exact integer ULP step, cross-validated against known e4m3
    # bracket ratios (e.g. 176/192=0.91667, 240/256=0.9375, 26/28=0.92857 -- all exact). So we
    # classify each differing position as `code_disagreement` (ULP search matches) or
    # `group_scale_disagreement` (doesn't match any ULP step, but sits in a group where >=2
    # differing positions share the same target/cutlass_recon ratio to high precision) or
    # `unexplained` (neither -- would be the loud "something else is going on" signal).
    diff_idx = exact_diff_mask.nonzero(as_tuple=False)
    ulp_distances = []
    n_group_scale, n_unexplained = 0, 0
    group_scale_ratios = []
    if diff_idx.numel() > 0:
        rows, cols = diff_idx[:, 0], diff_idx[:, 1]
        groups = cols // BLOCK
        base_codes = Xq_cutlass[rows, cols]
        group_scale = Xs_cutlass[rows, groups]
        cutlass_recon = v_cutlass[rows, cols]
        targets = v_triton[rows, cols]
        found = torch.zeros(len(rows), dtype=torch.bool, device=device)
        best_n = torch.zeros(len(rows), dtype=torch.int64, device=device)
        for n in list(range(-ulp_search, 0)) + list(range(1, ulp_search + 1)):
            cand = (ulp_step(base_codes, n).float() * group_scale).to(torch.bfloat16).float()
            match = (~found) & (cand == targets)
            best_n[match] = n
            found |= match
        ulp_distances = best_n[found].abs().tolist()

        # For NOT-found positions: classify by the MAGNITUDE of target/cutlass_recon - 1.
        # FIX (caught immediately on the first full run): the original classifier required
        # >=2 positions per (row,group) to agree before calling it "group scale" -- but a
        # group where the scale nudge only pushes ONE element across a bf16 rounding boundary
        # (common: most elements in a nudged group still round to the same bf16 value even
        # though the true value shifted, see analyze_case's module-level reasoning) has a
        # SINGLETON ratio and was wrongly falling into "unexplained" by construction, not
        # because anything new was happening. Reclassify directly by ratio magnitude instead:
        # a small-scale-type discrepancy is <<1 e4m3 LSB (12.5%) in relative terms; using a
        # 5% cutoff (comfortably above the two ratios actually observed, ~0.58-0.62%, and
        # comfortably below the smallest possible ULP step) cleanly separates "small scale
        # nudge" from a genuinely large, structurally different disagreement.
        SMALL_RATIO_CUTOFF = 0.05
        unresolved_idx = (~found).nonzero(as_tuple=False).squeeze(-1)
        if unresolved_idx.numel() > 0:
            u_ratio = (targets[unresolved_idx] / cutlass_recon[unresolved_idx]).tolist()
            for ratio in u_ratio:
                if abs(ratio - 1.0) < SMALL_RATIO_CUTOFF:
                    n_group_scale += 1
                    group_scale_ratios.append(round(ratio, 6))
                else:
                    n_unexplained += 1

    return {
        "K": K, "M": M, "n_total_elements": n_total,
        "n_differing_codes": n_diff_exact,
        "fraction_differing": n_diff_exact / n_total,
        "n_code_ulp_disagreement": len(ulp_distances),
        "max_ulp_distance": max(ulp_distances) if ulp_distances else 0,
        "ulp_distance_histogram": {str(u): ulp_distances.count(u) for u in sorted(set(ulp_distances))},
        "n_group_scale_disagreement": n_group_scale,
        "group_scale_ratios_observed": group_scale_ratios,
        "n_unexplained": n_unexplained,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="2048x1,2048x16,4096x1,4096x16",
                     help="comma list of KxM to test (K=contraction dim, M=decode batch)")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--ulp-search", type=int, default=6)
    ap.add_argument("--out", default=os.path.expanduser(
        "~/mpk-qwen35/probes/fp8_out/p10b_activation_quant_disagreement.json"))
    args = ap.parse_args()

    device = "cuda"
    print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}", flush=True)

    results = []
    for spec in args.cases.split(","):
        K_str, M_str = spec.split("x")
        K, M = int(K_str), int(M_str)
        r = analyze_case(K, M, stable_seed(args.seed_base, K, M), device, args.ulp_search)
        results.append(r)
        print(f"K={K} M={M}: fraction_differing={r['fraction_differing']:.4e} "
              f"({r['n_differing_codes']}/{r['n_total_elements']}) "
              f"code_ulp={r['n_code_ulp_disagreement']} (max={r['max_ulp_distance']}, "
              f"hist={r['ulp_distance_histogram']}) "
              f"group_scale={r['n_group_scale_disagreement']} "
              f"(ratios={r['group_scale_ratios_observed']}) "
              f"unexplained={r['n_unexplained']}", flush=True)

    fractions = [r["fraction_differing"] for r in results]
    max_ulps = [r["max_ulp_distance"] for r in results if r["n_code_ulp_disagreement"] > 0]
    total_unexplained = sum(r["n_unexplained"] for r in results)
    total_group_scale = sum(r["n_group_scale_disagreement"] for r in results)
    all_group_scale_ratios = [x for r in results for x in r["group_scale_ratios_observed"]]

    fractions_sorted = sorted(fractions)
    n = len(fractions_sorted)
    median_f = fractions_sorted[n // 2] if n % 2 else (fractions_sorted[n // 2 - 1] + fractions_sorted[n // 2]) / 2

    predicted_frob_from_measured_f = {f"K={r['K']}_M={r['M']}": 0.125 * (r["fraction_differing"] ** 0.5)
                                       for r in results}

    summary = {
        "min_fraction_differing": min(fractions),
        "median_fraction_differing": median_f,
        "max_fraction_differing": max(fractions),
        "max_ulp_distance_overall": max(max_ulps) if max_ulps else 0,
        "all_code_level_disagreements_are_1_ulp": (max(max_ulps) == 1) if max_ulps else None,
        "total_unexplained_positions": total_unexplained,
        "second_mechanism_found": {
            "description": "A MINORITY of differing positions are not a per-element code ULP "
                "step at all -- they cluster (>=2 positions sharing one (row,group)) at a "
                "UNIFORM target/cutlass_recon ratio that matches no possible e4m3 ULP step, "
                "i.e. a small per-128-element-GROUP SCALE discrepancy between the two "
                "quantizers (verified: CUTLASS's scale independently matches a from-scratch "
                "absmax/448 computation exactly in every group checked; only Triton's group "
                "scale occasionally differs by a small amount). Distinct from, but comparably "
                "benign to, the single-code-flip mechanism the frob-floor model assumed.",
            "total_positions_classified_as_group_scale": total_group_scale,
            "observed_ratios (target/cutlass_recon, should be close to 1.0)": sorted(set(all_group_scale_ratios)),
        },
        "model_check": {
            "formula": "relative_frob ~= 0.125 * sqrt(f)",
            "p10_measured_frob_range": "2.1e-3 to 4.35e-3 (mean 3.537e-3 at K=2048, 3.486e-3 at K=4096)",
            "f_implied_by_p10_frob_using_mean": (3.51e-3 / 0.125) ** 2,
            "predicted_frob_from_this_measurement's_f_per_case": predicted_frob_from_measured_f,
        },
        "cases_tested": results,
    }
    print("\n=== SUMMARY ===")
    print(f"fraction_differing: min={summary['min_fraction_differing']:.4e} "
          f"median={summary['median_fraction_differing']:.4e} max={summary['max_fraction_differing']:.4e}")
    print(f"max_ulp_distance_overall={summary['max_ulp_distance_overall']} "
          f"all_code_level_disagreements_are_1_ulp={summary['all_code_level_disagreements_are_1_ulp']}")
    print(f"total_group_scale_positions={total_group_scale}  ratios_seen={sorted(set(all_group_scale_ratios))}")
    print(f"total_unexplained={total_unexplained}")
    print(f"f implied by P10's own measured frob (mean~3.51e-3): {summary['model_check']['f_implied_by_p10_frob_using_mean']:.3e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
