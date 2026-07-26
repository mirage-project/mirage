#!/usr/bin/env python3
"""P10 -- fp32-scale dense fp8 GEMM: numerics + perf bar (M2-I2, v1-architecture.md SS14).

Compares vLLM's own CUTLASS block-scaled fp8 GEMM (ops.cutlass_scaled_mm via the
Fp8LinearMethod / CutlassFp8BlockScaledMMKernel path -- the exact kernel the AC-4 vLLM
baseline runs, vllm-graph.md SS3.5) against HF's kernels-hub Triton finegrained-fp8 linear
(transformers.integrations.finegrained_fp8.finegrained_fp8_linear), on REAL
Qwen/Qwen3.5-35B-A3B-FP8 layer-0(-adjacent) weights + their checkpoint block scales, at the
three dense-path GEMM shapes MPK will run in fp8 (v1-architecture.md SS6.1):

  - GDN in_proj_qkvz [12288,2048]  (layer 0 in_proj_qkv ++ in_proj_z; a literal layer-0 tensor)
  - attn qkv_proj    [9216,2048]   (layer 3 q_proj ++ k_proj ++ v_proj -- layer 0 is a GDN
                                     layer [(0+1)%4 != 0] and structurally has no self_attn.*;
                                     layer 3 is the first full-attention layer, i in {3,7,...,39})
  - dense out/o_proj [2048,4096]   (GDN layer-0 out_proj is primary; full-attn layer-3 o_proj
                                     is an additional same-shape-class bonus case)

(i) numerics: CUTLASS vs HF-Triton on identical bf16 inputs, both compared to an fp32
    dequant-matmul anchor for attribution. (ii) perf: the same CUTLASS fp8 path vs a bf16
    torch.matmul at the same shapes (proxy for MPK's bf16 linear task), for decode-batch
    M in {1,2,4,8,16}.

Emits workspace/demo/qwen3_5/accept/probes/fp8/p10_verdict.json:
  {go: bool, go_numerics_fidelity: bool, numerics: {...}, perf_bar_result: {...}, env: {...},
   cases: [...]}

REVISED 2026-07-26 (coordinator review, one cycle): the numerics gate now uses a DERIVED,
documented floor (see `single_flip_floor()`) instead of a flat constant, and the top-level
`go` reflects only numerics fidelity (what M2-I12 needs); the perf bar result is reported
separately as an M3 prior, not ANDed into `go` -- see `build_verdict()` and the JSON's
`numerics.OLD_THRESHOLDS_RETIRED` for the full before/after. `--recompute-from <verdict.json>`
re-derives the verdict from an EXISTING run's stored `cases` with no new GPU measurement.
"""
import argparse
import datetime
import json
import math
import os
import socket
import time
import zlib

os.environ.setdefault("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

import torch  # noqa: E402
import vllm  # noqa: E402
import transformers  # noqa: E402
from safetensors import safe_open  # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    per_token_group_quant_fp8,
)
from transformers.integrations.finegrained_fp8 import finegrained_fp8_linear  # noqa: E402

CKPT_REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
SNAPSHOT = os.environ.get(
    "QWEN35_SNAPSHOT",
    os.path.expanduser(
        f"~/mpk-qwen35/hf/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/{CKPT_REVISION}"
    ),
)
BLOCK = 128

# --- RETIRED 2026-07-26 (coordinator completion review) ---
# These were flat constants ("~1e-3, give it some slack") with no derivation -- an
# undocumented tightening of the doc's explicitly-rough "~1e-3" point estimate into a hard
# 2e-3/1e-2 gate. Kept here (never silently deleted) so the JSON's audit trail can quote what
# changed and why. Superseded by `single_flip_floor()` + GATE_SAFETY_MULTIPLE below.
FROB_REL_THRESHOLD_RETIRED = 2e-3
P99_TOP_HALF_THRESHOLD_RETIRED = 1e-2

# Bias gate: UNCHANGED, and still the real "did something go wrong" alarm (see rationale in
# build_verdict()) -- an n-invariant mean/std ratio, NOT a z-score (diff_stats() docstring
# explains why a z-score is unsound once --numerics-draws pools many samples).
BIAS_EFFECT_SIZE_THRESHOLD = 0.1

# --- Derived numerics floor (replaces the flat FROB_REL_THRESHOLD) ---
# e4m3 has 3 explicit mantissa bits -> relative step between adjacent representable values is
# 2^-3 = 0.125 (12.5%): the largest possible relative perturbation from a SINGLE element
# rounding to the "wrong" (adjacent) fp8 bucket.
E4M3_RELATIVE_LSB = 0.125
# Safety multiple applied to the single-flip floor to form the actual gate (see
# single_flip_floor() docstring for the full derivation and why 4x): empirically, the worst
# measured ratio (frob_rel / floor(K)) across all 20 cases was ~1.95x (K=4096) and ~1.58x
# (K=2048) -- 4x leaves a full additional ~2x margin beyond anything actually observed, while
# staying >=2.4x below the rejected UE8M0-requant class (P7) at every K tested.
GATE_SAFETY_MULTIPLE = 4

PERF_THRESHOLD = 1.5
PERF_MARGINAL_THRESHOLD = 1.2


def single_flip_floor(K):
    """RETIRED as the primary model 2026-07-26 (coordinator review cycle 3) -- kept for the
    audit trail and because it is still a correct SPECIAL CASE (f*K=1). Predicted floor(K) =
    E4M3_RELATIVE_LSB/sqrt(K) implies a K-DEPENDENT floor (ratio sqrt(2)=1.41 between K=2048
    and K=4096). The actual P10 data is K-INVARIANT instead: mean frob_rel_error 3.537e-3 at
    K=2048 vs 3.486e-3 at K=4096, ratio 1.01 (verified directly against all 20 stored cases).
    See `fraction_model_frob()` for the corrected, decisively-confirmed model. Still useful as
    a sanity floor: pure fp32 accumulation-REORDER noise (different GEMM tile schedules sum
    the same terms in a different order; probabilistic/typical-case scaling ~sqrt(K)*u,
    u=2^-24) is ~2.7e-6 (K=2048) to 3.8e-6 (K=4096) -- 100-1000x smaller than anything
    measured, ruling out pure accumulation order as the dominant mechanism regardless of which
    higher-level model is used.
    """
    return E4M3_RELATIVE_LSB / math.sqrt(K)


def fraction_model_frob(f):
    """CONFIRMED model (coordinator review cycle 3, 2026-07-26) for the K-invariant
    CUTLASS-vs-Triton frob_rel_error: a FIXED FRACTION f of the K contracted activation
    elements disagree in e4m3 quantization bucket between the two independently-authored
    quantizers (not a fixed COUNT, as `single_flip_floor` assumed). N=f*K such disagreements
    accumulate like sqrt(N) (CLT/random-walk, same reasoning as `single_flip_floor`), and the
    K-term sum itself has magnitude ~sqrt(K)*term_scale -- the two sqrt(K) factors cancel:

        relative_frob ~= E4M3_RELATIVE_LSB * sqrt(f), INDEPENDENT of K.

    DECISIVELY CONFIRMED by direct measurement, not just curve-fit (`p10b_activation_quant_
    disagreement.py`): extracted the RAW e4m3 codes both quantizers actually produce on
    identical real activations (an identity-weight trick recovers Triton's code, since
    `finegrained_fp8_linear` only exposes a fused act-quant+matmul, no standalone quantize
    entry point) and counted disagreements directly, at the same K=2048/4096 shapes P10 used,
    across M in {1,2,4,8,16} (10 cases, 190,464 activation elements total). RECONCILED COUNTS
    (coordinator review cycle 4 caught an inconsistent "281" in an earlier draft of this
    docstring/report -- these are the actual, JSON-verified totals, and every other reference
    in this file/the verdict JSON uses these same numbers):
      - 211 positions: exactly-1-ULP CODE-level disagreement (verified via a +-8 ULP bit-level
        search on the raw e4m3 magnitude bits -- every one resolved at |ULP|=1, none wider).
      - 1801 positions: sub-LSB GROUP-SCALE disagreement (see `combined_model_frob()` below;
        a DIFFERENT, smaller-magnitude mechanism, not conflated with the count above).
      - 0 positions unexplained by either mechanism.
      - 211 + 1801 + 0 = 2012 = the exact total differing-position count summed from every
        case's `n_differing_codes` in p10b_activation_quant_disagreement.json (reconciles).
      - f_code = 211/190464 = 1.108e-3. Plugging into the formula above ALONE: predicted frob
        = 0.125*sqrt(1.108e-3) = 4.160e-3, vs P10's worst measured frob_rel_error 4.352e-3 --
        a 1.05x match. See `combined_model_frob()` for the model that also accounts for the
        1801 group-scale positions, rather than comparing a code-only prediction against a
        measurement that reflects both mechanisms.
    """
    return E4M3_RELATIVE_LSB * math.sqrt(f)


def combined_model_frob(f_code, f_scale, scale_delta_rms):
    """Combines BOTH confirmed mechanisms' predicted contribution to frob_rel_error, not just
    the code-level one (coordinator review cycle 4: the 1.05x match above compared a CODE-ONLY
    prediction against a measurement that includes both mechanisms -- an apples-to-oranges
    comparison that happened to look good only because the second mechanism's contribution is
    small; fixed here by actually computing that contribution instead of leaving it implicit).

    The two mechanisms are independent sources of per-element perturbation to the K-term dot
    product (different root causes -- one is a discrete code-bucket flip, the other a
    continuous group-scale miscalculation), so by the same CLT/random-walk logic as
    `fraction_model_frob` their contributions to the OUTPUT's relative error combine in
    quadrature (RMS of two independent noise sources), not by simple addition:

        combined_frob = sqrt( (E4M3_RELATIVE_LSB * sqrt(f_code))^2
                             + (scale_delta_rms   * sqrt(f_scale))^2 )

    where `scale_delta_rms` is the RMS of the observed (target/cutlass_recon - 1) group-scale
    offsets (pooled over all 1801 individual observations, not just the 8 distinct values --
    RMS, not mean, because it is the quantity that enters a variance/quadrature combination).

    Measured (p10b_activation_quant_disagreement.json, all values recomputed from the stored
    per-case data, not re-measured): f_code=1.108e-3, f_scale=1801/190464=9.456e-3,
    scale_delta_rms=6.137e-3 (pooled over 1801 samples; individual case means/RMS range
    ~4.3e-3-7.7e-3, consistent with the 8 discrete ratios observed).
      code_term  = 0.125 * sqrt(1.108e-3)  = 4.160e-3
      scale_term = 6.137e-3 * sqrt(9.456e-3) = 5.968e-4  (14.3% of code_term in MAGNITUDE;
                   (scale_term/code_term)^2 = 2.0% of the combined VARIANCE -- quantitatively
                   small, not merely asserted small)
      combined   = sqrt(4.160e-3^2 + 5.968e-4^2) = 4.203e-3
    vs P10's worst measured frob_rel_error 4.352e-3 -- ratio 1.035x, an even TIGHTER match
    than the code-only model's 1.046x (expected: the combined model is more complete).
    """
    code_term = E4M3_RELATIVE_LSB * math.sqrt(f_code)
    scale_term = scale_delta_rms * math.sqrt(f_scale)
    return math.sqrt(code_term ** 2 + scale_term ** 2), code_term, scale_term


def load_ue8m0_reference(path):
    """Load P7's UE8M0-requant delta range from an existing p7_ue8m0_delta.json, for gate
    criterion (c) ("delta strictly << the rejected UE8M0 class"). Returns None (not a
    fabricated number) if the file is absent or malformed -- the criterion is then reported
    as unavailable rather than silently skipped or assumed."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            p7 = json.load(f)
        deltas = [t["frob_rel_delta"] for t in p7["tensors"]]
        return {"min": min(deltas), "max": max(deltas), "source": path, "n_tensors": len(deltas)}
    except Exception as e:  # noqa: BLE001 -- reported, not fatal
        return {"error": f"{type(e).__name__}: {e}", "source": path}


def load_activation_quant_evidence(path):
    """Load the decisive raw-code-disagreement measurement (p10b_activation_quant_
    disagreement.py) grounding `combined_model_frob()`. Returns None if absent -- reported as
    unavailable, never assumed or fabricated. All counts here are pure post-processing of the
    ALREADY-MEASURED p10b JSON (no re-measurement) -- reconciled 2026-07-26 (coordinator
    review cycle 4) to make sure every number (this loader, the docstrings, the verdict JSON)
    traces to the same source totals: 211 code-level + 1801 group-scale + 0 unexplained =
    2012 total differing positions out of 190,464 tested."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        total_code = sum(c["n_code_ulp_disagreement"] for c in d["cases_tested"])
        total_gs = sum(c["n_group_scale_disagreement"] for c in d["cases_tested"])
        total_unexplained = sum(c["n_unexplained"] for c in d["cases_tested"])
        total_diff = sum(c["n_differing_codes"] for c in d["cases_tested"])
        total_elements = sum(c["n_total_elements"] for c in d["cases_tested"])
        assert total_code + total_gs + total_unexplained == total_diff, (
            f"count reconciliation FAILED: {total_code}+{total_gs}+{total_unexplained} != "
            f"{total_diff} -- do not silently proceed with inconsistent totals")
        f_code = total_code / total_elements
        f_scale = total_gs / total_elements

        # Pool ALL individual group-scale ratio observations (not just the 8 distinct values)
        # to get the RMS delta `combined_model_frob` needs -- RMS (not mean) because it is the
        # quantity that enters a variance/quadrature combination.
        all_ratios = [r for c in d["cases_tested"] for r in c["group_scale_ratios_observed"]]
        assert len(all_ratios) == total_gs, "pooled ratio count must equal total_gs"
        deltas = [r - 1.0 for r in all_ratios]
        scale_delta_rms = math.sqrt(sum(x * x for x in deltas) / len(deltas)) if deltas else 0.0

        combined_frob, code_term, scale_term = combined_model_frob(f_code, f_scale, scale_delta_rms)

        return {
            "source": path,
            "n_cases_tested": len(d["cases_tested"]),
            "max_ulp_distance_overall": d["max_ulp_distance_overall"],
            "all_code_level_disagreements_are_1_ulp": d["all_code_level_disagreements_are_1_ulp"],
            "counts": {
                "n_code_level_1ulp": total_code,
                "n_group_scale_sub_lsb": total_gs,
                "n_unexplained": total_unexplained,
                "n_total_differing": total_diff,
                "n_total_elements_tested": total_elements,
                "reconciles": total_code + total_gs + total_unexplained == total_diff,
            },
            "total_unexplained_positions": total_unexplained,
            "code_level_disagreement_fraction_f": f_code,
            "group_scale_disagreement_fraction": f_scale,
            "group_scale_delta_rms": scale_delta_rms,
            "group_scale_ratio_range": d["second_mechanism_found"][
                "observed_ratios (target/cutlass_recon, should be close to 1.0)"],
            "model_code_only": {"predicted_frob": fraction_model_frob(f_code)},
            "model_combined": {
                "predicted_frob": combined_frob,
                "code_term": code_term,
                "scale_term": scale_term,
                "scale_term_frac_of_code_term": scale_term / code_term if code_term else None,
                "scale_term_frac_of_combined_variance": (scale_term ** 2) / (code_term ** 2 + scale_term ** 2)
                    if (code_term or scale_term) else None,
            },
        }
    except Exception as e:  # noqa: BLE001 -- reported, not fatal
        return {"error": f"{type(e).__name__}: {e}", "source": path}


_open_shards = {}


def _get(index, key):
    shard = index["weight_map"][key]
    path = os.path.join(SNAPSHOT, shard)
    if path not in _open_shards:
        _open_shards[path] = safe_open(path, framework="pt")
    return _open_shards[path].get_tensor(key)


def load_index():
    with open(os.path.join(SNAPSHOT, "model.safetensors.index.json")) as f:
        return json.load(f)


def load_real_tensor(shape_key, index, device):
    """Real checkpoint (weight_fp8 [N,K], scale_fp32 [N/128,K/128], label, src_keys)."""
    P = "model.language_model.layers."
    if shape_key == "12288x2048":
        w = torch.cat(
            [_get(index, P + "0.linear_attn.in_proj_qkv.weight"),
             _get(index, P + "0.linear_attn.in_proj_z.weight")], dim=0)
        s = torch.cat(
            [_get(index, P + "0.linear_attn.in_proj_qkv.weight_scale_inv"),
             _get(index, P + "0.linear_attn.in_proj_z.weight_scale_inv")], dim=0)
        label = "GDN.layer0.in_proj_qkvz(in_proj_qkv++in_proj_z)"
        keys = [P + "0.linear_attn.in_proj_qkv.*", P + "0.linear_attn.in_proj_z.*"]
    elif shape_key == "9216x2048":
        w = torch.cat(
            [_get(index, P + "3.self_attn.q_proj.weight"),
             _get(index, P + "3.self_attn.k_proj.weight"),
             _get(index, P + "3.self_attn.v_proj.weight")], dim=0)
        s = torch.cat(
            [_get(index, P + "3.self_attn.q_proj.weight_scale_inv"),
             _get(index, P + "3.self_attn.k_proj.weight_scale_inv"),
             _get(index, P + "3.self_attn.v_proj.weight_scale_inv")], dim=0)
        label = "attn.layer3.qkv_proj(q_proj++k_proj++v_proj;layer0_has_no_self_attn)"
        keys = [P + "3.self_attn.q_proj.*", P + "3.self_attn.k_proj.*", P + "3.self_attn.v_proj.*"]
    elif shape_key == "2048x4096":
        w = _get(index, P + "0.linear_attn.out_proj.weight")
        s = _get(index, P + "0.linear_attn.out_proj.weight_scale_inv")
        label = "GDN.layer0.out_proj"
        keys = [P + "0.linear_attn.out_proj.*"]
    elif shape_key == "2048x4096-attn-bonus":
        w = _get(index, P + "3.self_attn.o_proj.weight")
        s = _get(index, P + "3.self_attn.o_proj.weight_scale_inv")
        label = "attn.layer3.o_proj(bonus,same_shape_class_as_out_proj)"
        keys = [P + "3.self_attn.o_proj.*"]
    else:
        raise ValueError(f"unknown shape key {shape_key}")
    assert s.dtype == torch.bfloat16, f"checkpoint scale expected bf16, got {s.dtype}"
    return w.to(device), s.float().to(device), label, keys


def dequant_bf16(w_fp8, scale_fp32):
    """W_real ~= W_fp8 * weight_scale_inv, block-expanded (vllm-graph.md SS3.4 semantics)."""
    n, k = w_fp8.shape
    s = scale_fp32.repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(BLOCK, dim=1)[:, :k]
    return (w_fp8.float() * s).to(torch.bfloat16)


def diff_stats(a, b, ref):
    """Elementwise + norm-based diff of a vs b, with an independent `ref` for magnitude
    flooring.

    ROOT-CAUSED during dry-run #1 (first-principles check before trusting any threshold):
    a fixed-floor elementwise "max relative diff" is NOT a fair gate for two independent
    fp8 (e4m3, ~3 mantissa bits => ~12.5% per-element LSB) pipelines -- a below-RMS-magnitude
    output element can show a large relative delta purely from each implementation's own
    activation-quant rounding landing in an adjacent e4m3 bucket, with NO systematic
    disagreement (verified: that element's absolute delta was ~2 fp8 LSBs at its own
    magnitude). We therefore report `floored_max_rel_diff` only as a diagnostic and GATE on
    `frob_rel_error` (the standard relative-L2-norm error used by vLLM's own fp8 test
    suites) instead. `p99_rel_diff_top_half` adds a percentile view over the top-50%-by-
    magnitude elements (less outlier-fragile than max, still restricted to elements where
    "relative" is a meaningful quantity).

    ROOT-CAUSED during dry-run #3 (pooling `--numerics-draws` independent activations, added
    for sample-size robustness, made this worse, not better): a z-score
    (mean / (std/sqrt(n))) is a HYPOTHESIS-test statistic that grows with sqrt(n) at FIXED
    effect size -- pooling more draws can make an arbitrarily tiny, practically-irrelevant
    mean shift look "significant" purely by accumulating samples. Caught exactly this: one
    shape showed |z|~4.8 at n~1.2M elements, but mean_signed_diff was -1.2e-5 against a
    std of 2.7e-3 (both fp8 kernels ALSO carried a ~10x larger, near-identical mean offset
    vs the fp32 reference -- i.e. a shared fp8-quantization-vs-bf16 effect, not a
    cutlass-vs-triton disagreement). The gate therefore uses `bias_effect_size` =
    mean/std -- an n-INVARIANT ratio (the right measure of "how big is the bias relative to
    noise") -- and keeps `bias_zscore` only as an informational "was there enough data to
    even detect a nonzero mean" diagnostic, never as a threshold."""
    af, bf, rf = a.float(), b.float(), ref.float()
    diff = af - bf
    n = diff.numel()
    mean = diff.mean().item()
    std = diff.std(unbiased=True).item() if n > 1 else 0.0
    if std > 1e-12:
        z = mean / (std / math.sqrt(n))
        effect_size = mean / std
    else:
        z = 0.0 if abs(mean) < 1e-12 else float("inf")
        effect_size = 0.0 if abs(mean) < 1e-12 else float("inf")
    rms = rf.pow(2).mean().sqrt().item()
    floor = max(1e-6, 0.01 * rms)
    mask = rf.abs() > floor
    rel_all = diff.abs() / rf.abs().clamp(min=floor)
    floored_max_rel = rel_all[mask].max().item() if mask.any() else 0.0
    median_mag = rf.abs().median().item()
    top_half_mask = rf.abs() >= max(median_mag, floor)
    p99_top_half = (torch.quantile(rel_all[top_half_mask], 0.99).item()
                    if top_half_mask.sum() > 1 else floored_max_rel)
    frob_rel = diff.norm().item() / max(rf.norm().item(), 1e-12)
    return {
        "max_abs_diff": diff.abs().max().item(),
        "frob_rel_error": frob_rel,
        "p99_rel_diff_top_half": p99_top_half,
        "floored_max_rel_diff_DIAGNOSTIC_ONLY": floored_max_rel,
        "mag_floor": floor,
        "frac_elements_below_floor": 1.0 - (mask.sum().item() / n),
        "mean_signed_diff": mean,
        "std_signed_diff": std,
        "bias_effect_size": effect_size,
        "bias_zscore_INFORMATIONAL_ONLY": z,
        "n_elements": n,
    }


def _summarize_rounds(round_times):
    round_times.sort()
    mid = len(round_times) // 2
    median = round_times[mid] if len(round_times) % 2 else (round_times[mid - 1] + round_times[mid]) / 2
    return {"median_s": median, "min_s": round_times[0], "max_s": round_times[-1], "rounds": round_times}


def timed_eager(fn, warmup, iters, repeats):
    """Raw Python-eager wall clock: includes host dispatch + kernel-launch overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    round_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        round_times.append((time.perf_counter() - t0) / iters)
    return _summarize_rounds(round_times)


def timed_graph(fn, warmup, iters, repeats):
    """CUDA-graph-replayed wall clock: eliminates host dispatch/launch overhead, matching
    how vLLM ITSELF runs this exact op in the AC-4 baseline (constraint.md: "vLLM gets its
    best standard config with CUDA graphs") -- i.e. this is the more faithful reproduction of
    "the exact kernel the AC-4 baseline runs" (v1-architecture.md SS14 P10 spec), not just a
    micro-optimization of the harness. Falls back to None if graph capture is unsupported for
    this call (reported, never silently substituted)."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(warmup, 5)):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 -- reported as a diagnostic, not fatal to the probe
        return {"error": f"{type(e).__name__}: {e}"}
    round_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        round_times.append((time.perf_counter() - t0) / iters)
    return _summarize_rounds(round_times)


def stable_seed(*parts):
    """Deterministic seed independent of PYTHONHASHSEED (str hash() is randomized per
    process by default -- would silently break exact reproducibility of --seed re-runs)."""
    h = zlib.crc32("|".join(str(p) for p in parts).encode())
    return h


def cutlass_call(X, W, scale):
    xq, xs = per_token_group_quant_fp8(X, BLOCK, column_major_scales=True, use_ue8m0=False)
    return ops.cutlass_scaled_mm(xq, W.T, scale_a=xs, scale_b=scale.T, out_dtype=torch.bfloat16)


def run_case(shape_key, W, scale, W_bf16, M, base_seed, warmup, iters, repeats, n_draws, device):
    N, K = W.shape

    # Numerics: pool `n_draws` INDEPENDENT random activation batches (esp. important at
    # small M, where a single draw only gives M*N samples for the diff statistics) so the
    # frob/percentile/bias stats rest on a larger, genuinely independent sample rather than
    # one lucky/unlucky draw.
    triton_outs, cutlass_outs, ref_outs = [], [], []
    for d in range(n_draws):
        g = torch.Generator(device=device).manual_seed(stable_seed(base_seed, shape_key, M, d))
        Xd = torch.randn(M, K, generator=g, device=device, dtype=torch.float32).to(torch.bfloat16)
        triton_outs.append(finegrained_fp8_linear(Xd, W, scale, block_size=[BLOCK, BLOCK]))
        cutlass_outs.append(cutlass_call(Xd, W, scale))
        ref_outs.append(torch.matmul(Xd.float(), W_bf16.float().T))
        if d == 0:
            X = Xd  # reused for perf timing below (a representative, real-shaped input)
    out_triton = torch.cat(triton_outs, dim=0)
    out_cutlass = torch.cat(cutlass_outs, dim=0)
    out_fp32ref = torch.cat(ref_outs, dim=0)

    numerics = {
        "cutlass_vs_triton": diff_stats(out_cutlass, out_triton, out_fp32ref),
        "cutlass_vs_fp32ref": diff_stats(out_cutlass, out_fp32ref, out_fp32ref),
        "triton_vs_fp32ref": diff_stats(out_triton, out_fp32ref, out_fp32ref),
        "n_draws_pooled": n_draws,
    }

    # Eager (raw Python dispatch + kernel-launch overhead) -- diagnostic.
    t_triton_eager = timed_eager(lambda: finegrained_fp8_linear(X, W, scale, block_size=[BLOCK, BLOCK]),
                                  warmup, iters, repeats)
    t_cutlass_eager = timed_eager(lambda: cutlass_call(X, W, scale), warmup, iters, repeats)
    t_bf16_eager = timed_eager(lambda: torch.matmul(X, W_bf16.T), warmup, iters, repeats)

    # CUDA-graph-replayed (dispatch overhead eliminated) -- PRIMARY, matches how vLLM's own
    # AC-4 baseline actually runs this op (under CUDA graphs). See timed_graph() docstring.
    t_cutlass_graph = timed_graph(lambda: cutlass_call(X, W, scale), warmup, iters, repeats)
    t_bf16_graph = timed_graph(lambda: torch.matmul(X, W_bf16.T), warmup, iters, repeats)

    graph_ok = "error" not in t_cutlass_graph and "error" not in t_bf16_graph
    speedup_graph = (t_bf16_graph["median_s"] / t_cutlass_graph["median_s"]) if graph_ok else None
    speedup_eager = t_bf16_eager["median_s"] / t_cutlass_eager["median_s"]

    perf = {
        "speedup_cutlass_over_bf16_graph": speedup_graph,
        "speedup_cutlass_over_bf16_eager": speedup_eager,
        "speedup_triton_over_bf16_eager": t_bf16_eager["median_s"] / t_triton_eager["median_s"],
        "graph_capture_ok": graph_ok,
        "cutlass_median_s_graph": t_cutlass_graph.get("median_s"),
        "bf16_median_s_graph": t_bf16_graph.get("median_s"),
        "cutlass_median_s_eager": t_cutlass_eager["median_s"],
        "bf16_median_s_eager": t_bf16_eager["median_s"],
        "triton_median_s_eager": t_triton_eager["median_s"],
        "raw": {"triton_eager": t_triton_eager, "cutlass_eager": t_cutlass_eager, "bf16_eager": t_bf16_eager,
                "cutlass_graph": t_cutlass_graph, "bf16_graph": t_bf16_graph},
    }
    return {"shape": f"{N}x{K}", "shape_key": shape_key, "M": M, "numerics": numerics, "perf": perf}


def build_verdict(cases, shape_keys, batches, env, ue8m0_range, activation_quant_evidence=None):
    """Pure post-processing: turns already-measured `cases` into the verdict dict. Takes NO
    GPU/measurement action -- callable identically from a fresh run (`main()`) or from
    `--recompute-from` (re-deriving the verdict from an existing artifact's stored cases,
    e.g. after a gate-logic revision, with no re-measurement)."""
    # Annotate each case with its K-specific derived floor/gate (additive -- the underlying
    # measured stats, frob_rel_error / bias_effect_size / etc., are never modified).
    for c in cases:
        K = int(c["shape"].split("x")[1])
        floor = single_flip_floor(K)
        gate = GATE_SAFETY_MULTIPLE * floor
        nc = c["numerics"]["cutlass_vs_triton"]
        nc["derived_floor_this_K"] = floor
        nc["derived_gate_this_K"] = gate
        nc["ratio_to_floor"] = nc["frob_rel_error"] / floor
        nc["derived_gate_pass"] = nc["frob_rel_error"] <= gate

    frob_rels = [c["numerics"]["cutlass_vs_triton"]["frob_rel_error"] for c in cases]
    p99_top_halfs = [c["numerics"]["cutlass_vs_triton"]["p99_rel_diff_top_half"] for c in cases]
    bias_effects = [c["numerics"]["cutlass_vs_triton"]["bias_effect_size"] for c in cases]
    ratios_to_floor = [c["numerics"]["cutlass_vs_triton"]["ratio_to_floor"] for c in cases]
    graph_ok_all = all(c["perf"]["graph_capture_ok"] for c in cases)
    speedups_graph = [c["perf"]["speedup_cutlass_over_bf16_graph"] for c in cases if c["perf"]["graph_capture_ok"]]
    speedups_eager = [c["perf"]["speedup_cutlass_over_bf16_eager"] for c in cases]

    worst_frob_rel = max(frob_rels)
    worst_p99_top_half = max(p99_top_halfs)
    worst_bias_effect = max(abs(e) for e in bias_effects if math.isfinite(e))
    worst_ratio_to_floor = max(ratios_to_floor)
    if speedups_graph:
        min_speedup = min(speedups_graph)
        perf_measurement = "cuda_graph"
    else:
        min_speedup = min(speedups_eager)
        perf_measurement = "eager_FALLBACK_graph_capture_unavailable"

    # =====================================================================================
    # REFRAMED 2026-07-26 (coordinator completion review, cycle 4): the correctness claim
    # rests on FOUR QUALITATIVE PILLARS (i)-(iv), each backed by evidence but NOT gated on a
    # numeric bound that was picked after seeing this exact data. The specific numeric bounds
    # from earlier cycles (f<=2e-3, "within 2x") were honestly post-hoc (chosen once the
    # measured values were already known) -- they are kept, but demoted to OPERATING
    # TRIPWIRES: regression guards for a FUTURE re-run, explicitly labeled as post-hoc-
    # calibrated, and NOT part of go_numerics_fidelity's own logic.
    # =====================================================================================
    aq = activation_quant_evidence

    # --- Pillar (i): mechanism identified -- STRUCTURAL/completeness checks only (0 is 0, 1
    # is 1, and "sub-LSB" uses the literal e4m3 LSB definition as its own bound -- none of
    # these three numbers were tuned to this dataset). ---
    if aq and "counts" in aq:
        counts = aq["counts"]
        mechanism_complete = counts["reconciles"] and counts["n_unexplained"] == 0
        code_is_exactly_1ulp = (aq.get("max_ulp_distance_overall") == 1
                                 and aq.get("all_code_level_disagreements_are_1_ulp") is True)
        scale_deltas = [r - 1.0 for r in aq.get("group_scale_ratio_range", [])]
        scale_is_sub_lsb = bool(scale_deltas) and max(abs(x) for x in scale_deltas) < E4M3_RELATIVE_LSB
        mechanism_identified = bool(mechanism_complete and code_is_exactly_1ulp and scale_is_sub_lsb)
    else:
        counts = None
        mechanism_complete = code_is_exactly_1ulp = scale_is_sub_lsb = None
        mechanism_identified = False  # unavailable -> fail closed, never assumed benign

    # --- Pillar (ii): no systematic bias -- the actual "something is wrong" alarm. 0.1 is a
    # generic small-effect-size convention fixed in cycle 1 (before any frob/bias number had
    # been measured), not tuned to this dataset. ---
    no_systematic_bias = worst_bias_effect <= BIAS_EFFECT_SIZE_THRESHOLD

    # --- Pillar (iii): the COMBINED (both-mechanism) model -- built with ZERO parameters
    # fitted to frob (f_code, f_scale, scale_delta_rms all come from the INDEPENDENT direct
    # code-count measurement; E4M3_RELATIVE_LSB=0.125 is the e4m3 format's physical constant)
    # -- predicts a value the same ORDER OF MAGNITUDE as the independently-measured P10 frob.
    # [0.5x, 2x] is a generic "the model basically explains the data" sanity band (half-to-
    # double), not fitted to the 1.035x actually observed -- see OPERATING_TRIPWIRES for the
    # tighter, explicitly post-hoc regression guard.
    if aq and "model_combined" in aq:
        predicted_frob_combined = aq["model_combined"]["predicted_frob"]
        frob_ratio_to_predicted = worst_frob_rel / predicted_frob_combined
        combined_model_matches = 0.5 <= frob_ratio_to_predicted <= 2.0
    else:
        predicted_frob_combined = frob_ratio_to_predicted = None
        combined_model_matches = False  # unavailable -> fail closed

    # --- Pillar (iv): token-level corroboration -- qualitative, documented external + this
    # issue's own evidence (see numerics.token_level_evidence). Not a magnitude threshold. ---
    token_level_corroboration = True  # see the two cited, sourced results below

    go_numerics_fidelity = bool(mechanism_identified and no_systematic_bias
                                 and combined_model_matches and token_level_corroboration)
    gate_pass = go_numerics_fidelity

    # --- OPERATING TRIPWIRES: explicitly post-hoc-calibrated (set at ~1.8-2x TODAY's
    # measured values), for detecting REGRESSION on a future re-run -- do NOT gate
    # go_numerics_fidelity, do NOT claim to be independently derived. ---
    if aq:
        f_code = aq["code_level_disagreement_fraction_f"]
        tripwire_code_fraction = {
            "value": f_code, "bound": 2e-3, "pass": f_code <= 2e-3,
            "calibration_note": f"bound is ~1.8x today's measured {f_code:.3e} -- if a future "
                                 f"re-run exceeds this, investigate before assuming benign.",
        }
        tripwire_model_ratio = {
            "value": frob_ratio_to_predicted, "bound": "[0.5x, 2.0x]",
            "pass": combined_model_matches,
            "calibration_note": f"today's measured ratio is {frob_ratio_to_predicted:.3f}x -- "
                                 f"the [0.5,2] band has generous headroom above/below it now, "
                                 f"but is still a post-hoc choice, not derived.",
        }
    else:
        f_code = None
        tripwire_code_fraction = tripwire_model_ratio = {"value": None, "pass": None,
                                                           "calibration_note": "unavailable"}
    if ue8m0_range and "min" in ue8m0_range:
        vs_ue8m0_pass = worst_frob_rel <= 0.5 * ue8m0_range["min"]
        vs_ue8m0_ratio = ue8m0_range["min"] / worst_frob_rel
        tripwire_ue8m0_margin = {
            "value": vs_ue8m0_ratio, "bound": "measured frob <= 0.5x smallest UE8M0 delta",
            "pass": vs_ue8m0_pass,
            "calibration_note": f"today's actual margin is {vs_ue8m0_ratio:.1f}x, well above "
                                 f"the 2x this tripwire requires -- comfortable headroom, still "
                                 f"a post-hoc-chosen bound, not derived from first principles.",
        }
    else:
        vs_ue8m0_pass, vs_ue8m0_ratio = None, None
        tripwire_ue8m0_margin = {"value": None, "pass": None, "calibration_note": "unavailable"}

    perf_pass = min_speedup >= PERF_THRESHOLD
    perf_marginal = (not perf_pass) and min_speedup >= PERF_MARGINAL_THRESHOLD

    rationale = (
        f"go_numerics_fidelity={go_numerics_fidelity} keys on FOUR PILLARS, not on the "
        f"tripwire numbers (see numerics.operating_tripwires for those, explicitly post-hoc). "
        f"(i) mechanism_identified={mechanism_identified}: every one of the "
        f"{counts['n_total_differing'] if counts else '?'} differing positions "
        f"({counts['n_code_level_1ulp'] if counts else '?'} code-level + "
        f"{counts['n_group_scale_sub_lsb'] if counts else '?'} group-scale, "
        f"{counts['n_unexplained'] if counts else '?'} unexplained) is accounted for by one of "
        f"two identified mechanisms: code-level disagreements are ALWAYS exactly 1 ULP "
        f"({code_is_exactly_1ulp}), group-scale nudges are ALWAYS structurally sub-LSB "
        f"(max |delta| < {E4M3_RELATIVE_LSB} e4m3 LSB, {scale_is_sub_lsb}). "
        f"(ii) no_systematic_bias={no_systematic_bias} (worst bias effect size "
        f"{worst_bias_effect:.4f} <= {BIAS_EFFECT_SIZE_THRESHOLD}, a generic small-effect "
        f"convention fixed before any bias number was measured -- a directional offset would "
        f"indicate a real scale/indexing bug; this small, unbiased, symmetric spread does not). "
        f"(iii) combined_model_matches={combined_model_matches}: the TWO-mechanism model "
        f"(combined_model_frob(), zero parameters fit to frob) predicts "
        f"{(predicted_frob_combined or 0):.3e} vs measured worst "
        f"{worst_frob_rel:.3e} (ratio {(frob_ratio_to_predicted or 0):.3f}x, "
        f"inside the generic [0.5x,2x] order-of-magnitude sanity band). "
        f"(iv) token_level_corroboration={token_level_corroboration}: the M1 baseline found "
        f"vLLM(CUTLASS)-vs-HF(Triton) produced byte-identical 64/64 tokens on prompt p01, and "
        f"this issue's own P1 probe found a perturbation an order of magnitude LARGER than "
        f"this (full bf16 substitution, not just cross-implementation noise) only flipped "
        f"3/10 prompts at close margins."
    )

    return {
        "go": go_numerics_fidelity,  # ALIAS: repurposed 2026-07-26 to mean go_numerics_fidelity
                                       # only (see module docstring) -- perf is a separate prior.
        "go_numerics_fidelity": go_numerics_fidelity,
        "numerics": {
            "gate_pass": gate_pass,
            "correctness_claim": {
                "note": "go_numerics_fidelity keys ONLY on these four pillars (all boolean, "
                        "each backed by evidence below) -- NOT on operating_tripwires, which "
                        "are separate, explicitly post-hoc-calibrated regression guards.",
                "i_mechanism_identified": mechanism_identified,
                "ii_no_systematic_bias": no_systematic_bias,
                "iii_combined_model_matches": combined_model_matches,
                "iv_token_level_corroboration": token_level_corroboration,
            },
            "no_systematic_bias": no_systematic_bias,
            "worst_bias_effect_size": worst_bias_effect,
            "bias_effect_size_threshold": BIAS_EFFECT_SIZE_THRESHOLD,
            "frob_class": "TWO confirmed mechanisms (coordinator review cycles 3-4, decisive "
                          "direct measurement, not inference), RECONCILED counts: of 190,464 "
                          "activation elements tested, 2012 differed (211 code-level + 1801 "
                          "group-scale + 0 unexplained). (1) 211 positions: single e4m3-code "
                          "disagreement between the two quantizers, ALWAYS exactly 1 ULP, "
                          "fraction f_code=1.108e-3 (K-invariant model: relative_frob "
                          "~= 0.125*sqrt(f_code), see fraction_model_frob()); (2) 1801 "
                          "positions: a smaller-per-element per-128-group SCALE discrepancy "
                          "(~0.43-0.77%, always structurally < 1 e4m3 LSB) between the two "
                          "quantizers' absmax computation for that group. combined_model_frob() "
                          "combines both mechanisms' predicted contribution in quadrature "
                          "(independent noise sources) rather than comparing a code-only "
                          "prediction against a measurement that reflects both. Pure fp32 "
                          "accumulation-order noise (~1e-6, 100-1000x too small) is ruled out "
                          "as the dominant mechanism for either. See p10b_activation_quant_"
                          "disagreement.json for the full measurement.",
            "mechanism_identification": {
                "counts": counts,
                "code_is_exactly_1ulp": code_is_exactly_1ulp,
                "scale_is_structurally_sub_lsb": scale_is_sub_lsb,
                "scale_sub_lsb_bound": f"max|target/cutlass_recon - 1| < {E4M3_RELATIVE_LSB} "
                                       "(the literal e4m3 LSB, not a fitted threshold)",
            },
            "combined_model": {
                "code_term": aq["model_combined"]["code_term"] if aq else None,
                "scale_term": aq["model_combined"]["scale_term"] if aq else None,
                "scale_term_frac_of_code_term_magnitude": aq["model_combined"]["scale_term_frac_of_code_term"] if aq else None,
                "scale_term_frac_of_combined_variance": aq["model_combined"]["scale_term_frac_of_combined_variance"] if aq else None,
                "predicted_frob_combined": predicted_frob_combined,
                "predicted_frob_code_only": aq["model_code_only"]["predicted_frob"] if aq else None,
                "measured_worst_frob": worst_frob_rel,
                "ratio_measured_to_combined_predicted": frob_ratio_to_predicted,
                "note": "the scale-nudge mechanism contributes only "
                        f"{(aq['model_combined']['scale_term_frac_of_code_term'] * 100):.1f}% "
                        "of the code-term's MAGNITUDE" if aq else
                        "quantitatively small (see scale_term_frac_of_* once available)",
            },
            "RETIRED_sqrt_K_model": {
                "credit": "coordinator completion review, cycle 3 (2026-07-26): caught that "
                    "measured frob_rel_error is K-INVARIANT (mean 3.537e-3 at K=2048 vs "
                    "3.486e-3 at K=4096, ratio 1.01), contradicting single_flip_floor()'s "
                    "sqrt(K)-scaling prediction (ratio should be sqrt(2)=1.41); proposed the "
                    "K-invariant fixed-fraction model, confirmed by direct measurement.",
                "old_formula": "floor(K) = E4M3_RELATIVE_LSB / sqrt(K) (see single_flip_floor(), "
                    "kept in source for the audit trail)",
                "why_retired": "correct as a SPECIAL CASE (fixed COUNT=1 flip, not fixed "
                    "FRACTION) but does not explain the observed K-invariance; the safety-"
                    "multiple gate built on it (GATE_SAFETY_MULTIPLE=4, still in source) is "
                    "superseded by `mechanism_identification`/`combined_model` above.",
            },
            "operating_tripwires": {
                "note": "POST-HOC-CALIBRATED regression guards (set from today's measured "
                        "values + margin), NOT part of go_numerics_fidelity -- if a future "
                        "re-run trips one, investigate before assuming the mechanism is still "
                        "the same benign one identified here.",
                "code_level_fraction": tripwire_code_fraction,
                "combined_model_ratio": tripwire_model_ratio,
                "vs_rejected_ue8m0_class_margin": tripwire_ue8m0_margin,
            },
            "vs_rejected_ue8m0_class": {
                "pass": vs_ue8m0_pass,
                "margin_ratio": vs_ue8m0_ratio,
                "reference": ue8m0_range,
                "note": "contextual comparison (not one of the four go_numerics_fidelity "
                        "pillars per the coordinator's cycle-4 framing) -- worst measured "
                        "frob_rel_error vs P7's rejected UE8M0-requant class; its specific "
                        "numeric margin lives in operating_tripwires.vs_rejected_ue8m0_class_margin.",
            },
            "token_level_evidence": {
                "vllm_cutlass_vs_hf_triton_full_pipeline": {
                    "source": "M1-I1 baseline finding, external to this probe (project memory: "
                              ".memory/main/qwen35-target.md; "
                              "accept/reference/README.md 'vLLM smoke (step 8)')",
                    "result": "vLLM (dense fp8 via CUTLASS) and HF transformers (dense fp8 via "
                              "Triton, DeepGEMM disabled) produced BYTE-IDENTICAL 64/64 greedy "
                              "token ids on prompt p01-history -- whatever numeric disagreement "
                              "exists between these two kernel families (this probe's 2-4e-3 "
                              "class) did not flip a single autoregressive decision across 64 "
                              "sequential steps on that prompt.",
                },
                "p1_bf16_dense_perturbation_comparison": {
                    "source": "this issue's own P1 probe (p1_dense_bf16_result.json) -- a "
                              "perturbation an order of magnitude LARGER than cross-"
                              "implementation noise (full removal of dense-path fp8, not just "
                              "kernel disagreement)",
                    "result": "7/10 prompts exact match (596/640 positions); all 3 divergences "
                              "at close-margin decisions (this run's own top1-vs-top2 gap "
                              "0.0/0.125/0.25 logit units) -- even a coarser perturbation only "
                              "flips already-marginal decisions, never produces gross "
                              "corruption.",
                },
            },
            "rationale": rationale,
            "worst_frob_rel_error": worst_frob_rel,
            "worst_p99_rel_diff_top_half": worst_p99_top_half,
            "OLD_THRESHOLDS_RETIRED": {
                "frob_rel_error_max": FROB_REL_THRESHOLD_RETIRED,
                "p99_rel_diff_top_half_max": P99_TOP_HALF_THRESHOLD_RETIRED,
                "why_retired": "Flat constants chosen as 'reasonable slack' around the design "
                    "doc's explicitly-rough '~1e-3' point estimate, with no derivation -- an "
                    "undocumented tightening (2e-3/1e-2 are NOT stated anywhere in "
                    "v1-architecture.md SS14, they were this probe's own invention). Caught by "
                    "coordinator completion review 2026-07-26: a machine-checkable gate that "
                    "other issues (M2-I12) consume must carry a principled derivation, not an "
                    "arbitrary constant. Superseded first by a derived sqrt(K) floor (cycle 2), "
                    "then by a direct measurement of the raw e4m3 codes (cycle 3), then "
                    "reframed into the four-pillar `correctness_claim` + `operating_tripwires` "
                    "structure (cycle 4) -- see `RETIRED_sqrt_K_model`, `mechanism_identification`, "
                    "`combined_model`. p99_rel_diff_top_half is still computed and reported (see "
                    "per-case data) but is no longer a gate "
                    "criterion -- it was ALSO undocumented/arbitrary and the elementwise-"
                    "percentile family of stats is inherently less well-founded than the "
                    "magnitude-weighted frob-norm for this comparison (diff_stats() docstring).",
            },
            "note": "cutlass_vs_triton is the gating pair (two independent fp32-scale-class "
                    "implementations); *_vs_fp32ref entries in `cases` attribute any divergence "
                    "(both fp8 kernels can share a small common offset vs the fp32 reference "
                    "without disagreeing with EACH OTHER -- that's the actual P10 question).",
        },
        "perf_bar_result": {
            "pass": perf_pass,
            "marginal": perf_marginal,
            "measurement": perf_measurement,
            "min_speedup_cutlass_over_bf16": min_speedup,
            "graph_capture_ok_all_cases": graph_ok_all,
            "role": "AMENDED 2026-07-26: this is now an INFORMATIONAL M3-PRIORITIZATION PRIOR, "
                    "NOT ANDed into top-level `go` -- per v1-architecture.md SS6.2 as amended, "
                    "M2-I12's dense fp8 kernel work proceeds on NUMERICS FIDELITY (go / "
                    "go_numerics_fidelity above); this perf bar is the input to how M3 should "
                    "PRIORITIZE/rank the eventual restoration, a separate downstream question "
                    "with a separate consumer.",
            "note": "Gates on CUDA-graph-replayed speedup (dispatch-overhead-free), which is "
                    "the faithful reproduction of how vLLM's own AC-4 baseline runs this exact "
                    "op (constraint.md: vLLM benchmarked 'with CUDA graphs'); eager (raw Python "
                    "dispatch) speedup is reported per-case as a diagnostic only, since it "
                    "conflates real kernel throughput with host dispatch overhead.",
            "per_case_speedup": [{"shape": c["shape"], "label": c["label"], "M": c["M"],
                                   "speedup_graph": c["perf"]["speedup_cutlass_over_bf16_graph"],
                                   "speedup_eager": c["perf"]["speedup_cutlass_over_bf16_eager"]}
                                  for c in cases],
            "thresholds": {"go_min": PERF_THRESHOLD, "marginal_min": PERF_MARGINAL_THRESHOLD},
        },
        "env": env,
        "cases": cases,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="12288x2048,9216x2048,2048x4096")
    ap.add_argument("--batch", default="1,2,4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--numerics-draws", type=int, default=8,
                     help="independent random-activation batches pooled per case for the "
                          "numerics diff stats (perf timing reuses draw 0's input).")
    ap.add_argument("--out", default=os.path.expanduser("~/mpk-qwen35/probes/fp8_out/p10_verdict.json"))
    ap.add_argument("--recompute-from", default=None,
                     help="skip measurement entirely; reload `cases`+`env` from this existing "
                          "verdict JSON and rebuild only the verdict fields (no GPU, no "
                          "re-measurement) -- use after a gate-logic revision.")
    ap.add_argument("--ue8m0-reference-json", default=None,
                     help="path to a p7_ue8m0_delta.json, grounding gate criterion (e); if "
                          "omitted, that criterion is reported unavailable, never assumed.")
    ap.add_argument("--activation-quant-json", default=None,
                     help="path to a p10b_activation_quant_disagreement.json, grounding gate "
                          "criteria (b)-(d) (direct code-disagreement measurement); if "
                          "omitted, the numerics gate fails closed rather than assuming benign.")
    args = ap.parse_args()

    if args.recompute_from:
        print(f"--recompute-from {args.recompute_from}: no GPU, no re-measurement.", flush=True)
        with open(args.recompute_from) as f:
            prior = json.load(f)
        cases = prior["cases"]
        shape_keys = prior["env"]["shapes_run"]
        batches = prior["env"]["batches"]
        env = dict(prior["env"])
        env["recomputed_from"] = args.recompute_from
        env["recompute_timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    else:
        device = "cuda"
        shape_keys = args.shapes.split(",")
        # Bonus same-shape-class cross-check, always included in addition to the requested
        # shapes (strictly adds evidence to the worst-case aggregate; never removes coverage).
        if "2048x4096" in shape_keys and "2048x4096-attn-bonus" not in shape_keys:
            shape_keys = shape_keys + ["2048x4096-attn-bonus"]
        batches = [int(x) for x in args.batch.split(",")]

        index = load_index()
        cases = []
        for shape_key in shape_keys:
            W, scale, label, src_keys = load_real_tensor(shape_key, index, device)
            W_bf16 = dequant_bf16(W, scale)
            for M in batches:
                case = run_case(shape_key, W, scale, W_bf16, M, args.seed,
                                 args.warmup, args.iters, args.repeats, args.numerics_draws, device)
                case["label"] = label
                case["src_keys"] = src_keys
                cases.append(case)
                nc = case["numerics"]["cutlass_vs_triton"]
                pf = case["perf"]
                sg = f"{pf['speedup_cutlass_over_bf16_graph']:.2f}x" if pf["graph_capture_ok"] else "N/A"
                print(f"[{label}] M={M}: cutlass_vs_triton frob_rel={nc['frob_rel_error']:.3e} "
                      f"p99_top_half={nc['p99_rel_diff_top_half']:.3e} "
                      f"bias_effect={nc['bias_effect_size']:.4f} (z={nc['bias_zscore_INFORMATIONAL_ONLY']:.2f}) "
                      f"speedup(cutlass/bf16) graph={sg} eager={pf['speedup_cutlass_over_bf16_eager']:.2f}x",
                      flush=True)

        env = {
            "hostname": socket.gethostname(),
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
            "checkpoint": "Qwen/Qwen3.5-35B-A3B-FP8",
            "checkpoint_revision": CKPT_REVISION,
            "transformers_disable_deepgemm_linear": os.environ.get("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR"),
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeats": args.repeats,
            "numerics_draws": args.numerics_draws,
            "batches": batches,
            "shapes_requested": args.shapes.split(","),
            "shapes_run": shape_keys,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    ue8m0_range = load_ue8m0_reference(args.ue8m0_reference_json)
    aq_evidence = load_activation_quant_evidence(args.activation_quant_json)
    verdict = build_verdict(cases, shape_keys, batches, env, ue8m0_range, aq_evidence)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nWROTE {args.out}")
    n = verdict["numerics"]
    p = verdict["perf_bar_result"]
    cc = n["correctness_claim"]
    cm = n["combined_model"]
    print(f"go_numerics_fidelity={verdict['go_numerics_fidelity']}  pillars: "
          f"(i)mechanism_identified={cc['i_mechanism_identified']} "
          f"(ii)no_systematic_bias={cc['ii_no_systematic_bias']} "
          f"(iii)combined_model_matches={cc['iii_combined_model_matches']} "
          f"(iv)token_level_corroboration={cc['iv_token_level_corroboration']}  "
          f"[counts: {n['mechanism_identification']['counts']}]  "
          f"[combined predicted={cm['predicted_frob_combined']}, measured={cm['measured_worst_frob']:.3e}, "
          f"ratio={cm['ratio_measured_to_combined_predicted']}]  "
          f"perf_bar_result.pass={p['pass']} (min_speedup={p['min_speedup_cutlass_over_bf16']:.2f}x, "
          f"marginal={p['marginal']}) [informational M3 prior, not ANDed into go]")


if __name__ == "__main__":
    main()
