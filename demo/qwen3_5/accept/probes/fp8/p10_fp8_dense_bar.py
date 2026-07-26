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
    across M in {1,2,4,8,16} (10 cases, ~190K activation elements total).
      - Every code-level disagreement resolved to EXACTLY 1 ULP (verified via a +-8 ULP bit-
        level search on the raw e4m3 magnitude bits; 0/281 disagreements were >1 ULP or
        unexplained by either mechanism below).
      - Overall code-level disagreement fraction f = 211/190464 = 1.108e-3 (same order as,
        and within ~1.8x of, the value implied by inverting this formula against P10's own
        measured frob). Plugging f back in: predicted frob = 0.125*sqrt(1.108e-3) = 4.16e-3,
        vs P10's actual worst measured frob_rel_error 4.352e-3 -- a 1.05x match, decisively
        inside any reasonable consistency band.
      - A SECOND, smaller-magnitude, more numerous mechanism was found and is reported
        separately, not conflated with the LSB-level model above: an occasional per-128-
        element-GROUP scale discrepancy (~0.43%-0.77% relative, 8 discrete values observed,
        16-30x SMALLER than a full e4m3 LSB) between the two quantizers' absmax/448
        computation for that whole group (verified: CUTLASS's scale independently matches a
        from-scratch absmax/448 computation exactly in every group checked; only Triton's
        occasionally differs, uniformly across the whole group). This mechanism affects MORE
        elements (overall fraction 1801/190464 = 9.46e-3) but each contributes far less error,
        so its net contribution to frob is smaller than the LSB-level mechanism's, consistent
        with the observed P10 frob magnitude. It is one-directional AT THE ACTIVATION level
        (every observed ratio is Triton-scale-slightly-larger, never smaller) but this does
        NOT propagate to an output-level bias: real weights have mixed signs, so a uniformly-
        inflated activation contributes positive or negative error to the output depending on
        the weight sign it's multiplied against (consistent with P10's own already-passing,
        mixed-sign, near-zero bias_effect_size measurements). Full data:
        `p10b_activation_quant_disagreement.json`.
    """
    return E4M3_RELATIVE_LSB * math.sqrt(f)


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
    disagreement.py) grounding `fraction_model_frob()`. Returns None if absent -- reported as
    unavailable, never assumed or fabricated."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        total_code = sum(c["n_code_ulp_disagreement"] for c in d["cases_tested"])
        total_gs = sum(c["n_group_scale_disagreement"] for c in d["cases_tested"])
        total_unexplained = sum(c["n_unexplained"] for c in d["cases_tested"])
        total_elements = sum(c["n_total_elements"] for c in d["cases_tested"])
        f_code = total_code / total_elements
        return {
            "source": path,
            "n_cases_tested": len(d["cases_tested"]),
            "max_ulp_distance_overall": d["max_ulp_distance_overall"],
            "all_code_level_disagreements_are_1_ulp": d["all_code_level_disagreements_are_1_ulp"],
            "total_unexplained_positions": total_unexplained,
            "code_level_disagreement_fraction_f": f_code,
            "group_scale_disagreement_fraction": total_gs / total_elements,
            "group_scale_ratio_range": d["second_mechanism_found"][
                "observed_ratios (target/cutlass_recon, should be close to 1.0)"],
            "predicted_frob_from_f_code": fraction_model_frob(f_code),
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

    # --- Gate criterion (a): no systematic bias -- the actual "something is wrong" alarm ---
    no_systematic_bias = worst_bias_effect <= BIAS_EFFECT_SIZE_THRESHOLD

    # --- Gate criterion (b): frob-rel within GATE_SAFETY_MULTIPLE x the derived floor ---
    derived_gate_pass = all(c["numerics"]["cutlass_vs_triton"]["derived_gate_pass"] for c in cases)

    # --- Gate criterion (c): strictly << the rejected UE8M0-requant class (P7) ---
    if ue8m0_range and "min" in ue8m0_range:
        # "strictly much smaller" = at least 2x below the SMALLEST observed UE8M0 delta,
        # i.e. the whole numerics gate (worst case) must clear this with room to spare.
        vs_ue8m0_pass = worst_frob_rel <= 0.5 * ue8m0_range["min"]
        vs_ue8m0_ratio = ue8m0_range["min"] / worst_frob_rel
    else:
        vs_ue8m0_pass, vs_ue8m0_ratio = None, None  # unavailable, not assumed

    # --- Gate criteria (d)+(e): decisive DIRECT measurement of the raw e4m3 codes both
    # quantizers produce (coordinator review cycle 3) -- supersedes the K-dependent
    # single_flip_floor model, which the data itself contradicted (K-invariant, not
    # sqrt(K)-scaling: mean frob 3.537e-3 at K=2048 vs 3.486e-3 at K=4096, ratio 1.01, not the
    # predicted 1.41). See `fraction_model_frob()` for the confirmed replacement model.
    aq = activation_quant_evidence
    if aq and "code_level_disagreement_fraction_f" in aq:
        f_code = aq["code_level_disagreement_fraction_f"]
        predicted_frob = fraction_model_frob(f_code)
        boundary_fraction_pass = f_code <= 2e-3
        max_ulp_pass = aq.get("max_ulp_distance_overall") == 1 and aq.get("all_code_level_disagreements_are_1_ulp") is True
        frob_ratio_to_predicted = worst_frob_rel / predicted_frob
        frob_consistency_pass = frob_ratio_to_predicted <= 2.0
        unexplained_pass = aq.get("total_unexplained_positions") == 0
        decisive_measurement_pass = bool(boundary_fraction_pass and max_ulp_pass
                                          and frob_consistency_pass and unexplained_pass)
    else:
        f_code = predicted_frob = frob_ratio_to_predicted = None
        boundary_fraction_pass = max_ulp_pass = frob_consistency_pass = unexplained_pass = None
        decisive_measurement_pass = False  # unavailable -> fail closed, never assumed benign

    gate_pass = bool(no_systematic_bias and decisive_measurement_pass and (vs_ue8m0_pass is not False))
    go_numerics_fidelity = gate_pass

    perf_pass = min_speedup >= PERF_THRESHOLD
    perf_marginal = (not perf_pass) and min_speedup >= PERF_MARGINAL_THRESHOLD

    rationale = (
        f"gate_pass={gate_pass}: (a) no_systematic_bias={no_systematic_bias} "
        f"(worst bias effect size {worst_bias_effect:.4f} <= {BIAS_EFFECT_SIZE_THRESHOLD} bar -- "
        f"this is the real corruption alarm: a directional/systematic offset would indicate a "
        f"scale-application or indexing bug; a small, UNBIASED, symmetric spread is the "
        f"signature of two valid-but-different roundings, not corruption). "
        f"(b) measured_boundary_fraction<=2e-3: {boundary_fraction_pass} (direct measurement, "
        f"not inference: code-level disagreement fraction f={f_code:.3e}). "
        f"(c) max_ulp_of_disagreement==1: {max_ulp_pass}. "
        f"(d) frob consistent with 0.125*sqrt(f) within 2x: {frob_consistency_pass} "
        f"(predicted {predicted_frob:.3e} vs measured worst {worst_frob_rel:.3e}, ratio "
        f"{frob_ratio_to_predicted:.2f}x)."
        if aq else "gate_pass=False: decisive activation-quant-disagreement measurement "
                   "UNAVAILABLE (no p10b_activation_quant_disagreement.json found) -- fails "
                   "closed rather than assuming benign."
    ) + (
        f" (e) vs_rejected_ue8m0_class="
        + (f"{vs_ue8m0_pass} (measured worst case is {vs_ue8m0_ratio:.1f}x smaller than P7's "
           f"smallest UE8M0-requant delta)" if vs_ue8m0_ratio is not None else "UNAVAILABLE "
           "(no --ue8m0-reference-json / p7_ue8m0_delta.json found -- not assumed)."
           ) + " Corroborated by independent token-level evidence (see "
        f"numerics.token_level_evidence): the M1 baseline found vLLM(CUTLASS)-vs-HF(Triton) "
        f"produced byte-identical 64/64 tokens on prompt p01, and this issue's own P1 probe "
        f"found a perturbation an order of magnitude LARGER than this (full bf16 substitution, "
        f"not just cross-implementation noise) only flipped 3/10 prompts at close margins."
    )

    return {
        "go": go_numerics_fidelity,  # ALIAS: repurposed 2026-07-26 to mean go_numerics_fidelity
                                       # only (see module docstring) -- perf is a separate prior.
        "go_numerics_fidelity": go_numerics_fidelity,
        "numerics": {
            "gate_pass": gate_pass,
            "no_systematic_bias": no_systematic_bias,
            "worst_bias_effect_size": worst_bias_effect,
            "bias_effect_size_threshold": BIAS_EFFECT_SIZE_THRESHOLD,
            "frob_class": "TWO confirmed mechanisms (coordinator review cycle 3, decisive "
                          "direct measurement, not inference): (1) occasional single e4m3-code "
                          "disagreements between the two quantizers, exactly 1 ULP, fraction "
                          "f_code (K-invariant model: relative_frob ~= 0.125*sqrt(f_code)); "
                          "(2) a smaller-per-element, more numerous per-128-group SCALE "
                          "discrepancy (~0.4-0.8%, 16-30x below one e4m3 LSB) between the two "
                          "quantizers' absmax computation for that group. Pure fp32 "
                          "accumulation-order noise (~1e-6, 100-1000x too small) is ruled out "
                          "as the dominant mechanism for either. See fraction_model_frob() "
                          "docstring and p10b_activation_quant_disagreement.json for the full "
                          "measurement.",
            "decisive_measurement": {
                "pass": decisive_measurement_pass,
                "source": aq.get("source") if aq else None,
                "code_level_disagreement_fraction_f": f_code,
                "boundary_fraction_le_2e3": boundary_fraction_pass,
                "max_ulp_distance_overall": aq.get("max_ulp_distance_overall") if aq else None,
                "max_ulp_eq_1": max_ulp_pass,
                "predicted_frob_from_f": predicted_frob,
                "measured_worst_frob": worst_frob_rel,
                "ratio_measured_to_predicted": frob_ratio_to_predicted,
                "frob_consistent_within_2x": frob_consistency_pass,
                "total_unexplained_positions": aq.get("total_unexplained_positions") if aq else None,
                "unexplained_eq_0": unexplained_pass,
                "second_mechanism_group_scale_fraction": aq.get("group_scale_disagreement_fraction") if aq else None,
                "second_mechanism_ratio_range": aq.get("group_scale_ratio_range") if aq else None,
            },
            "RETIRED_sqrt_K_model": {
                "credit": "coordinator completion review, cycle 3 (2026-07-26): caught that "
                    "measured frob_rel_error is K-INVARIANT (mean 3.537e-3 at K=2048 vs "
                    "3.486e-3 at K=4096, ratio 1.01), contradicting single_flip_floor()'s "
                    "sqrt(K)-scaling prediction (ratio should be sqrt(2)=1.41); proposed the "
                    "K-invariant fixed-fraction model, confirmed by direct measurement above.",
                "old_formula": "floor(K) = E4M3_RELATIVE_LSB / sqrt(K) (see single_flip_floor(), "
                    "kept in source for the audit trail)",
                "why_retired": "correct as a SPECIAL CASE (fixed COUNT=1 flip, not fixed "
                    "FRACTION) but does not explain the observed K-invariance; the safety-"
                    "multiple gate built on it (GATE_SAFETY_MULTIPLE=4, still in source) is "
                    "superseded by `decisive_measurement` above.",
            },
            "vs_rejected_ue8m0_class": {
                "pass": vs_ue8m0_pass,
                "margin_ratio": vs_ue8m0_ratio,
                "reference": ue8m0_range,
                "note": "criterion (c): worst measured frob_rel_error must be <= 0.5x the "
                        "SMALLEST P7 UE8M0-requant delta (a plain 'strictly much smaller' "
                        "check); actual margin is typically larger.",
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
                    "then by `decisive_measurement` above (cycle 3, direct measurement of the "
                    "raw e4m3 codes, not inference) -- see `RETIRED_sqrt_K_model`. "
                    "p99_rel_diff_top_half is still computed and reported (see per-case data) "
                    "but is no longer a gate "
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
    dm = n["decisive_measurement"]
    print(f"go_numerics_fidelity={verdict['go_numerics_fidelity']}  "
          f"(no_systematic_bias={n['no_systematic_bias']}, "
          f"decisive_measurement.pass={dm['pass']}, "
          f"f_code={dm['code_level_disagreement_fraction_f']}, "
          f"worst_frob={n['worst_frob_rel_error']:.3e}, "
          f"ratio_to_predicted={dm['ratio_measured_to_predicted']})  "
          f"perf_bar_result.pass={p['pass']} (min_speedup={p['min_speedup_cutlass_over_bf16']:.2f}x, "
          f"marginal={p['marginal']}) [informational M3 prior, not ANDed into go]")


if __name__ == "__main__":
    main()
