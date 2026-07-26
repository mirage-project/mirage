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
  {go: bool, numerics: {...}, perf: {...}, env: {...}, cases: [...]}
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
# Numerics gate thresholds. Primary gate = frob_rel_error (magnitude-weighted relative L2
# norm, standard for fp8 GEMM validation) + bias_effect_size (spec's explicit "no systematic
# bias" wording, measured as an n-invariant mean/std ratio -- NOT a z-score, see diff_stats
# docstring). p99_rel_diff_top_half is reported but not gated (why a raw elementwise
# max/floored-max is not a fair gate for e4m3 outputs, also in diff_stats' docstring).
FROB_REL_THRESHOLD = 2e-3
P99_TOP_HALF_THRESHOLD = 1e-2
BIAS_EFFECT_SIZE_THRESHOLD = 0.1  # |mean| < 10% of one std of the cutlass-vs-triton spread
PERF_THRESHOLD = 1.5
PERF_MARGINAL_THRESHOLD = 1.2

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
    args = ap.parse_args()

    device = "cuda"
    shape_keys = args.shapes.split(",")
    # Bonus same-shape-class cross-check, always included in addition to the requested shapes
    # (strictly adds evidence to the worst-case aggregate; never removes coverage).
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

    frob_rels = [c["numerics"]["cutlass_vs_triton"]["frob_rel_error"] for c in cases]
    p99_top_halfs = [c["numerics"]["cutlass_vs_triton"]["p99_rel_diff_top_half"] for c in cases]
    bias_effects = [c["numerics"]["cutlass_vs_triton"]["bias_effect_size"] for c in cases]
    graph_ok_all = all(c["perf"]["graph_capture_ok"] for c in cases)
    speedups_graph = [c["perf"]["speedup_cutlass_over_bf16_graph"] for c in cases if c["perf"]["graph_capture_ok"]]
    speedups_eager = [c["perf"]["speedup_cutlass_over_bf16_eager"] for c in cases]

    worst_frob_rel = max(frob_rels)
    worst_p99_top_half = max(p99_top_halfs)
    worst_bias_effect = max(abs(e) for e in bias_effects if math.isfinite(e))
    # Perf gate uses graph-replayed speedup (matches how vLLM's AC-4 baseline actually runs
    # this op, see timed_graph() docstring); falls back to eager only if graph capture failed
    # for every case (reported loudly via `perf_measurement` rather than silently swapped).
    if speedups_graph:
        min_speedup = min(speedups_graph)
        perf_measurement = "cuda_graph"
    else:
        min_speedup = min(speedups_eager)
        perf_measurement = "eager_FALLBACK_graph_capture_unavailable"

    numerics_pass = (worst_frob_rel <= FROB_REL_THRESHOLD
                      and worst_p99_top_half <= P99_TOP_HALF_THRESHOLD
                      and worst_bias_effect <= BIAS_EFFECT_SIZE_THRESHOLD)
    perf_pass = min_speedup >= PERF_THRESHOLD
    perf_marginal = (not perf_pass) and min_speedup >= PERF_MARGINAL_THRESHOLD
    go = bool(numerics_pass and perf_pass)

    verdict = {
        "go": go,
        "numerics": {
            "pass": numerics_pass,
            "worst_frob_rel_error": worst_frob_rel,
            "worst_p99_rel_diff_top_half": worst_p99_top_half,
            "worst_bias_effect_size": worst_bias_effect,
            "thresholds": {
                "frob_rel_error_max": FROB_REL_THRESHOLD,
                "p99_rel_diff_top_half_max": P99_TOP_HALF_THRESHOLD,
                "bias_effect_size_max": BIAS_EFFECT_SIZE_THRESHOLD,
            },
            "note": "cutlass_vs_triton is the gating pair (two independent fp32-scale-class "
                    "implementations); *_vs_fp32ref entries in `cases` attribute any divergence "
                    "(both fp8 kernels can share a small common offset vs the fp32 reference "
                    "without disagreeing with EACH OTHER -- that's the actual P10 question). "
                    "Gate uses frob_rel_error (magnitude-weighted) + bias_effect_size "
                    "(n-invariant mean/std ratio), NOT a raw elementwise max and NOT a z-score "
                    "-- see diff_stats() docstring for both root-caused reasons (e4m3's ~12.5% "
                    "per-element LSB inflates relative error on sub-RMS elements with no "
                    "systematic disagreement; z-score grows with sqrt(n) at fixed effect size "
                    "so it isn't a fair gate once --numerics-draws pools many samples). "
                    "floored_max_rel and bias_zscore are kept per-case as diagnostics only.",
        },
        "perf": {
            "pass": perf_pass,
            "marginal": perf_marginal,
            "measurement": perf_measurement,
            "min_speedup_cutlass_over_bf16": min_speedup,
            "graph_capture_ok_all_cases": graph_ok_all,
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
        "env": {
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
        },
        "cases": cases,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nWROTE {args.out}")
    print(f"GO={go}  numerics_pass={numerics_pass} (frob_rel={worst_frob_rel:.3e}, "
          f"p99_top_half={worst_p99_top_half:.3e}, bias_effect={worst_bias_effect:.4f})  "
          f"perf_pass={perf_pass} (min_speedup={min_speedup:.2f}x, marginal={perf_marginal})")


if __name__ == "__main__":
    main()
