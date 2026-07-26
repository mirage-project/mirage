#!/usr/bin/env python3
"""P2 -- the fail-closed MoE-scale gate (M2-I13, v1-architecture.md 6.2 + 14).

QUESTION. MPK's routed-expert path keeps the checkpoint's 128x128 block scales
in the builder (`repeat_interleave(128)`, deepseek_v3/builder.py:983) but the
grouped GEMM converts them to UE8M0 *inside the kernel*
(fp8_group_gemm_sm100.cuh warp 6, `ue8m0 = (__float_as_uint(sf) >> 23) & 0xFF`)
-- the same exponent-only scale family both reference engines refuse on the
dense path (vllm-graph.md 3.5, mpk-gaps.md 2.2.1). Does that conversion move
OUR checkpoint's expert numerics enough to threaten AC-3?

MECHANISM UNDER TEST, stated before measuring (`prediction` in the emitted
JSON). `(bits >> 23) & 0xFF` keeps only the float32 exponent field, so for a
scale s = m * 2^e with mantissa m in [1,2) the kernel applies s/m: an exact
power-of-two TRUNCATION TOWARD ZERO, never rounding. Both operands are
truncated, so every output element is multiplied by 1/(m_a * m_b) in (0.25, 1].
Predictions: (i) a per-row multiplicative SHRINK with slope ~ E[1/m]^2 ~ 0.52
if checkpoint mantissas are log-uniform; (ii) frob-rel vs preserved-scale
semantics of order 0.3-0.6, i.e. ~100x the 2-4e-3 preserved-scale class P10
measured; (iii) the raw mean/std effect size stays SMALL, because a
multiplicative shrink of a ~zero-mean output produces a ~zero-mean residual --
so the n-invariant statistic that actually detects this mechanism is the
per-row projection slope, not the residual mean; (iv) a torch model that
truncates both scales the same way reproduces the kernel to the bf16
output-rounding floor.

METHOD (P10's pillar-gate framing, mpk-gaps.md 2.2.1 shapes):
  * REAL layer-0 routed-expert tensors, [256,1024,2048] w13 / [256,2048,512]
    w2, with the checkpoint's own `weight_scale_inv` (bf16 in the file, widened
    with .float() exactly as MPK's loader does).
  * REAL activations and REAL routing from the HF oracle
    (demo/qwen3_5/oracle: moe0.layer_input, moe0.topk_ids,
    moe0.topk_renorm_weights, moe0.routed_expert_output).
  * Four semantics on identical fp8 bytes: the shipped grouped kernel
    (internal UE8M0), the fp32-block-scale grouped kernel (this issue's
    fallback), an fp32 torch anchor with the checkpoint scales preserved, and
    an fp32 torch anchor with both scales exponent-truncated (the mechanism
    model).
  * Gap 7: w2's K=512 gives fp8_k_tile_count=4, below anything MPK has run
    (task_register.cc:2812-2818). Every grouped-kernel launch is a separate
    stage so the runner can wrap it in `timeout`; the w2 stage runs both
    num_ab_stages=4 and 8.

STAGES (each a separate process; the runner puts a timeout on each):
    --stage w13       w13 GEMM, all semantics; caches tensors for later stages
    --stage w2        w2 GEMM, all semantics, num_ab_stages in {8,4}
    --stage token     token-level projection through the MoE block boundary
    --stage assemble  merge stage JSONs -> p2_verdict.json
"""
import argparse
import datetime
import json
import math
import os
import socket
import sys

import torch

# fp32 references must be true IEEE fp32, not TF32 on tensor cores.
torch.backends.cuda.matmul.allow_tf32 = False

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10
NUM_EXPERTS = 256
NUM_TOPK = 8
BATCH = 16  # the wrapper's compiled BATCH_SIZE
W13_N, W13_K = 1024, 2048
W2_N, W2_K = 2048, 512

CKPT_REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
SNAPSHOT = os.environ.get(
    "QWEN35_SNAPSHOT",
    os.path.expanduser(
        "~/mpk-qwen35/hf/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/"
        + CKPT_REVISION
    ),
)
ORACLE = os.environ.get(
    "QWEN35_ORACLE_DUMPS",
    os.path.expanduser("~/mpk-qwen35/oracle-work/dumps"),
)

PREDICTION = {
    "mechanism": "fp8_group_gemm_sm100.cuh warp 6 keeps only the float32 "
    "exponent field of BOTH scale operands: applied = 2^floor(log2(s)), a "
    "power-of-two truncation toward zero (never rounding).",
    "predicted_per_element_factor_range": [0.25, 1.0],
    "predicted_row_slope_if_mantissas_log_uniform": 0.5204,
    "predicted_frob_rel_vs_fp32_scale_order": "3e-1 to 6e-1",
    "preserved_scale_class_from_p10": "2e-3 to 4.4e-3",
    "predicted_raw_bias_effect_size": "SMALL (<0.1): a multiplicative shrink "
    "of a zero-mean output leaves a near-zero-mean residual -- this is the "
    "methodology trap; the detector is the per-row projection slope.",
    "predicted_mechanism_residual_frob_rel": "~1e-3 (bf16 output rounding + "
    "accumulation order) if the truncation model is the whole story",
    "predicted_w2_4ktile_hang": "no hang: the 2026-04-22 ab_empty fix removed "
    "the wrap-around race, and num_ab_stages=8 > k_tile_count=4",
}


# ----------------------------------------------------------------- utilities
def trunc_pow2(s):
    """The kernel's internal UE8M0 conversion, in torch.

    ue8m0 = (__float_as_uint(sf) >> 23) & 0xFF ; applied = 2^(ue8m0 - 127).
    """
    bits = s.float().contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    return torch.exp2(exp.float() - 127.0)


def quantize_activation(x_bf16):
    """vLLM/HF QuantFP8, fp32 scales (vllm-graph.md 3.4, v1-architecture 6.1):
    absmax = max(max|x|, 1e-10); scale = absmax/448; x/scale (division);
    clamp to +-448 BEFORE the RN-even e4m3 cast.

    `absmax / 448.0` with a PYTHON scalar is lowered by PyTorch to a reciprocal
    multiply and lands 1 ULP away from MPK's quantize task, which divides
    (measured in tests/runtime_python/blackwell/sm100_fp8_moe_qwen35/
    test_quantize_fp8_f32scale_moe.py). Divide by a 0-dim tensor so these
    activation bytes are exactly the ones the runtime would hand the GEMM.
    """
    shape = x_bf16.shape
    k = shape[-1]
    xf = x_bf16.float().reshape(-1, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = torch.div(
        absmax, torch.tensor(FP8_MAX, dtype=torch.float32, device=absmax.device)
    )
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return (
        q.reshape(shape).to(torch.float8_e4m3fn).contiguous(),
        scale.reshape(*shape[:-1], k // BLOCK).float().contiguous(),
    )


def dequant_groups(q, scale):
    """x_fp8 * per-128-group scale."""
    shape = q.shape
    k = shape[-1]
    return (
        q.float().reshape(-1, k // BLOCK, BLOCK) * scale.reshape(-1, k // BLOCK, 1)
    ).reshape(shape)


def dequant_blocks(q, block_scale):
    """w_fp8 * weight_scale_inv, expanded from [N/128, K/128] to [N, K]."""
    n, k = q.shape
    s = block_scale.repeat_interleave(BLOCK, dim=0)[:n]
    s = s.repeat_interleave(BLOCK, dim=1)[:, :k]
    return q.float() * s


def frob_rel(actual, ref):
    return (actual - ref).norm().item() / max(ref.norm().item(), 1e-30)


def row_slopes(actual, ref):
    """Per-row least-squares projection <a,r>/<r,r>.

    n-INVARIANT by construction (a ratio of two sums over the same row), and it
    is the statistic a MULTIPLICATIVE mechanism moves. slope == 1 means "no
    gain error"; slope < 1 means the row was systematically shrunk.
    """
    a = actual.reshape(actual.shape[0], -1).double()
    r = ref.reshape(ref.shape[0], -1).double()
    num = (a * r).sum(dim=1)
    den = (r * r).sum(dim=1)
    keep = den > 0
    return (num[keep] / den[keep]).float()


def bias_stats(actual, ref):
    """P10's n-invariant residual statistic mean/std, plus its z diagnostic."""
    diff = (actual - ref).flatten().double()
    n = diff.numel()
    std = diff.std(unbiased=True).item()
    mean = diff.mean().item()
    if std <= 1e-30:
        return (0.0 if abs(mean) < 1e-30 else float("inf")), 0.0
    return mean / std, mean / (std / math.sqrt(n))


def compare(actual, ref, ref_name):
    """The full metric block for one (semantics, reference) pair."""
    a = actual.float()
    r = ref.float()
    sl = row_slopes(a, r)
    effect, z = bias_stats(a, r)
    rel_gain_err = (1.0 - sl).double()
    return {
        "vs": ref_name,
        "frob_rel": frob_rel(a, r),
        "row_slope_mean": sl.mean().item(),
        "row_slope_min": sl.min().item(),
        "row_slope_max": sl.max().item(),
        "row_slope_std": sl.std(unbiased=True).item() if sl.numel() > 1 else 0.0,
        # n-invariant effect size of the MULTIPLICATIVE mechanism: how many
        # row-to-row standard deviations the mean gain error sits away from 0.
        "gain_error_mean": rel_gain_err.mean().item(),
        "gain_error_effect_size": (
            rel_gain_err.mean().item()
            / max(rel_gain_err.std(unbiased=True).item(), 1e-30)
            if sl.numel() > 1
            else float("inf")
        ),
        # P10's raw-residual statistic, kept to show it is the WRONG detector
        # for a multiplicative mechanism (see the JSON's methodology note).
        "raw_bias_effect_size": effect,
        "raw_bias_zscore": z,
        "mean_abs_ratio": (a.abs().mean() / r.abs().mean().clamp(min=1e-30)).item(),
        "n_elements": a.numel(),
    }


# --------------------------------------------------------------- oracle load
def load_oracle(mode="prefill"):
    d = os.path.join(ORACLE, mode, "tensors")
    def g(name):
        return torch.load(os.path.join(d, name + ".pt"), map_location="cpu")
    return {
        "layer_input": g("moe0.layer_input"),
        "topk_ids": g("moe0.topk_ids"),
        "topk_w": g("moe0.topk_renorm_weights"),
        "routed_expert_output": g("moe0.routed_expert_output"),
    }


def load_expert_weights(expert_ids, device):
    """Real layer-0 routed-expert tensors at the shapes MPK's grouped GEMM
    consumes: w13 [E,1024,2048] (gate||up) and w2 [E,2048,512], plus the
    checkpoint's own block scales [E,8,16] / [E,16,4].

    Only the routed experts are materialised; the rest of the [256, ...]
    tensor is never read by the kernel (the expert mask lists exactly the
    activated ids) and stays zero.
    """
    import json as _json
    from safetensors import safe_open

    with open(os.path.join(SNAPSHOT, "model.safetensors.index.json")) as f:
        index = _json.load(f)
    shards = {}

    def get(key):
        path = os.path.join(SNAPSHOT, index["weight_map"][key])
        if path not in shards:
            shards[path] = safe_open(path, framework="pt")
        return shards[path].get_tensor(key)

    p = "model.language_model.layers.0.mlp.experts."
    w13 = torch.zeros((NUM_EXPERTS, W13_N, W13_K), dtype=torch.float8_e4m3fn)
    w13_s = torch.zeros((NUM_EXPERTS, W13_N // BLOCK, W13_K // BLOCK))
    w2 = torch.zeros((NUM_EXPERTS, W2_N, W2_K), dtype=torch.float8_e4m3fn)
    w2_s = torch.zeros((NUM_EXPERTS, W2_N // BLOCK, W2_K // BLOCK))

    for e in expert_ids:
        gate = get(f"{p}{e}.gate_proj.weight")
        up = get(f"{p}{e}.up_proj.weight")
        down = get(f"{p}{e}.down_proj.weight")
        gate_s = get(f"{p}{e}.gate_proj.weight_scale_inv")
        up_s = get(f"{p}{e}.up_proj.weight_scale_inv")
        down_s = get(f"{p}{e}.down_proj.weight_scale_inv")
        assert gate.dtype == torch.float8_e4m3fn, gate.dtype
        # The checkpoint stores weight_scale_inv in BF16; MPK's loader widens
        # with .float() and nothing else (mpk-gaps.md 2.2.1).
        assert gate_s.dtype == torch.bfloat16, gate_s.dtype
        w13[e] = torch.cat([gate, up], dim=0)
        w13_s[e] = torch.cat([gate_s, up_s], dim=0).float()
        w2[e] = down
        w2_s[e] = down_s.float()

    return (
        w13.to(device),
        w13_s.to(device),
        w2.to(device),
        w2_s.to(device),
    )


def build_routing(topk_ids, device):
    """routing[e, t] = topk slot + 1 (0 = not routed); mask lists the
    activated experts and mask[NUM_EXPERTS] holds their count -- the exact
    convention fp8_group_gemm_sm100.cuh reads."""
    tokens = topk_ids.shape[0]
    routing = torch.zeros((NUM_EXPERTS, BATCH), dtype=torch.int32)
    for t in range(tokens):
        for slot in range(NUM_TOPK):
            routing[int(topk_ids[t, slot]), t] = slot + 1
    activated = sorted({int(x) for x in topk_ids.flatten().tolist()})
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32)
    for i, e in enumerate(activated):
        mask[i] = e
    mask[NUM_EXPERTS] = len(activated)
    return routing.to(device), mask.to(device), activated


def expand_row_scales(block_scale, n):
    """Builder form for the UE8M0 grouped kernel: [E, N/128, K/128] ->
    [E*N, K/128] float32 (deepseek_v3/builder.py:983)."""
    e, nb, ks = block_scale.shape
    return (
        block_scale.repeat_interleave(BLOCK, dim=1)[:, :n]
        .reshape(e * n, ks)
        .contiguous()
    )


# ------------------------------------------------------------ torch semantics
def torch_grouped(x_q, x_s, w_q, w_s, topk_ids, tokens, out_n, truncate):
    """Grouped GEMM in torch under one scale semantics.

    x_q/x_s are [tokens, K] (w13) or [tokens, topk, K] (w2). Returns
    [tokens, topk, out_n] float32.
    """
    per_slot = x_q.dim() == 3
    out = torch.zeros(
        (tokens, NUM_TOPK, out_n), dtype=torch.float32, device=x_q.device
    )
    for t in range(tokens):
        for slot in range(NUM_TOPK):
            e = int(topk_ids[t, slot])
            xs = x_s[t, slot] if per_slot else x_s[t]
            xq = x_q[t, slot] if per_slot else x_q[t]
            ws = w_s[e]
            if truncate:
                xs = trunc_pow2(xs)
                ws = trunc_pow2(ws)
            xd = dequant_groups(xq.unsqueeze(0), xs.unsqueeze(0))
            wd = dequant_blocks(w_q[e], ws)
            out[t, slot] = (xd @ wd.t()).squeeze(0)
    return out


def torch_grouped_bf16(x_q, x_s, w_q, w_s, topk_ids, tokens, out_n):
    """The bf16-dequant reference: same fp8 bytes, block-dequantized to bf16,
    contracted as a bf16 GEMM -- what an MPK bf16 expert path would compute."""
    per_slot = x_q.dim() == 3
    out = torch.zeros(
        (tokens, NUM_TOPK, out_n), dtype=torch.float32, device=x_q.device
    )
    for t in range(tokens):
        for slot in range(NUM_TOPK):
            e = int(topk_ids[t, slot])
            xs = x_s[t, slot] if per_slot else x_s[t]
            xq = x_q[t, slot] if per_slot else x_q[t]
            xd = dequant_groups(xq.unsqueeze(0), xs.unsqueeze(0)).to(torch.bfloat16)
            wd = dequant_blocks(w_q[e], w_s[e]).to(torch.bfloat16)
            out[t, slot] = (xd @ wd.t()).float().squeeze(0)
    return out


def silu_mul(w13_out):
    """SwiGLU on the grouped w13 output: gate = [..., :512], up = [..., 512:]."""
    half = w13_out.shape[-1] // 2
    gate = w13_out[..., :half]
    up = w13_out[..., half:]
    return torch.nn.functional.silu(gate) * up


# ------------------------------------------------------------------- runtime
def import_kernel():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "../../../../.."))
    ext = os.path.join(
        repo, "tests/runtime_python/blackwell/sm100_fp8_moe_qwen35"
    )
    if ext not in sys.path:
        sys.path.insert(0, ext)
    import runtime_kernel_blackwell_fp8_moe_qwen35 as k

    return k


def env_block():
    return {
        "hostname": socket.gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch": torch.__version__,
        "checkpoint": "Qwen/Qwen3.5-35B-A3B-FP8",
        "checkpoint_revision": CKPT_REVISION,
        "oracle_dumps": ORACLE,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------- stages
def stage_w13(args):
    dev = "cuda"
    k = import_kernel()
    orc = load_oracle()
    tokens = orc["layer_input"].shape[0]
    topk_ids = orc["topk_ids"]
    routing, mask, activated = build_routing(topk_ids, dev)
    w13, w13_s, w2, w2_s = load_expert_weights(activated, dev)

    x = torch.zeros((BATCH, W13_K), dtype=torch.bfloat16, device=dev)
    x[:tokens] = orc["layer_input"].to(dev).to(torch.bfloat16)
    x_q, x_s = quantize_activation(x)

    w13_row_s = expand_row_scales(w13_s, W13_N)

    out_ue8m0 = torch.zeros(
        (BATCH, NUM_TOPK, W13_N), dtype=torch.bfloat16, device=dev
    )
    k.moe_w13_ue8m0_sm100(
        x_q, x_s, w13, w13_row_s, routing, mask, out_ue8m0, args.num_ab_stages
    )
    out_fallback = torch.zeros_like(out_ue8m0)
    if not args.no_fallback:
        k.moe_w13_blockscale_sm100(
            x_q, x_s, w13, w13_s.contiguous(), routing, mask, out_fallback
        )

    ref_fp32 = torch_grouped(
        x_q[:tokens], x_s[:tokens], w13, w13_s, topk_ids, tokens, W13_N, False
    )
    ref_trunc = torch_grouped(
        x_q[:tokens], x_s[:tokens], w13, w13_s, topk_ids, tokens, W13_N, True
    )
    ref_bf16 = torch_grouped_bf16(
        x_q[:tokens], x_s[:tokens], w13, w13_s, topk_ids, tokens, W13_N
    )

    ker_u = out_ue8m0[:tokens].float()
    ker_f = out_fallback[:tokens].float()
    res = {
        "stage": "w13",
        "shape": {"E": NUM_EXPERTS, "N": W13_N, "K": W13_K, "k_tiles": W13_K // BLOCK},
        "tokens": tokens,
        "activated_experts": len(activated),
        "num_ab_stages": args.num_ab_stages,
        "ue8m0_kernel": {
            "vs_fp32_scale": compare(ker_u, ref_fp32, "fp32_scale_torch"),
            "vs_bf16_dequant": compare(ker_u, ref_bf16, "bf16_dequant_torch"),
            "vs_truncation_model": compare(ker_u, ref_trunc, "ue8m0_trunc_model"),
        },
        "truncation_model": {
            "vs_fp32_scale": compare(ref_trunc, ref_fp32, "fp32_scale_torch"),
        },
        "bf16_dequant_reference": {
            "vs_fp32_scale": compare(ref_bf16, ref_fp32, "fp32_scale_torch"),
        },
        "env": env_block(),
    }
    if not args.no_fallback:
        res["blockscale_kernel"] = {
            "vs_fp32_scale": compare(ker_f, ref_fp32, "fp32_scale_torch"),
            "vs_bf16_dequant": compare(ker_f, ref_bf16, "bf16_dequant_torch"),
        }

    torch.save(
        {
            "x_q": x_q.cpu(),
            "x_s": x_s.cpu(),
            "w13_out_ue8m0": out_ue8m0[:tokens].cpu(),
            "w13_out_fallback": out_fallback[:tokens].cpu(),
            "w13_out_fp32ref": ref_fp32.cpu(),
            "w13_out_bf16ref": ref_bf16.cpu(),
            "w13_out_truncref": ref_trunc.cpu(),
            "activated": activated,
        },
        args.cache,
    )
    return res


def stage_w2(args):
    dev = "cuda"
    k = import_kernel()
    orc = load_oracle()
    tokens = orc["layer_input"].shape[0]
    topk_ids = orc["topk_ids"]
    routing, mask, activated = build_routing(topk_ids, dev)
    _, _, w2, w2_s = load_expert_weights(activated, dev)
    cache = torch.load(args.cache, map_location=dev)
    w2_row_s = expand_row_scales(w2_s, W2_N)

    def run_case(name, w13_out):
        """One w2 case: SwiGLU -> fp32-scale quantize -> w2 under every
        semantics, on the SAME input bytes."""
        mid = silu_mul(w13_out.float()).to(torch.bfloat16)
        padded = torch.zeros(
            (BATCH, NUM_TOPK, W2_K), dtype=torch.bfloat16, device=dev
        )
        padded[:tokens] = mid
        y_q, y_s = quantize_activation(padded)

        out = {}
        for stages in args.w2_stage_list:
            o = torch.zeros(
                (BATCH, NUM_TOPK, W2_N), dtype=torch.bfloat16, device=dev
            )
            k.moe_w2_ue8m0_sm100(
                y_q, y_s, w2, w2_row_s, routing, mask, o, stages
            )
            out[f"ue8m0_stages{stages}"] = o[:tokens].float()
        if not args.no_fallback:
            o = torch.zeros(
                (BATCH, NUM_TOPK, W2_N), dtype=torch.bfloat16, device=dev
            )
            k.moe_w2_blockscale_sm100(
                y_q, y_s, w2, w2_s.contiguous(), routing, mask, o
            )
            out["blockscale"] = o[:tokens].float()

        ref_fp32 = torch_grouped(
            y_q[:tokens], y_s[:tokens], w2, w2_s, topk_ids, tokens, W2_N, False
        )
        ref_trunc = torch_grouped(
            y_q[:tokens], y_s[:tokens], w2, w2_s, topk_ids, tokens, W2_N, True
        )
        ref_bf16 = torch_grouped_bf16(
            y_q[:tokens], y_s[:tokens], w2, w2_s, topk_ids, tokens, W2_N
        )
        block = {
            "input_source": name,
            "truncation_model": {
                "vs_fp32_scale": compare(ref_trunc, ref_fp32, "fp32_scale_torch")
            },
            "bf16_dequant_reference": {
                "vs_fp32_scale": compare(ref_bf16, ref_fp32, "fp32_scale_torch")
            },
        }
        for key, val in out.items():
            block[key] = {
                "vs_fp32_scale": compare(val, ref_fp32, "fp32_scale_torch"),
                "vs_bf16_dequant": compare(val, ref_bf16, "bf16_dequant_torch"),
                "vs_truncation_model": compare(
                    val, ref_trunc, "ue8m0_trunc_model"
                ),
            }
        # Cross-check that the two grouped pipeline depths agree bit-for-bit:
        # a depth-dependent difference would be a pipeline bug, not numerics.
        if len(args.w2_stage_list) > 1:
            a = out[f"ue8m0_stages{args.w2_stage_list[0]}"]
            b = out[f"ue8m0_stages{args.w2_stage_list[1]}"]
            block["stage_depth_bitwise_identical"] = bool(torch.equal(a, b))
        return block, out, {"fp32": ref_fp32, "trunc": ref_trunc, "bf16": ref_bf16}

    common, common_out, common_refs = run_case(
        "common_input_from_fp32ref_w13", cache["w13_out_fp32ref"].to(dev)
    )
    chain_u, chain_u_out, _ = run_case(
        "chain_ue8m0_w13", cache["w13_out_ue8m0"].to(dev)
    )
    res = {
        "stage": "w2",
        "shape": {"E": NUM_EXPERTS, "N": W2_N, "K": W2_K, "k_tiles": W2_K // BLOCK},
        "tokens": tokens,
        "activated_experts": len(activated),
        "gap7_4ktile_regime": {
            "fp8_k_tile_count": W2_K // BLOCK,
            "num_ab_stages_run": args.w2_stage_list,
            "completed_without_hang": True,
        },
        "common_input": common,
        "chain_ue8m0_input": chain_u,
        "env": env_block(),
    }
    save = {
        "w2_common_fp32ref": common_refs["fp32"].cpu(),
        "w2_chain_ue8m0": chain_u_out[
            f"ue8m0_stages{args.w2_stage_list[0]}"
        ].cpu(),
    }
    if not args.no_fallback:
        chain_f, chain_f_out, _ = run_case(
            "chain_blockscale_w13", cache["w13_out_fallback"].to(dev)
        )
        res["chain_blockscale_input"] = chain_f
        save["w2_chain_blockscale"] = chain_f_out["blockscale"].cpu()
    # The fp32-scale chain's w2 input IS the common input (both start from the
    # fp32-scale w13 output), so `common_refs` already holds its w2 outputs.
    save["w2_chain_fp32ref"] = common_refs["fp32"].cpu()
    save["w2_chain_bf16ref"] = common_refs["bf16"].cpu()
    torch.save(save, args.cache_w2)
    return res


def stage_token(args):
    """Project each semantics' expert outputs through the MoE block boundary:
    routed_out[t] = sum_slot topk_renorm_weight[t, slot] * w2_out[t, slot, :],
    compared against the HF oracle's own moe0.routed_expert_output."""
    dev = "cuda"
    orc = load_oracle()
    tokens = orc["layer_input"].shape[0]
    w = orc["topk_w"].to(dev).float()[:tokens]  # [tokens, topk]
    oracle_routed = orc["routed_expert_output"].to(dev).float()[:tokens]
    cache = torch.load(args.cache_w2, map_location=dev)

    def combine(x):
        return (x.float() * w.unsqueeze(-1)).sum(dim=1)

    variants = {}
    for name in (
        "w2_chain_ue8m0",
        "w2_chain_blockscale",
        "w2_chain_fp32ref",
        "w2_chain_bf16ref",
    ):
        if name in cache:
            variants[name] = combine(cache[name].to(dev))

    res = {"stage": "token", "tokens": tokens, "variants": {}}
    ref_name = "w2_chain_fp32ref"
    for name, val in variants.items():
        entry = {"vs_oracle_routed_expert_output": compare(
            val, oracle_routed, "oracle_hf_fp8"
        )}
        if name != ref_name and ref_name in variants:
            entry["vs_fp32_scale_chain"] = compare(
                val, variants[ref_name], "fp32_scale_chain"
            )
        res["variants"][name] = entry
    res["env"] = env_block()
    return res


# ------------------------------------------------------------------ assemble
def build_verdict(w13, w2, token, activation_quant=None):
    """The fail-closed decision, framed as P10's pillar gate: identify the
    MECHANISM and show its size relative to the preserved-scale class -- not a
    threshold picked after seeing the number."""
    u13 = w13["ue8m0_kernel"]["vs_fp32_scale"]
    u13_mech = w13["ue8m0_kernel"]["vs_truncation_model"]
    t13 = w13["truncation_model"]["vs_fp32_scale"]
    b13 = w13["bf16_dequant_reference"]["vs_fp32_scale"]
    u2 = w2["common_input"]["ue8m0_stages8"]["vs_fp32_scale"]
    u2_mech = w2["common_input"]["ue8m0_stages8"]["vs_truncation_model"]
    b2 = w2["common_input"]["bf16_dequant_reference"]["vs_fp32_scale"]

    # Pillar 1 -- the delta is the exponent-truncation mechanism and nothing
    # else: the torch truncation model reproduces the kernel to the same floor
    # the bf16-dequant reference sits at (both are bf16-rounding-limited).
    mechanism_identified = (
        u13_mech["frob_rel"] <= 5.0 * b13["frob_rel"]
        and u2_mech["frob_rel"] <= 5.0 * b2["frob_rel"]
    )
    # Pillar 2 -- it is SYSTEMATIC, not noise: a one-sided multiplicative
    # shrink (every row slope below 1) whose mean gain error is many
    # row-to-row standard deviations from zero.
    systematic = (
        u13["row_slope_max"] < 1.0
        and u2["row_slope_max"] < 1.0
        and abs(u13["gain_error_effect_size"]) > 3.0
    )
    # Pillar 3 -- it is far outside the preserved-scale class (P10: 2-4.4e-3)
    # and far outside the bf16-dequant reference's own distance.
    p10_class_hi = 4.4e-3
    outside_class = (
        u13["frob_rel"] > 10.0 * p10_class_hi
        and u13["frob_rel"] > 10.0 * b13["frob_rel"]
    )
    # Pillar 4 -- it survives to the MoE block boundary.
    tok = token["variants"].get("w2_chain_ue8m0", {})
    tok_ref = token["variants"].get("w2_chain_fp32ref", {})
    token_level = None
    if tok and tok_ref:
        token_level = (
            tok["vs_oracle_routed_expert_output"]["frob_rel"]
            > 10.0 * tok_ref["vs_oracle_routed_expert_output"]["frob_rel"]
        )

    biased = bool(
        mechanism_identified and systematic and outside_class and (token_level is not False)
    )
    verdict = {
        "biased": biased,
        "fallback_implemented": "blockscale_kernel" in w13,
        "mechanism": PREDICTION["mechanism"],
        "mechanism_source": [
            "include/mirage/persistent_kernel/tasks/blackwell/"
            "fp8_group_gemm_sm100.cuh:1379 (weight scale SFA)",
            "include/mirage/persistent_kernel/tasks/blackwell/"
            "fp8_group_gemm_sm100.cuh:1394 (activation scale SFB)",
            "tests/runtime_python/blackwell/sm100_fp8_moe/test_fp8_moe_gemm.py:"
            "68-75 float32_to_ue8m0_approx -- the in-tree DSV3 test compares "
            "the kernel against a reference that ALREADY applies this "
            "power-of-two floor, which is why the loss never showed up there",
        ],
        "pillars": {
            "mechanism_identified": mechanism_identified,
            "systematic_multiplicative_shrink": systematic,
            "outside_preserved_scale_class": outside_class,
            "survives_to_moe_block_boundary": token_level,
        },
        "evidence": {
            "w13": {
                "ue8m0_vs_fp32_scale": u13,
                "ue8m0_vs_truncation_model": u13_mech,
                "truncation_model_vs_fp32_scale": t13,
                "bf16_dequant_vs_fp32_scale": b13,
            },
            "w2": {
                "ue8m0_vs_fp32_scale": u2,
                "ue8m0_vs_truncation_model": u2_mech,
                "bf16_dequant_vs_fp32_scale": b2,
            },
            "token_level": token["variants"],
            "gap7_hang_guard": w2["gap7_4ktile_regime"],
            "preserved_scale_class_p10": [2.1e-3, 4.35e-3],
        },
        "methodology_note": (
            "The raw residual effect size (mean/std of actual-reference, P10's "
            "bias statistic) is NOT the detector for this mechanism: an "
            "exponent truncation multiplies each output element by "
            "1/(m_a*m_b), and a multiplicative shrink of a near-zero-mean "
            "output leaves a near-zero-mean residual. Both statistics are "
            "reported; the decision uses the per-row projection slope "
            "<actual,ref>/<ref,ref>, which is n-invariant and is exactly what "
            "a gain error moves. frob-rel (magnitude-weighted L2), never an "
            "elementwise max, is used for size -- a below-RMS e4m3 output "
            "element can show a large relative delta from rounding alone."
        ),
        "prediction": PREDICTION,
    }
    if w13.get("blockscale_kernel"):
        verdict["fallback_evidence"] = {
            "w13_vs_fp32_scale": w13["blockscale_kernel"]["vs_fp32_scale"],
            "w2_vs_fp32_scale": w2["common_input"]
            .get("blockscale", {})
            .get("vs_fp32_scale"),
            "kernel": "include/mirage/persistent_kernel/tasks/blackwell/"
            "moe_fp8_blockscale_sm100.cuh",
            "task_ids": [280, 281],
        }
    if activation_quant is not None:
        verdict["activation_quant_fp32_scale"] = activation_quant
    verdict["env"] = w13.get("env", {})
    verdict["stage_artifacts"] = {
        "w13": "p2_w13.json",
        "w2": "p2_w2.json",
        "token": "p2_token.json",
        "activation_quant": "p2_activation_quant.json",
    }
    verdict["measurement"] = {
        "shapes": {"w13": [256, 1024, 2048], "w2": [256, 2048, 512]},
        "tokens": w13.get("tokens"),
        "activated_experts": w13.get("activated_experts"),
        "activations": "HF oracle moe0.layer_input (real layer-0 MoE input)",
        "routing": "HF oracle moe0.topk_ids / moe0.topk_renorm_weights",
        "weights": "checkpoint layer-0 experts.{e}.{gate,up,down}_proj.weight "
        "+ weight_scale_inv (bf16 in the file, widened with .float() exactly "
        "as MPK's loader does)",
        "harness": "tests/runtime_python/blackwell/sm100_fp8_moe_qwen35/"
        "runtime_kernel_wrapper_qwen35.cu",
    }
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage", required=True,
        choices=["w13", "w2", "token", "assemble"],
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/p2_w13_cache.pt")
    ap.add_argument("--cache-w2", dest="cache_w2", default="/tmp/p2_w2_cache.pt")
    ap.add_argument("--num-ab-stages", dest="num_ab_stages", type=int, default=8)
    ap.add_argument("--w2-stages", dest="w2_stages", default="8,4")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--w13-json")
    ap.add_argument("--w2-json")
    ap.add_argument("--token-json")
    ap.add_argument("--activation-json")
    args = ap.parse_args()
    args.w2_stage_list = [int(x) for x in args.w2_stages.split(",")]

    if args.stage == "assemble":
        with open(args.w13_json) as f:
            w13 = json.load(f)
        with open(args.w2_json) as f:
            w2 = json.load(f)
        with open(args.token_json) as f:
            token = json.load(f)
        aq = None
        if args.activation_json and os.path.exists(args.activation_json):
            with open(args.activation_json) as f:
                aq = json.load(f)
        res = build_verdict(w13, w2, token, aq)
    elif args.stage == "w13":
        res = stage_w13(args)
    elif args.stage == "w2":
        res = stage_w2(args)
    else:
        res = stage_token(args)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=False)
    print(json.dumps(res, indent=1)[:4000])
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
