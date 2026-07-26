"""Preserved-block-scale dense FP8 GEMM on REAL Qwen3.5-35B-A3B-FP8 tensors.

Drives the kernel with layer-0 (and layer-3) checkpoint weights AND their
checkpoint `weight_scale_inv` block scales, fed through untouched: no UE8M0
requantization, no per-row collapse (docs/qwen35/v1-architecture.md 6.2).

Metrics follow the P10 methodology
(demo/qwen3_5/accept/probes/fp8/p10_fp8_dense_bar.py `diff_stats`): the gate is
`frob_rel_error` (magnitude-weighted relative L2), never an elementwise max --
a below-RMS output element can show a large relative delta purely from e4m3
rounding -- plus `bias_effect_size` = mean/std, the n-invariant "is there a
systematic bias" statistic. P10 keeps the z-score only as a diagnostic because
it grows with sqrt(n) at fixed effect size; here it is additionally used as a
DETECTABILITY guard, since these per-case tensors are small enough that the
effect-size estimator is itself noisy (see `bias_stats`).

Three references, in increasing distance from the kernel:
  fp32_exact      -- same fp8 bytes and same fp32 scales, contracted in fp32.
                     The kernel's only legitimate deviation from this is the
                     bf16 rounding of its own output, so the reported ratio to
                     that floor must be ~1.
  bf16_dequant    -- ACCEPTANCE reference: the same fp8 bytes dequantized to
                     bf16 and contracted as a bf16 GEMM, i.e. what MPK's bf16
                     linear task would compute from a block-dequantized
                     checkpoint. Expected in the 2-4e-3 class.
  bf16_dense      -- diagnostic only: the bf16 scaffold path, which never
                     quantizes the activation. The gap here is the cost of fp8
                     activation quantization, not a kernel error.

Run:  QWEN35_SNAPSHOT=... python test_linear_fp8_blockscale_ckpt.py
"""

import json
import os

import torch
from safetensors import safe_open

import runtime_kernel_blackwell_linear_fp8_blockscale as linear_kernel

# The fp32 references below must be true IEEE fp32, not TF32 on tensor cores,
# or the "kernel is within one bf16 rounding of exact" check compares against a
# reference that is itself ~1e-3 off.
torch.backends.cuda.matmul.allow_tf32 = False

BLOCK = 128
FP8_MAX = 448.0
EPS = 1e-10
CKPT_REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
SNAPSHOT = os.environ.get(
    "QWEN35_SNAPSHOT",
    os.path.expanduser(
        f"~/mpk-qwen35/hf/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/{CKPT_REVISION}"
    ),
)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../../.."))
ORACLE_DUMPS = os.path.join(
    REPO_ROOT, "demo/qwen3_5/oracle/dumps_sample/prefill/tensors"
)

# P10's gate on the cutlass-vs-triton spread. Reused here for the bias check,
# paired with a detectability guard (see bias_stats) because these per-case
# tensors are far smaller than P10's pooled ones.
BIAS_EFFECT_SIZE_THRESHOLD = 0.1
BIAS_ZSCORE_THRESHOLD = 3.0
# frob_rel vs the bf16 dequant reference. The mechanism is the bf16 rounding of
# the reference's own operands and output (~2^-8/sqrt(3) relative), so this sits
# in P10's 2-4e-3 class by construction; 6e-3 leaves margin without admitting a
# scale bug (which moves the fp32_exact ratio below, not this number).
FROB_REL_VS_BF16_THRESHOLD = 6e-3
# The kernel must not exceed its own output-rounding floor by more than this.
FP32_EXACT_RATIO_THRESHOLD = 1.6

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


def quantize_activation(x_bf16):
    """vLLM's dynamic per-token, per-128-group fp8 quantization, fp32 scales
    (docs/qwen35/vllm-graph.md 3.4): absmax = max(max|x|, 1e-10),
    scale = absmax / 448, x / scale, clamp before the RN-even e4m3 cast."""
    m, k = x_bf16.shape
    xf = x_bf16.float().reshape(m, k // BLOCK, BLOCK)
    absmax = xf.abs().amax(dim=-1).clamp(min=EPS)
    scale = absmax / FP8_MAX
    q = (xf / scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    return q.reshape(m, k).to(torch.float8_e4m3fn), scale.contiguous()


def dequant_blocks(q, scale):
    """W_real = W_fp8 * weight_scale_inv, block-expanded."""
    n, k = q.shape
    s = scale.repeat_interleave(BLOCK, dim=0)[:n].repeat_interleave(BLOCK, dim=1)[:, :k]
    return q.float() * s


def dequant_groups(q, scale):
    m, k = q.shape
    return (q.float().reshape(m, k // BLOCK, BLOCK) * scale.unsqueeze(-1)).reshape(m, k)


def frob_rel(actual, ref):
    return (actual - ref).norm().item() / max(ref.norm().item(), 1e-30)


def bias_stats(actual, ref):
    """P10's n-INVARIANT bias statistic mean/std, plus its z-score diagnostic.

    P10 gates on the effect size because a z-score grows with sqrt(n) at fixed
    effect size, so pooling more samples can make an irrelevant mean shift look
    significant. The converse matters here: these per-case tensors are as small
    as B*N = 128 elements, where the effect-size ESTIMATOR itself has a standard
    error of 1/sqrt(n) ~ 0.088. A case therefore fails only when the bias is
    both practically large (|effect| > threshold) AND statistically detectable
    (|z| > 3), which is what "no systematic bias" actually asserts.
    """
    diff = (actual - ref).flatten()
    n = diff.numel()
    std = diff.std(unbiased=True).item()
    mean = diff.mean().item()
    if std <= 1e-30:
        effect = 0.0 if abs(mean) < 1e-30 else float("inf")
        return effect, 0.0 if abs(mean) < 1e-30 else float("inf")
    return mean / std, mean / (std / (n ** 0.5))


def real_out_proj_activation():
    """Layer-0 GDN gated-norm output = the real input to out_proj [8, 4096].

    Stored by the HF oracle as [tokens*heads, head_dim] = [256, 128]
    (demo/qwen3_5/oracle/dumps_sample/prefill/manifest.json).
    """
    path = os.path.join(ORACLE_DUMPS, "gdn.gated_norm_out.pt")
    if not os.path.exists(path):
        return None
    t = torch.load(path, map_location="cpu")
    return t.reshape(8, 4096).to("cuda").to(torch.bfloat16).contiguous()


def build_cases(index):
    """(label, weight_key, output rows, activation source) for the dense GEMMs
    Qwen3.5 runs in fp8 (docs/qwen35/v1-architecture.md 6.1)."""
    p = "model.language_model.layers."
    candidates = [
        ("GDN.layer0.in_proj_qkv", p + "0.linear_attn.in_proj_qkv.weight", None),
        ("GDN.layer0.in_proj_z", p + "0.linear_attn.in_proj_z.weight", None),
        ("GDN.layer0.out_proj", p + "0.linear_attn.out_proj.weight", "gated_norm"),
        ("attn.layer3.q_proj", p + "3.self_attn.q_proj.weight", None),
        ("attn.layer3.o_proj", p + "3.self_attn.o_proj.weight", None),
        (
            "shared_expert.layer0.down_proj",
            p + "0.mlp.shared_expert.down_proj.weight",
            None,
        ),
    ]
    cases = []
    for label, key, act_source in candidates:
        if key not in index["weight_map"]:
            print(f"  [skip] {label}: {key} absent from the checkpoint index")
            continue
        cases.append((label, key, act_source))
    return cases


def run_case(label, weight_key, act_source, index, output_size, batch_size, generator):
    w_full = _get(index, weight_key)
    s_full = _get(index, weight_key + "_scale_inv")
    assert w_full.dtype == torch.float8_e4m3fn, w_full.dtype
    # The checkpoint stores weight_scale_inv in BF16; MPK's loader widens with
    # .float() (docs/qwen35/mpk-gaps.md 2.2.1). Do exactly that, nothing else.
    assert s_full.dtype == torch.bfloat16, s_full.dtype
    reduction_size = w_full.shape[1]

    w_q = w_full[:output_size].contiguous().to("cuda")
    w_scale = s_full[: output_size // BLOCK].float().contiguous().to("cuda")
    assert tuple(w_scale.shape) == (output_size // BLOCK, reduction_size // BLOCK)

    if act_source == "gated_norm":
        real = real_out_proj_activation()
        if real is None or batch_size > real.shape[0]:
            x_bf16 = torch.randn(
                (batch_size, reduction_size),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            act_kind = "synthetic"
        else:
            x_bf16 = real[:batch_size].contiguous()
            act_kind = "oracle-layer0"
    else:
        x_bf16 = torch.randn(
            (batch_size, reduction_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        act_kind = "synthetic"

    x_q, x_scale = quantize_activation(x_bf16)
    output = torch.empty(
        (batch_size, output_size), device="cuda", dtype=torch.bfloat16
    )
    linear_kernel.linear_fp8_blockscale_sm100(
        x_q, x_scale, w_q, w_scale, None, output
    )
    torch.cuda.synchronize()

    w_deq = dequant_blocks(w_q, w_scale)
    x_deq = dequant_groups(x_q, x_scale)
    ref_fp32 = x_deq @ w_deq.t()
    ref_bf16 = (
        x_deq.to(torch.bfloat16) @ w_deq.to(torch.bfloat16).t()
    ).float()
    ref_bf16_dense = (x_bf16 @ w_deq.to(torch.bfloat16).t()).float()

    out_f32 = output.float()
    floor = frob_rel(ref_fp32.to(torch.bfloat16).float(), ref_fp32)
    effect, zscore = bias_stats(out_f32, ref_bf16)
    stats = {
        "case": label,
        "N": output_size,
        "K": reduction_size,
        "B": batch_size,
        "activation": act_kind,
        "frob_rel_vs_bf16_dequant": frob_rel(out_f32, ref_bf16),
        "frob_rel_vs_fp32_exact": frob_rel(out_f32, ref_fp32),
        "bf16_output_rounding_floor": floor,
        "fp32_exact_ratio_to_floor": frob_rel(out_f32, ref_fp32) / max(floor, 1e-30),
        "bias_effect_size_vs_bf16_dequant": effect,
        "bias_zscore_vs_bf16_dequant": zscore,
        "frob_rel_bf16_dense_scaffold": frob_rel(ref_bf16_dense, ref_fp32),
    }
    print(
        f"  {label:<32} N={output_size:<4} K={reduction_size:<5} B={batch_size:<3} "
        f"act={act_kind:<13} "
        f"frob_rel(bf16_dequant)={stats['frob_rel_vs_bf16_dequant']:.3e} "
        f"frob_rel(fp32_exact)={stats['frob_rel_vs_fp32_exact']:.3e} "
        f"ratio_to_floor={stats['fp32_exact_ratio_to_floor']:.2f} "
        f"bias_effect={effect:+.3f} bias_z={zscore:+.2f} "
        f"[bf16-dense scaffold vs fp32 exact: "
        f"{stats['frob_rel_bf16_dense_scaffold']:.3e}]"
    )
    assert stats["frob_rel_vs_bf16_dequant"] <= FROB_REL_VS_BF16_THRESHOLD, stats
    assert stats["fp32_exact_ratio_to_floor"] <= FP32_EXACT_RATIO_THRESHOLD, stats
    assert (
        abs(effect) <= BIAS_EFFECT_SIZE_THRESHOLD
        or abs(zscore) <= BIAS_ZSCORE_THRESHOLD
    ), stats
    return stats


def main():
    if not os.path.isdir(SNAPSHOT):
        raise SystemExit(f"checkpoint snapshot not found: {SNAPSHOT}")
    index = load_index()
    generator = torch.Generator(device="cuda").manual_seed(20260726)

    print("=== preserved fp32 block scales on REAL Qwen3.5 checkpoint tensors ===")
    print(f"snapshot: {SNAPSHOT}")
    cases = build_cases(index)
    all_stats = []
    for label, key, act_source in cases:
        for output_size in (128, 256):
            for batch_size in (1, 8, 16):
                all_stats.append(
                    run_case(
                        label,
                        key,
                        act_source,
                        index,
                        output_size,
                        batch_size,
                        generator,
                    )
                )

    summary = {
        "cases": len(all_stats),
        "max_frob_rel_vs_bf16_dequant": max(
            s["frob_rel_vs_bf16_dequant"] for s in all_stats
        ),
        "max_fp32_exact_ratio_to_floor": max(
            s["fp32_exact_ratio_to_floor"] for s in all_stats
        ),
        "max_abs_bias_effect_size": max(
            abs(s["bias_effect_size_vs_bf16_dequant"]) for s in all_stats
        ),
        "max_abs_bias_zscore": max(
            abs(s["bias_zscore_vs_bf16_dequant"]) for s in all_stats
        ),
        "max_frob_rel_bf16_dense_scaffold": max(
            s["frob_rel_bf16_dense_scaffold"] for s in all_stats
        ),
        "thresholds": {
            "frob_rel_vs_bf16_dequant": FROB_REL_VS_BF16_THRESHOLD,
            "fp32_exact_ratio_to_floor": FP32_EXACT_RATIO_THRESHOLD,
            "bias_effect_size": BIAS_EFFECT_SIZE_THRESHOLD,
            "bias_zscore": BIAS_ZSCORE_THRESHOLD,
        },
    }
    print("SUMMARY " + json.dumps(summary))
    print("ALL CHECKPOINT-TENSOR TESTS PASSED")


if __name__ == "__main__":
    main()
