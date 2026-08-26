"""Regression guard for the 2026-06-13 UE8M0 conversion-contract bug (bug B).

DeepSeek-V3 FP8 expert weights are quantized against the RAW fp32 block
scale_inv (0% of which are powers of two), but the SM100 grouped GEMM packs
UE8M0 = 2^ceil(log2 s) scales and applies THOSE to the raw-quantized bytes —
inflating every 128x128 block by ceil/raw in [1,2) (measured ~53% GEMM error)
unless the FP8 payload is requantized. builder._requantize_moe_fp8_for_pow2
fixes it by rescaling the payload by raw/ceil so q_new * 2^ceil ~= q_old * raw.

This test reproduces the REAL contract (payload-vs-raw, kernel-applies-ceil)
that the self-consistent test quantizers hide, and asserts:
  1. the UN-requantized dequant has the bug-signature error (large),
  2. the requantized dequant matches the intended (raw-scale) weights,
  3. the requantize is idempotent (sentinel guard).

CPU-only (no GPU / no megakernel) — pure numeric guard on the builder helper.

Run: python tests/runtime_python/test_requantize_moe_fp8_pow2_testmode.py
"""

import os
import sys

import torch

THIS = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.join(THIS, "..", "..", "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from mirage.mpk.models.deepseek_v3.builder import DeepSeekV3Builder  # noqa: E402

FP8_MAX = 448.0
BLK = 128


def _quantize_against_raw(w_bf16):
    """Per-128x128-block quantize against the RAW scale = amax/448 (the
    DeepSeek checkpoint contract). Returns (fp8 payload (E,N,K),
    raw scale_inv (E, N//128, K//128) fp32)."""
    E, N, K = w_bf16.shape
    nb_n, nb_k = N // BLK, K // BLK
    payload = torch.empty(E, N, K, dtype=torch.float8_e4m3fn)
    scale = torch.empty(E, nb_n, nb_k, dtype=torch.float32)
    wf = w_bf16.float()
    for e in range(E):
        for bn in range(nb_n):
            for bk in range(nb_k):
                blk = wf[e, bn * BLK:(bn + 1) * BLK, bk * BLK:(bk + 1) * BLK]
                raw = max(blk.abs().max().item(), 1e-10) / FP8_MAX
                scale[e, bn, bk] = raw
                payload[e, bn * BLK:(bn + 1) * BLK, bk * BLK:(bk + 1) * BLK] = \
                    torch.clamp(blk / raw, -FP8_MAX, FP8_MAX).to(
                        torch.float8_e4m3fn)
    return payload, scale


def _dequant(payload_fp8, scale_block):
    E, N, K = payload_fp8.shape
    se = scale_block.repeat_interleave(BLK, 1).repeat_interleave(BLK, 2)
    return payload_fp8.float() * se[:, :N, :K]


def _relerr(a, b):
    nz = b.abs() > 0
    return ((a - b).abs()[nz] / b.abs()[nz]).mean().item()


def main():
    torch.manual_seed(0)
    E, N, K = 2, 256, 384  # non-pow2 K-block count exercised per expert
    # bf16 weights with realistic block-varying magnitude so raw scales are
    # generically NOT powers of two (the whole point of the bug).
    w = (torch.randn(E, N, K) * torch.rand(E, N, 1).add(0.1)).to(torch.bfloat16)

    payload0, raw_scale = _quantize_against_raw(w)
    ceil_scale = torch.pow(2.0, torch.ceil(torch.log2(raw_scale.clamp(min=1e-30))))

    # The kernel applies the CEIL scale to whatever payload is attached.
    intended = _dequant(payload0, raw_scale)            # what we MUST compute
    buggy = _dequant(payload0, ceil_scale)              # un-requantized (bug)
    buggy_err = _relerr(buggy, intended)
    print(f"un-requantized (payload-vs-raw, kernel-applies-ceil) rel err: "
          f"{buggy_err*100:.1f}%  (bug signature — must be large)")
    assert buggy_err > 0.15, (
        f"expected the un-requantized contract to show the bug-signature "
        f"inflation (>15%); got {buggy_err*100:.2f}% — the test no longer "
        f"reproduces the real raw-scale contract (bug B would be invisible).")

    # Run the production fix on a checkpoint-style state_dict.
    sd = {"w.weight": payload0.clone(),
          "w.weight_scale_inv": raw_scale.clone()}
    DeepSeekV3Builder._requantize_moe_fp8_for_pow2(sd, "w.weight",
                                                   "w.weight_scale_inv")
    fixed = _dequant(sd["w.weight"], ceil_scale)        # kernel still applies ceil
    fixed_err = _relerr(fixed, intended)
    print(f"requantized dequant rel err vs intended: {fixed_err*100:.2f}%  "
          f"(must be small — pure FP8 re-round)")
    assert fixed_err < 0.08, (
        f"requantize did NOT restore the intended weights: rel err "
        f"{fixed_err*100:.2f}% (expected <8%, pure fp8 re-rounding).")
    assert fixed_err < buggy_err / 3, (
        f"requantize barely helped ({fixed_err*100:.2f}% vs buggy "
        f"{buggy_err*100:.2f}%) — fix ineffective.")

    # Idempotence: re-applying must be a no-op (sentinel guard).
    before = sd["w.weight"].view(torch.uint8).clone()
    DeepSeekV3Builder._requantize_moe_fp8_for_pow2(sd, "w.weight",
                                                   "w.weight_scale_inv")
    assert torch.equal(sd["w.weight"].view(torch.uint8), before), \
        "requantize is not idempotent — re-applying compounded the shrink."

    print("PASS: UE8M0 requantize restores raw-scale weights, idempotent, "
          "and the bug-signature contract is reproduced.")


if __name__ == "__main__":
    main()
