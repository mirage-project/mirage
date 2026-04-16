import argparse

import torch

import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import _E2M1_LUT, _UE4M3_LUT, encode_ue4m3, interleave_sf_tensor

DEVICE = "cuda"
GROUP_SIZE = 16
MIN_FP4 = -6.0
MAX_FP4 = 6.0
EPS = 1.0e-6
UE4M3_ONE = encode_ue4m3(1.0)


def quantize_nvfp4_reference(
    x: torch.Tensor,
    ue4m3_lut: torch.Tensor,
    e2m1_mag_lut: torch.Tensor,
    e2m1_mag_tie_break: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    groups = hidden // GROUP_SIZE
    blocked = x.view(rows, groups, GROUP_SIZE)
    group_scale = (blocked.abs().amax(dim=-1).clamp_min(EPS) / MAX_FP4)
    scale_bytes = (
        group_scale.unsqueeze(-1) - ue4m3_lut.view(1, 1, -1)
    ).abs().argmin(dim=-1).to(torch.uint8)
    quant = (blocked / ue4m3_lut[scale_bytes.long()].unsqueeze(-1)).clamp(
        MIN_FP4, MAX_FP4
    )
    nibbles = (
        quant.abs().unsqueeze(-1) - e2m1_mag_lut.view(1, 1, 1, -1)
    ).abs().add(e2m1_mag_tie_break.view(1, 1, 1, -1)).argmin(dim=-1).to(
        torch.uint8
    )
    nibbles = nibbles | ((quant < 0).to(torch.uint8) << 3)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).reshape(
        rows, hidden // 2
    )

    padded_rows = ((rows + 127) // 128) * 128
    if padded_rows != rows:
        padded_packed = torch.zeros(
            (padded_rows, hidden // 2), device=x.device, dtype=torch.uint8
        )
        padded_scale_bytes = torch.full(
            (padded_rows, groups), UE4M3_ONE, device=x.device, dtype=torch.uint8
        )
        padded_packed[:rows].copy_(packed)
        padded_scale_bytes[:rows].copy_(scale_bytes)
        packed = padded_packed
        scale_bytes = padded_scale_bytes

    return packed, interleave_sf_tensor(scale_bytes)


def benchmark(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=37)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert args.k in {128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168}
    assert args.k % 64 == 0

    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device=DEVICE, dtype=torch.float32)
    ue4m3_lut = torch.tensor(_UE4M3_LUT, device=DEVICE)
    e2m1_mag_lut = torch.tensor(_E2M1_LUT[:8], device=DEVICE)
    e2m1_mag_tie_break = torch.tensor(
        [0.0 if (i & 1) == 0 else 1.0e-7 for i in range(8)], device=DEVICE
    )

    ref_packed, ref_scale = quantize_nvfp4_reference(
        x, ue4m3_lut, e2m1_mag_lut, e2m1_mag_tie_break
    )
    out_packed, out_scale = runtime_kernel_blackwell.quantize_nvfp4_sm100(x)

    packed_match = torch.equal(out_packed, ref_packed)
    scale_match = torch.equal(out_scale, ref_scale)

    print(f"shape: M={args.m}, K={args.k}")
    print(f"packed shape: {tuple(out_packed.shape)}")
    print(f"scale shape: {tuple(out_scale.shape)}")
    print(f"packed exact match: {packed_match}")
    print(f"scale exact match: {scale_match}")

    if not packed_match:
        print("packed mismatches:", int((out_packed != ref_packed).sum().item()))
    if not scale_match:
        print("scale mismatches:", int((out_scale != ref_scale).sum().item()))

    assert packed_match
    assert scale_match

    custom_ms = benchmark(
        lambda: runtime_kernel_blackwell.quantize_nvfp4_sm100(x),
        args.warmup,
        args.reps,
    )
    reference_ms = benchmark(
        lambda: quantize_nvfp4_reference(
            x, ue4m3_lut, e2m1_mag_lut, e2m1_mag_tie_break
        ),
        args.warmup,
        args.reps,
    )

    print(f"custom:    {custom_ms:.6f} ms")
    print(f"reference: {reference_ms:.6f} ms")
    print(f"speedup:   {reference_ms / custom_ms:.2f}x")


if __name__ == "__main__":
    main()
