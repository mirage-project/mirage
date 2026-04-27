import argparse

import _runtime_path  # noqa: F401
import torch

import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import _E2M1_LUT, _UE4M3_LUT, encode_ue4m3, interleave_sf_tensor


DEVICE = "cuda"
DEFAULT_M_VALUES = list(range(1, 129)) + [4096]
DEFAULT_K_VALUES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168]
GROUP_SIZE = 16
MIN_FP4 = -6.0
MAX_FP4 = 6.0
EPS = 1.0e-6
UE4M3_ONE = encode_ue4m3(1.0)


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def benchmark_us(fn, warmup: int, reps: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / reps


def quantize_nvfp4_reference(
    x: torch.Tensor,
    ue4m3_lut: torch.Tensor,
    e2m1_mag_lut: torch.Tensor,
    e2m1_mag_tie_break: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, hidden = x.shape
    groups = hidden // GROUP_SIZE
    blocked = x.view(rows, groups, GROUP_SIZE)
    scale = blocked.abs().amax(dim=-1).clamp_min(EPS) / MAX_FP4
    scale_bytes = (
        scale.unsqueeze(-1) - ue4m3_lut.view(1, 1, -1)
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
        padded_scale = torch.full(
            (padded_rows, groups), UE4M3_ONE, device=x.device, dtype=torch.uint8
        )
        padded_packed[:rows].copy_(packed)
        padded_scale[:rows].copy_(scale_bytes)
        packed = padded_packed
        scale_bytes = padded_scale
    return packed, interleave_sf_tensor(scale_bytes)


def interleaved_nvfp4_scale_offset(
    row_idx: int, group_idx: int, num_k_outer: int, scale_outer_stride: int = 32 * 4 * 4
) -> int:
    row_in_block = row_idx & 127
    return (
        (row_idx >> 7) * num_k_outer * scale_outer_stride
        + (group_idx >> 2) * scale_outer_stride
        + (row_in_block & 31) * 16
        + ((row_in_block >> 5) & 3) * 4
        + (group_idx & 3)
    )


def locate_scale_mismatch(flat_idx: int, rows: int, hidden: int) -> tuple[int, int] | None:
    padded_rows = ((rows + 127) // 128) * 128
    num_groups = hidden // GROUP_SIZE
    num_k_outer = num_groups // 4
    for row_idx in range(padded_rows):
        for group_idx in range(num_groups):
            if (
                interleaved_nvfp4_scale_offset(row_idx, group_idx, num_k_outer)
                == flat_idx
            ):
                return row_idx, group_idx
    return None


def print_failure_details(
    x: torch.Tensor,
    ref_packed: torch.Tensor,
    ref_scale: torch.Tensor,
    out_packed: torch.Tensor,
    out_scale: torch.Tensor,
) -> None:
    hidden = x.shape[1]
    packed_mismatch_count = int((out_packed != ref_packed).sum().item())
    scale_mismatch_count = int((out_scale != ref_scale).sum().item())
    print(
        f"  mismatch_counts | packed={packed_mismatch_count} | scale={scale_mismatch_count}"
    )

    scale_group = None
    if scale_mismatch_count:
        flat_idx = int((out_scale != ref_scale).flatten().nonzero()[0].item())
        scale_loc = locate_scale_mismatch(flat_idx, rows=x.shape[0], hidden=hidden)
        ref_byte = int(ref_scale.flatten()[flat_idx].item())
        out_byte = int(out_scale.flatten()[flat_idx].item())
        if scale_loc is not None:
            row_idx, group_idx = scale_loc
            scale_group = (row_idx, group_idx)
            group_vals = x[row_idx, group_idx * GROUP_SIZE : (group_idx + 1) * GROUP_SIZE]
            group_max = float(group_vals.abs().amax().item())
            target_scale = max(group_max, EPS) / MAX_FP4
            ref_value = float(_UE4M3_LUT[ref_byte])
            out_value = float(_UE4M3_LUT[out_byte])
            print(
                "  first_scale_mismatch"
                f" | row={row_idx} | group={group_idx}"
                f" | target_scale={target_scale:.9f}"
                f" | ref_byte={ref_byte} ({ref_value:.9f})"
                f" | out_byte={out_byte} ({out_value:.9f})"
                f" | ref_err={abs(target_scale - ref_value):.9f}"
                f" | out_err={abs(target_scale - out_value):.9f}"
            )

    if packed_mismatch_count:
        flat_idx = int((out_packed != ref_packed).flatten().nonzero()[0].item())
        cols = hidden // 2
        row_idx = flat_idx // cols
        col_idx = flat_idx % cols
        group_idx = col_idx // (GROUP_SIZE // 2)
        byte_in_group = col_idx % (GROUP_SIZE // 2)
        elem_idx = group_idx * GROUP_SIZE + byte_in_group * 2
        pair_vals = x[row_idx, elem_idx : elem_idx + 2]
        print(
            "  first_packed_mismatch"
            f" | row={row_idx} | col={col_idx} | group={group_idx}"
            f" | byte_in_group={byte_in_group}"
            f" | ref_byte={int(ref_packed[row_idx, col_idx].item())}"
            f" | out_byte={int(out_packed[row_idx, col_idx].item())}"
            f" | source_pair={[float(v) for v in pair_vals.tolist()]}"
        )
        if scale_group == (row_idx, group_idx):
            print(
                "  note | packed mismatch is in the same 16-element group as the first scale mismatch."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", type=parse_int_list, default=DEFAULT_M_VALUES)
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Only print shapes where packed or scale correctness fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print mismatch counts and the first failing row/group for each failing shape.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ue4m3_lut = torch.tensor(_UE4M3_LUT, device=DEVICE)
    e2m1_mag_lut = torch.tensor(_E2M1_LUT[:8], device=DEVICE)
    e2m1_mag_tie_break = torch.tensor(
        [0.0 if (i & 1) == 0 else 1.0e-7 for i in range(8)], device=DEVICE
    )

    print(
        f"quantize_nvfp4_sm100 | shapes={len(args.m_values) * len(args.k_values)} | warmup={args.warmup} | reps={args.reps} | units=us"
    )
    for m in args.m_values:
        for k in args.k_values:
            x = torch.randn((m, k), device=DEVICE, dtype=torch.float32)
            ref_packed, ref_scale = quantize_nvfp4_reference(
                x, ue4m3_lut, e2m1_mag_lut, e2m1_mag_tie_break
            )
            out_packed, out_scale = runtime_kernel_blackwell.quantize_nvfp4_sm100(x)
            packed_match = torch.equal(out_packed, ref_packed)
            scale_match = torch.equal(out_scale, ref_scale)
            if args.only_failures and packed_match and scale_match:
                continue
            custom_us = benchmark_us(
                lambda: runtime_kernel_blackwell.quantize_nvfp4_sm100(x),
                args.warmup,
                args.reps,
            )
            reference_us = benchmark_us(
                lambda: quantize_nvfp4_reference(
                    x,
                    ue4m3_lut,
                    e2m1_mag_lut,
                    e2m1_mag_tie_break,
                ),
                args.warmup,
                args.reps,
            )
            speedup = reference_us / custom_us
            print(
                f"M={m:4d} K={k:4d} | custom {custom_us:8.1f} us | "
                f"reference {reference_us:8.1f} us | speedup {speedup:5.2f}x | "
                f"packed {str(packed_match):5s} | scale {str(scale_match):5s}"
            )
            if args.verbose and (not packed_match or not scale_match):
                print_failure_details(
                    x, ref_packed, ref_scale, out_packed, out_scale
                )


if __name__ == "__main__":
    main()
