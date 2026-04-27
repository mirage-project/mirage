import argparse

import _runtime_path  # noqa: F401
import torch

import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import (
    _pad_rows_uint8,
    _to_blocked_sf,
    encode_ue4m3,
    interleave_sf_tensor,
    make_random_nvfp4_tensors,
)


DEVICE = "cuda"
DTYPE_OUT = torch.float32
DEFAULT_M_VALUES = [4096]#list(range(1, 129)) + [4096]
DEFAULT_N_VALUES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168]
DEFAULT_K_VALUES = [256, 512, 768, 1024, 1536, 2048, 4096, 7168]
SUPPORTED_N_VALUES = set(DEFAULT_N_VALUES)
SUPPORTED_K_VALUES = set(DEFAULT_K_VALUES)
SMALL_M_MAX = 128
LARGE_SHAPE = (4096, 128, 768)
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


def scaled_mm_runner(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    logical_rows: int,
):
    padded_rows = ((logical_rows + 127) // 128) * 128
    if padded_rows != logical_rows:
        x = _pad_rows_uint8(x, padded_rows, fill_value=0)
        x_scale = _pad_rows_uint8(x_scale, padded_rows, fill_value=UE4M3_ONE)
    weight_fp4_t = weight.view(torch.float4_e2m1fn_x2).transpose(0, 1)
    x_fp4 = x.view(torch.float4_e2m1fn_x2)
    blocked_weight_scale = _to_blocked_sf(weight_scale)
    blocked_x_scale = _to_blocked_sf(x_scale)

    def run():
        return torch._scaled_mm(
            x_fp4,
            weight_fp4_t,
            blocked_x_scale,
            blocked_weight_scale,
            bias=None,
            out_dtype=DTYPE_OUT,
        )[:logical_rows]

    return run


def supported_shapes(m_values: list[int], n_values: list[int], k_values: list[int]):
    for m in m_values:
        if 1 <= m <= SMALL_M_MAX:
            for n in n_values:
                for k in k_values:
                    if n in SUPPORTED_N_VALUES and k in SUPPORTED_K_VALUES:
                        yield m, n, k
        elif (
            m == LARGE_SHAPE[0]
            and LARGE_SHAPE[1] in n_values
            and LARGE_SHAPE[2] in k_values
        ):
            yield LARGE_SHAPE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", type=parse_int_list, default=DEFAULT_M_VALUES)
    parser.add_argument("--n-values", type=parse_int_list, default=DEFAULT_N_VALUES)
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    shapes = list(dict.fromkeys(supported_shapes(args.m_values, args.n_values, args.k_values)))

    print(
        f"linear_nvfp4_sm100_no_quantization | shapes={len(shapes)} | warmup={args.warmup} | reps={args.reps} | units=us"
    )
    for m, n, k in shapes:
        x, weight, x_scale, weight_scale = make_random_nvfp4_tensors(m, n, k, device=DEVICE)
        x_scale_interleaved = interleave_sf_tensor(x_scale)
        weight_scale_interleaved = interleave_sf_tensor(weight_scale)
        output = torch.empty((m, n), device=DEVICE, dtype=DTYPE_OUT)

        baseline = scaled_mm_runner(x, x_scale, weight, weight_scale, m)
        reference = baseline()

        runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(
            x, x_scale_interleaved, weight, weight_scale_interleaved, None, output
        )
        max_diff = (output - reference).abs().max().item()
        custom_us = benchmark_us(
            lambda: runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(
                x, x_scale_interleaved, weight, weight_scale_interleaved, None, output
            ),
            args.warmup,
            args.reps,
        )
        baseline_us = benchmark_us(baseline, args.warmup, args.reps)
        speedup = baseline_us / custom_us
        path = "swapAB" if m <= SMALL_M_MAX else "1d2d"
        print(
            f"M={m:4d} N={n:4d} K={k:4d} | {path:6s} | custom {custom_us:8.1f} us | "
            f"baseline {baseline_us:8.1f} us | speedup {speedup:5.2f}x | diff {max_diff:.6f}"
        )


if __name__ == "__main__":
    main()
