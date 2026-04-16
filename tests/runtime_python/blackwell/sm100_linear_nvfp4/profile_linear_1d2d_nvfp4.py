import argparse
import warnings

import torch

import runtime_kernel_blackwell_linear_nvfp4 as runtime_kernel_blackwell
from nvfp4_util import (
    _pad_rows_uint8,
    _to_blocked_sf,
    encode_ue4m3,
    interleave_sf_tensor,
    make_random_nvfp4_tensors,
)

try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False
    warnings.warn("flashinfer not available; skipping flashinfer baselines")


DEVICE = "cuda"
DTYPE_OUT = torch.float32
DEFAULT_M_VALUES = list(range(1, 129)) + [4096]
DEFAULT_N_VALUES = [128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168]
DEFAULT_K_VALUES = [256, 512, 768, 1024, 1536, 2048, 4096, 7168]
SUPPORTED_N_VALUES = set(DEFAULT_N_VALUES)
SUPPORTED_K_VALUES = set(DEFAULT_K_VALUES)
SMALL_M_MAX = 128
LARGE_SHAPE = (4096, 128, 768)
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


def benchmark_graph_us(fn, warmup: int, reps: int) -> float:
    """Benchmark using CUDA graph replay to eliminate CPU dispatch overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def flashinfer_runner(x: torch.Tensor, weight: torch.Tensor, backend: str):
    """Build a pre-quantized flashinfer mm_fp4 runner.

    x:       (M, K) float32 activations
    weight:  (N, K) bfloat16 weights (quantized once offline)
    backend: 'trtllm' or 'cutlass'

    Pre-quantizes tensors so that only the GEMM kernel itself is timed.
    """
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)

    x_gsf = (448.0 * 6.0) / x_bf16.float().abs().nan_to_num().clamp_min(EPS).max()
    w_gsf = (448.0 * 6.0) / w_bf16.float().abs().nan_to_num().clamp_min(EPS).max()
    alpha = 1.0 / (x_gsf * w_gsf)

    w_do_shuffle = (backend == "trtllm")
    x_fp4, x_sf = flashinfer.nvfp4_quantize(x_bf16, x_gsf, do_shuffle=False)
    w_fp4, w_sf = flashinfer.nvfp4_quantize(w_bf16, w_gsf, do_shuffle=w_do_shuffle)

    # Use .clone() (not .contiguous()) to preserve the F-order strides trtllm requires.
    w_fp4_t = w_fp4.T.clone()
    w_sf_t = w_sf.T.clone()

    def run():
        return flashinfer.mm_fp4(
            x_fp4,
            w_fp4_t,
            x_sf,
            w_sf_t,
            alpha,
            torch.bfloat16,
            None,
            backend=backend,
        )

    return run


def scaled_mm_runner(x: torch.Tensor, weight: torch.Tensor):
    """Build a pre-quantized torch._scaled_mm runner (GEMM only, no quantization).

    Uses flashinfer to quantize into plain (M, K/16) scale factor layout that
    _to_blocked_sf expects.

    x:       (M, K) float32 activations
    weight:  (N, K) bfloat16 weights
    """
    x_bf16 = x.to(torch.bfloat16)
    w_bf16 = weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)

    x_gsf = (448.0 * 6.0) / x_bf16.float().abs().nan_to_num().clamp_min(EPS).max()
    w_gsf = (448.0 * 6.0) / w_bf16.float().abs().nan_to_num().clamp_min(EPS).max()

    x_fp4_raw, x_sf_raw = flashinfer.nvfp4_quantize(x_bf16, x_gsf, do_shuffle=False)
    w_fp4_raw, w_sf_raw = flashinfer.nvfp4_quantize(w_bf16, w_gsf, do_shuffle=False)

    # flashinfer returns (M, K/2) uint8 packed and (M, K/16) uint8 scale factors
    # Reinterpret as float4_e2m1fn_x2 for _scaled_mm
    a_fp4 = w_fp4_raw.view(torch.float4_e2m1fn_x2)   # (N, K/2)
    b_fp4 = x_fp4_raw.view(torch.float4_e2m1fn_x2)   # (M, K/2)

    # Convert scale factors to blocked layout; pad x_sf rows to multiple of 128
    logical_rows = x_sf_raw.shape[0]
    padded_rows = ((logical_rows + 127) // 128) * 128
    x_sf_padded = _pad_rows_uint8(x_sf_raw.cuda(), padded_rows, fill_value=UE4M3_ONE)
    scale_a = _to_blocked_sf(w_sf_raw.cuda())
    scale_b = _to_blocked_sf(x_sf_padded)

    def run():
        return torch._scaled_mm(
            b_fp4,
            a_fp4.transpose(0, 1),
            scale_b,
            scale_a,
            bias=None,
            out_dtype=torch.float32,
        )

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
    parser.add_argument(
        "--no-flashinfer",
        action="store_true",
        help="Skip flashinfer baselines even if the package is available",
    )
    parser.add_argument(
        "--no-scaled-mm",
        action="store_true",
        help="Skip torch._scaled_mm baseline",
    )
    args = parser.parse_args()

    use_flashinfer = _FLASHINFER_AVAILABLE and not args.no_flashinfer
    use_scaled_mm = not args.no_scaled_mm

    torch.manual_seed(args.seed)
    shapes = list(dict.fromkeys(supported_shapes(args.m_values, args.n_values, args.k_values)))

    print(
        f"linear_nvfp4_sm100 | shapes={len(shapes)} | warmup={args.warmup} | reps={args.reps} | units=us"
    )
    for m, n, k in shapes:
        x = torch.randn((m, k), device=DEVICE, dtype=torch.float32)
        _, weight, _, weight_scale = make_random_nvfp4_tensors(1, n, k, device=DEVICE)
        weight_scale_interleaved = interleave_sf_tensor(weight_scale)
        output = torch.empty((m, n), device=DEVICE, dtype=DTYPE_OUT)

        # Pre-quantize activations once — timing measures GEMM only, not quantization.
        x_q_list = runtime_kernel_blackwell.quantize_nvfp4_sm100(x)
        x_q, x_sf = x_q_list[0], x_q_list[1]

        smm_us = float("nan")
        if use_scaled_mm and _FLASHINFER_AVAILABLE:
            w_smm = torch.randn((n, k), device=DEVICE, dtype=torch.bfloat16)
            try:
                smm_us = benchmark_us(
                    scaled_mm_runner(x, w_smm),
                    args.warmup,
                    args.reps,
                )
            except Exception as e:
                warnings.warn(f"scaled_mm M={m} N={n} K={k}: {e}")

        custom_us = benchmark_us(
            lambda: runtime_kernel_blackwell.linear_nvfp4_sm100_no_quantization(
                x_q, x_sf, weight, weight_scale_interleaved, None, output
            ),
            args.warmup,
            args.reps,
        )
        path = "swapAB" if m <= SMALL_M_MAX else "1d2d"

        fi_cols = ""
        if use_flashinfer:
            # Use a fresh bf16 weight for flashinfer — same shape, avoids unpacking
            # the packed fp4 weight which has dtype quirks.
            w_fi = torch.randn((n, k), device=DEVICE, dtype=torch.bfloat16)

            try:
                fi_cutlass_us = benchmark_graph_us(
                    flashinfer_runner(x, w_fi, "cutlass"),
                    args.warmup,
                    args.reps,
                )
            except Exception as e:
                warnings.warn(f"flashinfer cutlass M={m} N={n} K={k}: {e}")
                fi_cutlass_us = float("nan")

            try:
                fi_trtllm_us = benchmark_graph_us(
                    flashinfer_runner(x, w_fi, "trtllm"),
                    args.warmup,
                    args.reps,
                )
            except Exception as e:
                warnings.warn(f"flashinfer trtllm M={m} N={n} K={k}: {e}")
                fi_trtllm_us = float("nan")

            fi_cols = (
                f" | fi_cutlass {fi_cutlass_us:8.1f} us"
                f" | fi_trtllm {fi_trtllm_us:8.1f} us"
            )

        smm_col = f" | scaled_mm {smm_us:8.1f} us" if use_scaled_mm else ""
        print(
            f"M={m:4d} N={n:4d} K={k:4d} | {path:6s} | custom {custom_us:8.1f} us"
            f"{smm_col}{fi_cols}"
        )


if __name__ == "__main__":
    main()
