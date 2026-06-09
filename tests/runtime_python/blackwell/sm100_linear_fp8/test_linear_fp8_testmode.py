"""Test: linear_fp8_layer (SM100) via PersistentKernel test_mode at DSV3 shapes.

Builds a single FP8 block-scaled linear layer through the full MPK
compilation pipeline, runs it once, and validates against ``linear_fp8_ref``
from ``pytorch_reference.py``.

This is the decode-path FP8 linear (bs <= 16). It is exercised across the
union-of-axes (tp, bs) matrix at the DeepSeek-V3 dense-MLP gate_up projection
shape (N = 2*INTERMEDIATE_SIZE / tp, K = HIDDEN_SIZE = 7168) -- a projection
whose output dim N is tensor-parallel-sharded by world_size.

Run:
    python tests/runtime_python/blackwell/sm100_linear_fp8/test_linear_fp8_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from sm100_fp8_scale_layout import (  # noqa: E402
    aligned_scale_outer_dim,
    packed_scale_k_for_reduction_size,
    quantize_to_fp8_deepgemm_style,
)
from pytorch_reference import linear_fp8_ref  # noqa: E402

# ---------------------------------------------------------------------------
# DeepSeek-V3 dimensions (see python/mirage/mpk/models/deepseek_v3/builder.py)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 7168          # K for the dense-MLP gate_up projection (not TP-sharded)
INTERMEDIATE_SIZE = 18432   # dense MLP intermediate; gate_up N = 2*INTERMEDIATE/tp


def _gate_up_n(tp: int) -> int:
    """gate_up output dim N for a given tensor-parallel degree.

    Mirrors builder: ``self.intermediate_size = INTERMEDIATE_SIZE // world_size``
    and the fused gate+up GEMM emits ``2 * self.intermediate_size`` columns.
    """
    assert (2 * INTERMEDIATE_SIZE) % tp == 0
    return (2 * INTERMEDIATE_SIZE) // tp


def _pick_grid_x(output_size: int, num_workers: int) -> int:
    """Mirror builder._fp8_linear_grid_x: largest divisor of N/128 that fits
    in one worker wave (per-task N stays a multiple of 128)."""
    assert output_size % 128 == 0
    max_n_tiles = output_size // 128
    if max_n_tiles <= num_workers:
        return max_n_tiles
    best = 1
    i = 1
    while i * i <= max_n_tiles:
        if max_n_tiles % i == 0:
            if i <= num_workers:
                best = max(best, i)
            other = max_n_tiles // i
            if other <= num_workers:
                best = max(best, other)
        i += 1
    if best * 4 < num_workers:
        return max_n_tiles
    return best


def _input_scale_for_mpk(input_bf16):
    """Quantize ``input_bf16`` to FP8 + scales in the layout the MPK
    linear_fp8_sm100 task expects for the input (SFA).

    The runtime TMA descriptor for SFA reads the input-scale DTensor with
    logical shape ``(packed_k, aligned_batch)`` row-major contiguous (matches
    builder ``_fp8_buffers_for_reduction``: ``scale_buf`` is
    ``(packed_k, aligned_batch)`` uint32). ``quantize_to_fp8_deepgemm_style``
    produces logical shape ``(batch, packed_k)`` with strides
    ``(1, aligned_batch)``; its underlying storage already has the desired
    ``(packed_k, aligned_batch)`` row-major contiguous layout, so we copy it
    through a fresh contiguous tensor.
    """
    batch, reduction = input_bf16.shape
    packed_k = packed_scale_k_for_reduction_size(reduction)
    aligned_batch = aligned_scale_outer_dim(batch)

    x_fp8, x_scale_strided = quantize_to_fp8_deepgemm_style(input_bf16)

    scale_packed = torch.zeros(
        (packed_k, aligned_batch), dtype=torch.uint32, device=input_bf16.device
    )
    # x_scale_strided[b, k] is the packed scale for batch row b, packed-k k.
    # We need scale_packed[k, b] = x_scale_strided[b, k].
    scale_packed[:, :batch] = x_scale_strided.t().contiguous()
    return x_fp8, scale_packed


def _input_scale_dequant_view(scale_packed, batch):
    """Reinterpret a ``(packed_k, aligned_batch)`` row-major packed scale
    tensor as a ``(batch, packed_k)`` strided tensor compatible with the
    common ``dequant_from_packed_ue8m0`` helper (deepgemm col-major)."""
    packed_k = scale_packed.shape[0]
    aligned_batch = scale_packed.shape[1]
    return torch.as_strided(
        scale_packed, size=(batch, packed_k), stride=(1, aligned_batch)
    )


def _cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return (a @ b / denom).item()


def _rel_mean(out, ref):
    out = out.float()
    ref = ref.float()
    denom = ref.abs().mean().item()
    if denom == 0:
        return 0.0
    return (out - ref).abs().mean().item() / denom


def _run_case(tp: int, bs: int) -> bool:
    """Compile + run linear_fp8_layer end-to-end at the DSV3 gate_up shape
    for tensor-parallel degree ``tp`` (shards N) and batch ``bs`` (decode)."""
    assert bs <= 16, "linear_fp8 decode kernel caps batch <= 16"
    device = "cuda"
    output_size = _gate_up_n(tp)        # TP-sharded N
    reduction_size = HIDDEN_SIZE        # K (not sharded)

    print(f"\n{'='*72}")
    print(f"linear_fp8  tp={tp} bs={bs}  N={output_size} K={reduction_size}")

    g = torch.Generator(device=device).manual_seed(1234 + tp * 17 + bs)
    x_bf16 = (
        torch.randn((bs, reduction_size), device=device, dtype=torch.bfloat16,
                    generator=g) * 0.1
    )
    w_bf16 = (
        torch.randn((output_size, reduction_size), device=device,
                    dtype=torch.bfloat16, generator=g)
        / (reduction_size ** 0.5)
    )

    x_fp8, x_scale_packed = _input_scale_for_mpk(x_bf16)
    # weight_scale: deepgemm col-major (logical (N, packed_k), stride (1, aligned_N))
    w_fp8, w_scale_strided = quantize_to_fp8_deepgemm_style(w_bf16)

    output = torch.zeros((bs, output_size), device=device, dtype=torch.bfloat16)

    x_scale_ref_view = _input_scale_dequant_view(x_scale_packed, bs)
    ref = linear_fp8_ref(x_fp8, x_scale_ref_view, w_fp8, w_scale_strided)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    grid_x = _pick_grid_x(output_size, num_workers)
    assert (output_size // grid_x) % 128 == 0
    print(f"  num_workers={num_workers} grid_x={grid_x} "
          f"per-task N={output_size // grid_x}")

    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    # TP is a SHAPE selector only: world_size=1 (single physical GPU), the
    # per-rank-sharded N is passed directly (no NVSHMEM).
    params["world_size"] = 1
    params["max_num_batched_tokens"] = bs
    params["max_num_batched_requests"] = bs
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x_fp8, name="input_fp8")
    xs_dt = pk.attach_input(x_scale_packed, name="input_scale")
    w_dt = pk.attach_input(w_fp8, name="weight_fp8")
    ws_dt = pk.attach_input(w_scale_strided, name="weight_scale")
    out_dt = pk.attach_input(output, name="output")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    pk.linear_fp8_layer(
        input_fp8=x_dt,
        input_scale=xs_dt,
        weight_fp8=w_dt,
        weight_scale=ws_dt,
        output=out_dt,
        grid_dim=(grid_x, 1, 1),
        block_dim=block_dim,
    )

    print("  Compiling...")
    pk.compile(output_dir=THIS_DIR)
    print("  Running...")
    pk()
    torch.cuda.synchronize()

    finite = torch.isfinite(output).all().item()
    max_diff = (output.float() - ref.float()).abs().max().item()
    cos = _cosine(output, ref)
    rel = _rel_mean(output, ref)
    print(f"  output[0, :6]:    {output[0, :6].tolist()}")
    print(f"  reference[0, :6]: {ref[0, :6].tolist()}")
    print(f"  finite={finite} max_abs_diff={max_diff:.6f} "
          f"cosine={cos:.6f} rel_mean={rel*100:.4f}%")

    pk.finalize()

    # FP8 block-scaled tolerance: cosine > 0.99 OR relative mean <= 5%.
    ok = finite and (cos > 0.99 or rel <= 0.05)
    print(f"  {'PASS' if ok else 'FAIL'}: linear_fp8 tp={tp} bs={bs}")
    return ok


# union-of-axes (tp, bs) matrix, bs capped at <= 16 (decode-only):
#   {tp=1} x {bs=1,2,4,8,16} U {bs=16} x {tp=2,4,8} U {tp=8, bs=1}
MATRIX = (
    [(1, bs) for bs in (1, 2, 4, 8, 16)]
    + [(tp, 16) for tp in (2, 4, 8)]
    + [(8, 1)]
)


def test_linear_fp8_testmode():
    results = [((tp, bs), _run_case(tp, bs)) for (tp, bs) in MATRIX]
    print(f"\n{'='*72}\nSummary (linear_fp8):")
    for (tp, bs), ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  tp={tp} bs={bs}")
    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n{n_pass}/{len(results)} passed")
    assert n_pass == len(results), "some linear_fp8 configs failed"


if __name__ == "__main__":
    test_linear_fp8_testmode()
