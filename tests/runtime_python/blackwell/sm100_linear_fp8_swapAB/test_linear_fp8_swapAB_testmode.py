"""End-to-end test_mode coverage for the MPK-native FP8 swap-AB Linear kernel.

This test drives the full MPK compilation pipeline (Python layer →
register_task → codegen → nvcc → persistent kernel runtime) for shape
configurations representative of DeepSeek V3 dense FP8 linear layers.

Per-task output (= full N / grid_dim.x) must be a multiple of 128, and
batch ≤ 16. Each test below picks a `grid_x` so the per-task tile lands at
128, 256, or 512 — the kernel template instantiations covered by
register_linear_fp8_swapAB_sm100_task.

Run:
  CUDA_VISIBLE_DEVICES=<free-gpu> python tests/runtime_python/blackwell/sm100_linear_fp8_swapAB/test_linear_fp8_swapAB_testmode.py
"""

import os
import sys
import torch

# Make the common scale-layout helpers importable. We're at
# tests/runtime_python/blackwell/sm100_linear_fp8_swapAB/, common is sibling.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from sm100_fp8_scale_layout import (  # noqa: E402
    quantize_to_fp8_packed_ue8m0,
    dequant_from_packed_ue8m0,
)


# Per-compile artifacts (mpk_launcher_*.so, task_graph_*.json, test_rank0.cu)
# go to /tmp so they don't pile up in the source tree and don't hit ENOSPC on
# shared mounts. Override with MPK_TEST_OUTPUT_DIR if you need to inspect.
FOLDER = os.environ.get("MPK_TEST_OUTPUT_DIR", "/tmp/mpk_test_swapAB")
os.makedirs(FOLDER, exist_ok=True)


def _make_pk(batch_size: int) -> PersistentKernel:
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch_size,
        max_num_batched_requests=batch_size,
    )
    return PersistentKernel(**params)


def _quantize(x_bf16: torch.Tensor):
    """Quantize a 2D BF16 tensor to FP8 + UE8M0-packed scale (row-major)."""
    x_q, packed_scales = quantize_to_fp8_packed_ue8m0(x_bf16)
    return x_q.contiguous(), packed_scales.contiguous()


def _run_case(label: str, batch: int, full_n: int, k: int, grid_x: int,
              tol: float = 0.05):
    """Compile and run linear_fp8_swapAB_layer end-to-end for one shape.

    Compares against a dequantized FP8 reference (the kernel computes on
    the FP8-quantized inputs, so the BF16-input matmul is not the right
    target). Tolerance is generous because UE8M0 per-128-K block scales
    discard sub-block precision.
    """
    assert full_n % grid_x == 0, "full_n must be divisible by grid_x"
    per_task_n = full_n // grid_x
    assert per_task_n % 128 == 0, (
        f"per-task N = {per_task_n} not a multiple of 128 "
        f"(grid_x={grid_x}, full_n={full_n})")
    assert batch <= 16, "MPK swap-AB kernel currently caps BATCH ≤ 16"

    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  B={batch}  full_N={full_n}  K={k}  grid_x={grid_x}  "
          f"per-task N={per_task_n}")

    device = "cuda"
    torch.manual_seed(42)
    input_bf16 = (torch.randn(batch, k, dtype=torch.bfloat16, device=device)
                  * 0.1).contiguous()
    weight_bf16 = (torch.randn(full_n, k, dtype=torch.bfloat16, device=device)
                   / (k ** 0.5)).contiguous()
    input_fp8, input_scale = _quantize(input_bf16)
    weight_fp8, weight_scale = _quantize(weight_bf16)
    output = torch.zeros(batch, full_n, dtype=torch.bfloat16, device=device)

    # Reference matches what the kernel actually computes: dequantized FP8.
    input_dq = dequant_from_packed_ue8m0(input_fp8, input_scale)
    weight_dq = dequant_from_packed_ue8m0(weight_fp8, weight_scale)
    ref = (input_dq.float() @ weight_dq.float().T).to(torch.bfloat16)

    pk = _make_pk(batch)
    i_fp8 = pk.attach_input(input_fp8, name="input_fp8")
    i_sc = pk.attach_input(input_scale, name="input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="weight_scale")
    o = pk.attach_input(output, name="output")

    pk.linear_fp8_swapAB_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        output=o,
        grid_dim=(grid_x, 1, 1),
        block_dim=(256, 1, 1),
    )

    print("Compiling...")
    pk.compile(output_dir=FOLDER)
    print("Running...")
    pk()
    torch.cuda.synchronize()

    assert torch.isfinite(output).all(), "output has non-finite values"

    diff = (output.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"  output[0, :8]:    {output[0, :8].tolist()}")
    print(f"  reference[0, :8]: {ref[0, :8].tolist()}")
    print(f"  max-abs-error:    {max_abs:.4f}  (tol {tol})")
    print(f"  mean-abs-error:   {mean_abs:.6f}")

    pk.finalize()
    if max_abs > tol:
        print(f"  FAIL: {label}")
        return False
    print(f"  PASS: {label}")
    return True


# ============================================================================
# Test cases — DeepSeek V3 dense FP8 Linear layer shapes.
#
# Reference dimensions (DeepSeek V3 default config, see
# python/mirage/mpk/models/deepseek_v3/builder.py):
#   HIDDEN_SIZE          = 7168
#   Q_LORA_RANK          = 1536
#   KV_LORA_RANK         = 512
#   INTERMEDIATE_SIZE    = 18432  (dense MLP)
#   num_heads * v_head_dim = 128 * 128 = 16384  (o_proj raw)
#
# These tests use grid_x sized so per-task N is a multiple of 128 — the
# only configuration the MPK swap-AB kernel currently supports.
# ============================================================================


def test_smoke():
    """Single-CTA smoke (B=16, OUT=128, K=128)."""
    return _run_case("smoke (single CTA)",
                     batch=16, full_n=128, k=128, grid_x=1)


def test_q_a():
    """q_a projection: hidden_size → q_lora_rank.
    full_N=1536, K=7168. grid_x=12 → per-task N=128."""
    return _run_case("q_a (hidden→q_lora)",
                     batch=16, full_n=1536, k=7168, grid_x=12)


def test_q_b():
    """q_b family input: from q_lora_rank=1536. Use a representative
    full_N=1536 (multiple of 128). grid_x=12 → per-task N=128."""
    return _run_case("q_b (q_lora→head_dim)",
                     batch=16, full_n=1536, k=1536, grid_x=12)


def test_down_tp4():
    """down projection at TP=4: K = INTERMEDIATE_SIZE / 4 = 4608,
    full_N = HIDDEN_SIZE = 7168. grid_x=56 → per-task N=128."""
    return _run_case("down TP4 (intermediate/4 → hidden)",
                     batch=16, full_n=7168, k=4608, grid_x=56)


def test_o_proj():
    """o_proj: K = num_heads * v_head_dim = 16384, full_N = hidden = 7168.
    grid_x=56 → per-task N=128."""
    return _run_case("o_proj (heads→hidden)",
                     batch=16, full_n=7168, k=16384, grid_x=56)


def test_q_a_small_batch():
    """q_a equivalent at small decode batch B=4 (verify B varies cleanly)."""
    return _run_case("q_a @ B=4 (hidden→q_lora)",
                     batch=4, full_n=1536, k=7168, grid_x=12)


def test_q_a_wider_tile():
    """q_a with grid_x=6 → per-task N=256 (wider tile variant)."""
    return _run_case("q_a wider tile (per-task N=256)",
                     batch=16, full_n=1536, k=7168, grid_x=6)


# ----------------------------------------------------------------------------
# B=1 cases — single-token decode. Stresses the BATCH < MMA_N path (TMA box
# of 16 rows but gmem extent of 1 → 15 OOB-zero-filled rows in SMEM).
# ----------------------------------------------------------------------------


def test_smoke_b1():
    """Single-CTA single-token (B=1, OUT=128, K=128)."""
    return _run_case("smoke B=1 (single CTA, single token)",
                     batch=1, full_n=128, k=128, grid_x=1)


def test_q_a_b1():
    """q_a at single-token decode."""
    return _run_case("q_a @ B=1 (hidden→q_lora)",
                     batch=1, full_n=1536, k=7168, grid_x=12)


def test_down_tp4_b1():
    """down TP4 at single-token decode."""
    return _run_case("down TP4 @ B=1 (intermediate/4 → hidden)",
                     batch=1, full_n=7168, k=4608, grid_x=56)


def test_o_proj_b1():
    """o_proj at single-token decode (largest K)."""
    return _run_case("o_proj @ B=1 (heads→hidden)",
                     batch=1, full_n=7168, k=16384, grid_x=56)


def main():
    cases = [
        test_smoke,
        test_q_a,
        test_q_b,
        test_down_tp4,
        test_o_proj,
        test_q_a_small_batch,
        test_q_a_wider_tile,
        test_smoke_b1,
        test_q_a_b1,
        test_down_tp4_b1,
        test_o_proj_b1,
    ]
    results = [(c.__name__, c()) for c in cases]
    print(f"\n{'='*72}")
    print("Summary:")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n{n_pass}/{len(results)} passed")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
