"""End-to-end test_mode coverage for `linear_splitk_swapAB_fp8_layer`.

Drives the full MPK compilation pipeline (Python layer → register_task →
codegen → nvcc → persistent kernel runtime) with grid.y CTAs each
contributing a partial K-slice via TMA reduce-add. Output tensors are
pre-zeroed before each run.

Test cases mirror the K-bound DeepSeek V3 dense FP8 Linear layers:
  o_proj : full_N=7168, full_K=16384 — biggest split-K win
  q_a    : full_N=1536, full_K=7168
  down   : full_N=7168, full_K=4608  (TP4 intermediate)

Per-task M = full_N / grid.x must be a multiple of 128.
Per-task K = full_K / grid.y must be a multiple of 128.

Run:
  CUDA_VISIBLE_DEVICES=<free-gpu> python tests/runtime_python/blackwell/sm100_linear_splitk_fp8_swapAB/test_linear_splitk_fp8_swapAB_testmode.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "common"))

import mirage  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel  # noqa: E402
from sm100_fp8_scale_layout import (  # noqa: E402
    quantize_to_fp8_packed_ue8m0,
    dequant_from_packed_ue8m0,
)


# Per-compile artifacts go to /tmp to avoid ENOSPC on shared mounts.
FOLDER = os.environ.get("MPK_TEST_OUTPUT_DIR", "/tmp/mpk_test_splitk_swapAB")
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


def _quantize(x_bf16):
    x_q, packed = quantize_to_fp8_packed_ue8m0(x_bf16)
    return x_q.contiguous(), packed.contiguous()


def _run_case(label, batch, full_n, full_k, grid_x, grid_y, tol=0.05):
    assert full_n % grid_x == 0
    assert full_k % grid_y == 0
    per_task_n = full_n // grid_x
    per_task_k = full_k // grid_y
    assert per_task_n % 128 == 0, f"per-task N={per_task_n} not multiple of 128"
    assert per_task_k % 128 == 0, f"per-task K={per_task_k} not multiple of 128"

    print(f"\n{'='*72}")
    print(f"Test: {label}")
    print(f"  B={batch}  full_N={full_n}  full_K={full_k}  "
          f"grid=({grid_x}, {grid_y})  per-task N={per_task_n}  K={per_task_k}")

    device = "cuda"
    torch.manual_seed(42)
    input_bf16 = (torch.randn(batch, full_k, dtype=torch.bfloat16, device=device)
                  * 0.1).contiguous()
    weight_bf16 = (torch.randn(full_n, full_k, dtype=torch.bfloat16, device=device)
                   / (full_k ** 0.5)).contiguous()
    input_fp8, input_scale = _quantize(input_bf16)
    weight_fp8, weight_scale = _quantize(weight_bf16)

    # *** Output MUST be zero-initialized — kernel reduce-adds. ***
    output = torch.zeros(batch, full_n, dtype=torch.bfloat16, device=device)

    # Reference: dequantized FP8 matmul.
    input_dq = dequant_from_packed_ue8m0(input_fp8, input_scale)
    weight_dq = dequant_from_packed_ue8m0(weight_fp8, weight_scale)
    ref = (input_dq.float() @ weight_dq.float().T).to(torch.bfloat16)

    pk = _make_pk(batch)
    i_fp8 = pk.attach_input(input_fp8, name="input_fp8")
    i_sc = pk.attach_input(input_scale, name="input_scale")
    w_fp8 = pk.attach_input(weight_fp8, name="weight_fp8")
    w_sc = pk.attach_input(weight_scale, name="weight_scale")
    o = pk.attach_input(output, name="output")

    pk.linear_splitk_swapAB_fp8_layer(
        input_fp8=i_fp8, input_scale=i_sc,
        weight_fp8=w_fp8, weight_scale=w_sc,
        output=o,
        grid_dim=(grid_x, grid_y, 1),
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
# Test cases (DeepSeek V3 typical configurations)
# ============================================================================


def test_o_proj_split4():
    """o_proj: K=16384 split 4-way → per-task K=4096."""
    return _run_case("o_proj split_k=4 (heads→hidden)",
                     batch=16, full_n=7168, full_k=16384, grid_x=56, grid_y=4)


def test_o_proj_split4_b1():
    """o_proj at single-token decode."""
    return _run_case("o_proj @ B=1 split_k=4",
                     batch=1, full_n=7168, full_k=16384, grid_x=56, grid_y=4)


def test_q_a_split2():
    """q_a: K=7168 split 2-way → per-task K=3584."""
    return _run_case("q_a split_k=2 (hidden→q_lora)",
                     batch=16, full_n=1536, full_k=7168, grid_x=12, grid_y=2)


def test_down_split3():
    """down at TP4: K=4608 split 3-way → per-task K=1536 (multiple of 512).
    K=4608/512=9, so split_k must divide 9 — splits ∈ {1, 3, 9}."""
    return _run_case("down TP4 split_k=3 (intermediate/4 → hidden)",
                     batch=16, full_n=7168, full_k=4608, grid_x=56, grid_y=3)


def test_smoke_split2():
    """Smallest shape that satisfies K_per_task % 512 == 0: full_K=1024 split 2."""
    return _run_case("smoke split_k=2 (full_K=1024)",
                     batch=16, full_n=128, full_k=1024, grid_x=1, grid_y=2)


def main():
    cases = [
        test_smoke_split2,
        test_q_a_split2,
        test_down_split3,
        test_o_proj_split4,
        test_o_proj_split4_b1,
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
