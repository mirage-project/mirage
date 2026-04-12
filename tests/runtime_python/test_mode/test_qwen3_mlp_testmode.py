"""
Test: Qwen3 dense MLP layers via PersistentKernel test_mode.

Tests the MLP sub-graph from demo/qwen3/demo.py (dense model, not MoE):
  1. linear_layer (gate+up fused):  input @ w_gatedup.T  -> mlp_mid
  2. silu_mul_layer:                SiLU(gate) * up       -> silu_mul_out
  3. linear_with_residual_layer:    silu_mul_out @ w_down.T + residual -> mlp_out

Two test functions:
  - test_gateup_only:          Just the gate+up linear layer
  - test_gateup_silu_down:     Full MLP pipeline (gate+up → silu_mul → down+residual)

Run:
    python tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def grid_for_linear(output_dim):
    """Compute grid_dim.x for linear layer, matching demo/qwen3/demo.py logic."""
    if output_dim % 96 == 0:
        return output_dim // 96
    elif output_dim % 64 == 0:
        return output_dim // 64
    else:
        assert False, f"Unsupported linear output_dim={output_dim}"


def test_gateup_only():
    """Test only the gate+up linear layer."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Qwen3-like config (smaller for fast compilation)
    batch_size = 8
    hidden_size = 4096        # K
    intermediate_size = 2048  # N/2; fused gate+up = 2*intermediate = 4096
    fused_outdim = 2 * intermediate_size

    print(f"\n{'='*60}")
    print(f"Test: Gate+Up linear only (dense Qwen3 MLP)")
    print(f"  B={batch_size}, hidden={hidden_size}, intermediate={intermediate_size}")

    # Weights: gate_proj [intermediate, hidden] + up_proj [intermediate, hidden]
    # Fused: w_gatedup [2*intermediate, hidden]
    w_gatedup = torch.randn(fused_outdim, hidden_size, dtype=dtype, device=device) * 0.01

    # Input: [batch, hidden]
    input_act = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    # Output: [batch, 2*intermediate]
    mlp_mid = torch.zeros(batch_size, fused_outdim, dtype=dtype, device=device)

    # PyTorch reference
    ref = (input_act.float() @ w_gatedup.float().T).to(dtype)

    # Build PersistentKernel
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    input_dt = pk.attach_input(input_act, name="input")
    w_dt = pk.attach_input(w_gatedup, name="w_gatedup")
    mlp_mid_dt = pk.attach_input(mlp_mid, name="mlp_mid")

    num_tasks = grid_for_linear(fused_outdim)
    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)

    pk.linear_layer(
        input=input_dt,
        weight=w_dt,
        output=mlp_mid_dt,
        grid_dim=(num_tasks, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nmlp_mid[0, :8]:   {mlp_mid[0, :8]}")
    print(f"reference[0, :8]: {ref[0, :8]}")

    max_diff = (mlp_mid.float() - ref.float()).abs().max().item()
    print(f"\nMax absolute diff: {max_diff:.6f}")

    if max_diff < 0.5:
        print("\nPASSED: gate+up linear layer produces correct output")
    else:
        print(f"\nFAILED: max diff {max_diff:.6f} exceeds 0.5 tolerance")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


def test_gateup_silu_down():
    """Test full MLP pipeline: gate+up linear → silu_mul → down+residual."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    hidden_size = 4096
    intermediate_size = 2048
    fused_outdim = 2 * intermediate_size

    print(f"\n{'='*60}")
    print(f"Test: Full MLP pipeline (gate+up → silu_mul → down+residual)")
    print(f"  B={batch_size}, hidden={hidden_size}, intermediate={intermediate_size}")

    # Weights: gate and up separately (will be shuffled)
    w_gate = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) * 0.01
    w_up = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) * 0.01
    w_down = torch.randn(hidden_size, intermediate_size, dtype=dtype, device=device) * 0.01

    # Input and residual
    input_act = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    # Intermediate and output tensors
    mlp_mid = torch.zeros(batch_size, fused_outdim, dtype=dtype, device=device)
    silu_mul_out = torch.zeros(batch_size, intermediate_size, dtype=dtype, device=device)
    mlp_out = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

    # PyTorch reference
    ref_gate = input_act.float() @ w_gate.float().T
    ref_up = input_act.float() @ w_up.float().T
    ref_silu = torch.nn.functional.silu(ref_gate) * ref_up
    ref_out = (ref_silu @ w_down.float().T + residual.float()).to(dtype)

    # Build PersistentKernel
    qo_indptr_buffer = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr_buffer[batch_size] = batch_size

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    params["meta_tensors"] = {"qo_indptr_buffer": qo_indptr_buffer}
    pk = PersistentKernel(**params)

    input_dt = pk.attach_input(input_act, name="input")
    w_gate_dt = pk.attach_input(w_gate, name="w_gate")
    w_up_dt = pk.attach_input(w_up, name="w_up")
    w_down_dt = pk.attach_input(w_down, name="w_down")
    residual_dt = pk.attach_input(residual, name="residual")
    mlp_mid_dt = pk.attach_input(mlp_mid, name="mlp_mid")
    silu_mul_dt = pk.attach_input(silu_mul_out, name="silu_mul_out")
    mlp_out_dt = pk.attach_input(mlp_out, name="mlp_out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    num_tasks_gatedup = grid_for_linear(fused_outdim)

    # shuffle_tensors interleaves gate and up weight rows
    w_gatedup_dt = pk.shuffle_tensors(
        inputs=[w_gate_dt, w_up_dt],
        shuffled_dim=0,
        num_groups=num_tasks_gatedup // 2,
        name="w_gatedup",
    )

    # Layer 1: Gate+Up linear
    pk.linear_layer(
        input=input_dt,
        weight=w_gatedup_dt,
        output=mlp_mid_dt,
        grid_dim=(num_tasks_gatedup, 1, 1),
        block_dim=block_dim,
    )

    # Layer 2: SiLU-Mul
    pk.silu_mul_layer(
        input=mlp_mid_dt,
        output=silu_mul_dt,
        grid_dim=(num_tasks_gatedup // 2, 1, 1),
        block_dim=block_dim,
    )

    # Layer 3: Down projection + residual
    pk.linear_with_residual_layer(
        input=silu_mul_dt,
        weight=w_down_dt,
        residual=residual_dt,
        output=mlp_out_dt,
        grid_dim=(hidden_size // 64, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nmlp_out[0, :8]:   {mlp_out[0, :8]}")
    print(f"reference[0, :8]: {ref_out[0, :8]}")

    max_diff = (mlp_out.float() - ref_out.float()).abs().max().item()
    print(f"\nMax absolute diff: {max_diff:.6f}")

    if max_diff < 1.0:
        print("\nPASSED: full MLP pipeline produces correct output")
    else:
        print(f"\nFAILED: max diff {max_diff:.6f} exceeds 1.0 tolerance")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


def test_gateup_silu():
    """Test gate+up linear → silu_mul (no down projection)."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    batch_size = 8
    hidden_size = 4096
    intermediate_size = 2048
    fused_outdim = 2 * intermediate_size

    print(f"\n{'='*60}")
    print(f"Test: Gate+Up linear + SiLU-Mul")
    print(f"  B={batch_size}, hidden={hidden_size}, intermediate={intermediate_size}")

    # Weights: gate_proj and up_proj separately (will be shuffled)
    w_gate = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) * 0.01
    w_up = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device) * 0.01

    # Input
    input_act = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)

    # Intermediate and output
    mlp_mid = torch.zeros(batch_size, fused_outdim, dtype=dtype, device=device)
    silu_mul_out = torch.zeros(batch_size, intermediate_size, dtype=dtype, device=device)

    # PyTorch reference: gate and up are computed from the SHUFFLED weight,
    # but since shuffle interleaves row groups, the final matmul result is the same —
    # the silu_mul kernel just reads gate/up from interleaved positions.
    # For the reference, compute gate and up projections directly:
    ref_gate = (input_act.float() @ w_gate.float().T)
    ref_up = (input_act.float() @ w_up.float().T)
    ref_silu = (torch.nn.functional.silu(ref_gate) * ref_up).to(dtype)
    torch.cuda.synchronize()

    # Build PersistentKernel
    qo_indptr_buffer = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr_buffer[batch_size] = batch_size

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    params["meta_tensors"] = {"qo_indptr_buffer": qo_indptr_buffer}
    pk = PersistentKernel(**params)

    input_dt = pk.attach_input(input_act, name="input")
    w_gate_dt = pk.attach_input(w_gate, name="w_gate")
    w_up_dt = pk.attach_input(w_up, name="w_up")
    mlp_mid_dt = pk.attach_input(mlp_mid, name="mlp_mid")
    silu_mul_dt = pk.attach_input(silu_mul_out, name="silu_mul_out")

    block_dim = (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)
    num_tasks = grid_for_linear(fused_outdim)

    # shuffle_tensors interleaves gate and up weight rows so that each CTA's
    # output slice contains matching gate/up pairs for silu_mul.
    # This matches demo/qwen3/demo.py's usage.
    w_gatedup_dt = pk.shuffle_tensors(
        inputs=[w_gate_dt, w_up_dt],
        shuffled_dim=0,
        num_groups=num_tasks // 2,
        name="w_gatedup",
    )

    # Layer 1: Gate+Up linear
    pk.linear_layer(
        input=input_dt,
        weight=w_gatedup_dt,
        output=mlp_mid_dt,
        grid_dim=(num_tasks, 1, 1),
        block_dim=block_dim,
    )

    # Layer 2: SiLU-Mul
    pk.silu_mul_layer(
        input=mlp_mid_dt,
        output=silu_mul_dt,
        grid_dim=(num_tasks // 2, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"\nsilu_mul_out[0, :8]:   {silu_mul_out[0, :8]}")
    print(f"reference[0, :8]:     {ref_silu[0, :8]}")

    max_diff = (silu_mul_out.float() - ref_silu.float()).abs().max().item()
    print(f"\nMax absolute diff: {max_diff:.6f}")

    if max_diff < 0.5:
        print("\nPASSED: gate+up + silu_mul produces correct output")
    else:
        print(f"\nFAILED: max diff {max_diff:.6f} exceeds 0.5 tolerance")
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_gateup_only()
    test_gateup_silu()
    test_gateup_silu_down()
