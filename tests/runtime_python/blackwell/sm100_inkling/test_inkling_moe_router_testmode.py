"""Test-mode test for the Inkling MoE router (TASK_INKLING_MOE_ROUTER_SM100).

Production shape: 256 routed experts + 2 folded shared experts, top-6,
route_scale 8.0, logits row stride padded to 384 for the gate linear.
Checks:
  - routing weights vs softmax(logsigmoid(selected)) * route_scale * gscale
  - indices tensor [num_total, rows] holds k_idx+1 at selected experts, else 0
  - active-id compaction (set equality + count at active[num_total])
  - the first num_total logits columns are zeroed after the read
    (split-k gate linear reuse) and padded columns are untouched
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import inkling_moe_router_ref

NUM_ROUTED = 256
N_SHARED = 2
NUM_TOTAL = NUM_ROUTED + N_SHARED
TOPK = 6
K_OUT = TOPK + N_SHARED
GATE_PADDED = 384
ROUTE_SCALE = 8.0


def main():
    torch.manual_seed(0)
    device = "cuda"
    rows = 2

    logits = 2.0 * torch.randn(
        rows, GATE_PADDED, dtype=torch.bfloat16, device=device
    )
    logits_orig = logits.clone()
    bias = 0.05 * torch.randn(NUM_ROUTED, dtype=torch.float32, device=device)
    gscale = torch.tensor([1.7], dtype=torch.float32, device=device)
    weights = torch.full(
        (rows, K_OUT), -7.0, dtype=torch.float32, device=device
    )
    indices = torch.full(
        (NUM_TOTAL, rows), -7, dtype=torch.int32, device=device
    )
    active = torch.full((NUM_TOTAL + 1,), -7, dtype=torch.int32, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    pk = PersistentKernel(**params)

    logits_dt = pk.attach_input(logits, name="logits")
    bias_dt = pk.attach_input(bias, name="bias")
    gscale_dt = pk.attach_input(gscale, name="gscale")
    weights_dt = pk.attach_input(weights, name="weights")
    indices_dt = pk.attach_input(indices, name="indices")
    active_dt = pk.attach_input(active, name="active")

    pk.inkling_moe_router_layer(
        logits=logits_dt,
        bias=bias_dt,
        global_scale=gscale_dt,
        output=(weights_dt, indices_dt, active_dt),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
        route_scale=ROUTE_SCALE,
        n_shared=N_SHARED,
    )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ref_w, ref_sel = inkling_moe_router_ref(
        logits_orig, bias, gscale,
        topk=TOPK, n_shared=N_SHARED, route_scale=ROUTE_SCALE,
    )

    ok = True

    w_diff = (weights - ref_w).abs().max().item()
    print(f"weights max diff: {w_diff:.3e}")
    print(f"weights[0]: {weights[0].tolist()}")
    print(f"ref    [0]: {ref_w[0].tolist()}")
    try:
        torch.testing.assert_close(weights, ref_w, atol=1e-4, rtol=1e-4)
    except AssertionError as e:
        print(f"weights FAILED: {e}")
        ok = False

    exp_indices = torch.zeros(NUM_TOTAL, rows, dtype=torch.int32, device=device)
    for r in range(rows):
        for k in range(K_OUT):
            exp_indices[ref_sel[r, k], r] = k + 1
    if not torch.equal(indices, exp_indices):
        bad = (indices != exp_indices).nonzero()
        print(f"indices FAILED at {bad[:10].tolist()}")
        ok = False
    else:
        print("indices OK")

    count = int(active[NUM_TOTAL].item())
    exp_active = set(ref_sel.flatten().tolist())
    got_active = set(active[:count].tolist())
    if count != len(exp_active) or got_active != exp_active:
        print(f"active FAILED: count={count} (expected {len(exp_active)}), "
              f"missing={exp_active - got_active}, "
              f"extra={got_active - exp_active}")
        ok = False
    else:
        print(f"active OK ({count} experts)")

    if not torch.equal(
        logits[:, :NUM_TOTAL],
        torch.zeros(rows, NUM_TOTAL, dtype=torch.bfloat16, device=device),
    ):
        print("logits zeroing FAILED: first num_total columns not cleared")
        ok = False
    elif not torch.equal(logits[:, NUM_TOTAL:], logits_orig[:, NUM_TOTAL:]):
        print("logits zeroing FAILED: padded columns were modified")
        ok = False
    else:
        print("logits zeroing OK")

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("PASSED: inkling_moe_router test_mode produces correct output")


if __name__ == "__main__":
    main()
