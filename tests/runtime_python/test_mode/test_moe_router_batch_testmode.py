"""
Test MoE topk+softmax routing at batch>1 via PersistentKernel test_mode.

Regression for #740: router must write top-k weights in (batch, topk)
row-major layout so moe_mul_sum_add can consume them.

Run:
    python tests/runtime_python/test_mode/test_moe_router_batch_testmode.py
"""

import os
import sys

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def _run_router(batch_size: int):
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    num_experts = 128
    topk = 8

    gate_logits = torch.randn(batch_size, num_experts, dtype=dtype, device=device)
    topk_w = torch.zeros(batch_size, topk, dtype=torch.float32, device=device)
    routing = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

    probs = F.softmax(gate_logits.float(), dim=-1)
    ref_w, ref_idx = probs.topk(topk, dim=-1)
    ref_w = (ref_w / ref_w.sum(dim=-1, keepdim=True)).float()
    ref_routing = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    for t in range(batch_size):
        for k in range(topk):
            ref_routing[ref_idx[t, k].item(), t] = k + 1

    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch_size,
        max_num_batched_requests=batch_size,
    )
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    pk = PersistentKernel(**params)

    pk.moe_topk_softmax_routing_layer(
        input=pk.attach_input(gate_logits, "gate"),
        output=(
            pk.attach_input(topk_w, "tw"),
            pk.attach_input(routing, "ri"),
            pk.attach_input(mask, "mask"),
        ),
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1),
    )

    outdir = os.path.join(os.path.dirname(__file__), f"_router_b{batch_size}")
    os.makedirs(outdir, exist_ok=True)
    pk.compile(output_dir=outdir)
    pk()
    torch.cuda.synchronize()

    w_err = (topk_w.float() - ref_w).abs().max().item()
    route_counts = [(routing[:, t] > 0).sum().item() for t in range(batch_size)]
    counts_ok = all(c == topk for c in route_counts)
    pk.finalize()
    return w_err, counts_ok


def main():
    for batch_size in (1, 2, 8, 16, 17):
        w_err, counts_ok = _run_router(batch_size)
        print(
            f"batch={batch_size}: weight_max_err={w_err:.6f} "
            f"topk_count_ok={counts_ok}"
        )
        if w_err > 1e-3 or not counts_ok:
            print(f"FAILED at batch={batch_size}")
            sys.exit(1)
    print("PASSED: MoE router (batch, topk) weight layout")


if __name__ == "__main__":
    main()
