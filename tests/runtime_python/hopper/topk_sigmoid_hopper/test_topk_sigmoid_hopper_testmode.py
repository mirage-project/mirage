"""moe_topk_sigmoid_routing_layer through the full MPK pipeline on Hopper.

On sm_90 the layer registers moe_topk_sigmoid_hopper. Two routers share one
test_mode PersistentKernel: the DeepSeek-V3 shape at 16 tokens (past the
sm100 router's 8-row limit) and an ungrouped 128-expert router. Checks the
top-k weights, the routing_indices matrix, the active expert list and the
zeroed logits against the PyTorch reference.

    python tests/runtime_python/hopper/topk_sigmoid_hopper/test_topk_sigmoid_hopper_testmode.py
"""

import os
import sys
import tempfile

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_reference import sigmoid_routing_reference  # noqa: E402

BATCH = 16
SCALE = 2.5
# (experts, groups, groups kept, experts per token)
CASES = [(256, 8, 4, 8), (128, 1, 1, 8)]


def main():
    assert torch.cuda.get_device_capability() == (9, 0), "needs an sm_90 GPU"
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0,
                  world_size=1, max_num_batched_tokens=BATCH,
                  max_num_batched_requests=BATCH, max_num_pages=BATCH)
    pk = PersistentKernel(**params)

    g = torch.Generator(device="cuda").manual_seed(0)
    runs = []
    for n, (experts, groups, topk_group, topk) in enumerate(CASES):
        logits = torch.randn(BATCH, experts, device="cuda",
                             dtype=torch.bfloat16, generator=g)
        bias = torch.randn(experts, device="cuda", generator=g) * 0.1
        weights = torch.zeros(BATCH, topk, device="cuda")
        routing = torch.full((experts, BATCH), 7, dtype=torch.int32,
                             device="cuda")
        active = torch.full((experts + 1,), 1234, dtype=torch.int32,
                            device="cuda")
        ref = sigmoid_routing_reference(logits, bias, groups, topk_group, topk,
                                        SCALE)
        pk.moe_topk_sigmoid_routing_layer(
            input=pk.attach_input(logits, name=f"logits{n}"),
            bias=pk.attach_input(bias, name=f"bias{n}"),
            output=(pk.attach_input(weights, name=f"weights{n}"),
                    pk.attach_input(routing, name=f"routing{n}"),
                    pk.attach_input(active, name=f"active{n}")),
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1), num_groups=groups,
            topk_group=topk_group, routed_scaling_factor=SCALE)
        runs.append((logits, weights, routing, active, ref))

    print("Compiling test kernel...")
    # Generated sources go to a temp dir; *.cu here is the tracked wrapper.
    pk.compile(output_dir=tempfile.mkdtemp(prefix="mpk_hopper_testmode_"))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    ok = True
    for (experts, groups, topk_group, topk), run in zip(CASES, runs):
        logits, weights, routing, active, (ref_w, ref_r, ref_used) = run
        count = int(active[-1])
        ids = ref_used.nonzero().flatten().to(torch.int32)
        checks = {
            "weights": torch.allclose(weights, ref_w, rtol=1e-5, atol=1e-5),
            "routing_indices": torch.equal(routing, ref_r),
            "active_expert_ids": count == ids.numel()
            and torch.equal(active[:count], ids)
            and bool((active[count:experts] == -1).all()),
            "logits zeroed": not bool(logits.any()),
        }
        passed = all(checks.values())
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] experts={experts} "
              f"groups={groups} topk_group={topk_group} topk={topk} "
              f"batch={BATCH}  {checks}")
    pk.finalize()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
