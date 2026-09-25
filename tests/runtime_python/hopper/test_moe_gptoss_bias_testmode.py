"""SM90 coverage for per-expert GPT-OSS W13 and W2 biases."""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel

NUM_EXPERTS = 8
NUM_TOPK = 4
BATCH = 8
HIDDEN = 512
INTER = 256


def make_routing(device):
    routing = torch.zeros(NUM_EXPERTS, BATCH, dtype=torch.int32, device=device)
    for token in range(BATCH):
        for slot in range(NUM_TOPK):
            routing[(token * NUM_TOPK + slot) % NUM_EXPERTS, token] = slot + 1
    active = [e for e in range(NUM_EXPERTS) if routing[e].any()]
    mask = torch.zeros(NUM_EXPERTS + 1, dtype=torch.int32, device=device)
    mask[:len(active)] = torch.tensor(active, dtype=torch.int32, device=device)
    mask[-1] = len(active)
    return routing, mask


def reference(x, weight, bias, routing):
    result = torch.zeros(
        BATCH, NUM_TOPK, weight.shape[1], dtype=torch.float32, device=x.device
    )
    for token in range(BATCH):
        for slot in range(NUM_TOPK):
            expert = int(torch.where(routing[:, token] == slot + 1)[0].item())
            row = x[token, slot] if x.ndim == 3 else x[token]
            result[token, slot] = row.float() @ weight[expert].float().T
            result[token, slot] += bias[expert].float()
    return result


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    workers, schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=workers,
        num_local_schedulers=schedulers,
        max_num_batched_tokens=BATCH,
        max_num_batched_requests=1,
    )
    pk = PersistentKernel(**params)
    if pk.target_cc != 90:
        raise RuntimeError(f"This test requires SM90, got SM{pk.target_cc}")

    routing, mask = make_routing(device)
    routing_dt = pk.attach_input(routing, name="routing")
    mask_dt = pk.attach_input(mask, name="mask")

    x = torch.randn(BATCH, HIDDEN, dtype=dtype, device=device) * 0.1
    w13 = torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, dtype=dtype,
                      device=device) * 0.05
    b13 = torch.randn(NUM_EXPERTS, 2 * INTER, dtype=dtype, device=device)
    w13_out = torch.zeros(BATCH, NUM_TOPK, 2 * INTER, dtype=dtype,
                          device=device)
    pk.moe_w13_linear_layer(
        input=pk.attach_input(x, name="x"),
        weight=pk.attach_input(w13, name="w13"),
        moe_routing_indices=routing_dt,
        moe_mask=mask_dt,
        output=pk.attach_input(w13_out, name="w13_out"),
        grid_dim=(10, (2 * INTER) // 128, 1),
        block_dim=(256, 1, 1),
        bias=pk.attach_input(b13, name="b13"),
    )

    act = torch.randn(BATCH, NUM_TOPK, INTER, dtype=dtype,
                      device=device) * 0.1
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, dtype=dtype,
                     device=device) * 0.05
    b2 = torch.randn(NUM_EXPERTS, HIDDEN, dtype=dtype, device=device)
    w2_out = torch.zeros(BATCH, NUM_TOPK, HIDDEN, dtype=dtype, device=device)
    pk.moe_w2_linear_layer(
        input=pk.attach_input(act, name="act"),
        weight=pk.attach_input(w2, name="w2"),
        moe_routing_indices=routing_dt,
        moe_mask=mask_dt,
        output=pk.attach_input(w2_out, name="w2_out"),
        grid_dim=(8, HIDDEN // 64, 1),
        block_dim=(256, 1, 1),
        bias=pk.attach_input(b2, name="b2"),
    )

    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    pk()
    torch.cuda.synchronize()

    ok = True
    for name, got, expected in (
        ("w13", w13_out, reference(x, w13, b13, routing)),
        ("w2", w2_out, reference(act, w2, b2, routing)),
    ):
        diff = (got.float() - expected).abs().max().item()
        tolerance = max(0.02, 0.01 * expected.abs().max().item())
        print(f"[{name}] max |kernel - reference| = {diff:.4f} "
              f"(tol {tolerance:.4f})")
        if diff >= tolerance:
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("PASSED: SM90 W13 and W2 include per-expert biases")


if __name__ == "__main__":
    main()
