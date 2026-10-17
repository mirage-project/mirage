"""Hopper routed MoE W13/W2 with per-expert biases (BF16 and MXFP4 weights)."""

import pytest
import torch

from test_moe_mxfp4_hopper_testmode import _decode, _make_pk

EXPERTS, TOPK, BATCH, HIDDEN, INTER = 8, 4, 8, 512, 256


def _routing(device):
    routing = torch.zeros(EXPERTS, BATCH, dtype=torch.int32, device=device)
    for token in range(BATCH):
        for slot in range(TOPK):
            routing[(token * TOPK + slot) % EXPERTS, token] = slot + 1
    active = [e for e in range(EXPERTS) if routing[e].any()]
    mask = torch.zeros(EXPERTS + 1, dtype=torch.int32, device=device)
    mask[:len(active)] = torch.tensor(active, dtype=torch.int32, device=device)
    mask[-1] = len(active)
    return routing, mask


def _weight(out_dim, red_dim, mxfp4, device):
    """Returns (layer weight args, the dense BF16 weight they represent)."""
    if not mxfp4:
        w = torch.randn(EXPERTS, out_dim, red_dim, device=device) * 0.05
        return (w.to(torch.bfloat16),), w.to(torch.bfloat16)
    packed = torch.randint(0, 256, (EXPERTS, out_dim, red_dim // 2),
                           dtype=torch.uint8, device=device)
    scales = torch.randint(118, 124, (EXPERTS, out_dim, red_dim // 32),
                           dtype=torch.uint8, device=device)
    return (packed, scales), _decode(packed, scales)


def _reference(x, weight, bias, routing):
    out = torch.empty(BATCH, TOPK, weight.shape[1], device=x.device)
    for token in range(BATCH):
        for expert in range(EXPERTS):
            slot = int(routing[expert, token].item()) - 1
            if slot < 0:
                continue
            row = x[token, slot] if x.ndim == 3 else x[token]
            out[token, slot] = (row.float() @ weight[expert].float().T
                                + bias[expert].float())
    return out


@pytest.mark.parametrize("mxfp4", [False, True], ids=["bf16", "mxfp4"])
def test_routed_moe_bias_hopper(tmp_path, mxfp4):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90/Hopper GPU")

    torch.manual_seed(0)
    device = "cuda"
    routing, mask = _routing(device)
    x13 = (torch.randn(BATCH, HIDDEN, device=device) * 0.1).to(torch.bfloat16)
    w13_args, w13 = _weight(2 * INTER, HIDDEN, mxfp4, device)
    b13 = torch.randn(EXPERTS, 2 * INTER, device=device).to(torch.bfloat16)
    x2 = (torch.randn(BATCH, TOPK, INTER, device=device) * 0.1).to(torch.bfloat16)
    w2_args, w2 = _weight(HIDDEN, INTER, mxfp4, device)
    b2 = torch.randn(EXPERTS, HIDDEN, device=device).to(torch.bfloat16)
    out13 = torch.zeros(BATCH, TOPK, 2 * INTER, dtype=torch.bfloat16, device=device)
    out2 = torch.zeros(BATCH, TOPK, HIDDEN, dtype=torch.bfloat16, device=device)

    pk = _make_pk(BATCH)
    routing_dt = pk.attach_input(routing, "routing")
    mask_dt = pk.attach_input(mask, "mask")
    w13_layer = pk.moe_w13_mxfp4_layer if mxfp4 else pk.moe_w13_linear_layer
    w2_layer = pk.moe_w2_mxfp4_layer if mxfp4 else pk.moe_w2_linear_layer
    # Several output tiles per expert, so each task reads its own bias slice.
    w13_layer(
        pk.attach_input(x13, "x13"),
        *(pk.attach_input(t, f"w13_{i}") for i, t in enumerate(w13_args)),
        routing_dt, mask_dt, pk.attach_input(out13, "out13"),
        grid_dim=(5, 2 * INTER // 128, 1), block_dim=(256, 1, 1),
        bias=pk.attach_input(b13, "b13"),
    )
    w2_layer(
        pk.attach_input(x2, "x2"),
        *(pk.attach_input(t, f"w2_{i}") for i, t in enumerate(w2_args)),
        routing_dt, mask_dt, pk.attach_input(out2, "out2"),
        grid_dim=(4, HIDDEN // 64, 1), block_dim=(256, 1, 1),
        bias=pk.attach_input(b2, "b2"),
    )
    pk.compile(output_dir=str(tmp_path))
    pk()
    torch.cuda.synchronize()

    torch.testing.assert_close(out13.float(), _reference(x13, w13, b13, routing),
                               rtol=0.02, atol=0.02)
    torch.testing.assert_close(out2.float(), _reference(x2, w2, b2, routing),
                               rtol=0.02, atol=0.02)
    pk.finalize()
