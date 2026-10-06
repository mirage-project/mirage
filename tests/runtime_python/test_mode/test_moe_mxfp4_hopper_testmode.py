"""Hopper routed MXFP4 W13/W2 tests (packed E2M1 + E8M0 scales)."""

import pytest
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device="cuda",
    dtype=torch.float32,
)


def _decode(packed, scales):
    low = _E2M1[(packed & 0x0F).long()]
    high = _E2M1[((packed >> 4) & 0x0F).long()]
    values = torch.stack((low, high), dim=-1).flatten(-2)
    scale_values = torch.where(
        scales == 0,
        torch.zeros_like(scales, dtype=torch.float32),
        torch.pow(2.0, scales.float() - 127.0),
    )
    return (values.reshape(*values.shape[:-1], -1, 32) * scale_values.unsqueeze(-1)) \
        .reshape(*values.shape[:-1], -1).to(torch.bfloat16)


def _make_pk(batch_size):
    workers, schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=workers,
        num_local_schedulers=schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch_size,
        max_num_batched_requests=batch_size,
    )
    return PersistentKernel(**params)


def test_routed_mxfp4_w13_w2_hopper(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90/Hopper GPU")

    torch.manual_seed(42)
    batch, experts, topk, hidden, intermediate = 2, 2, 2, 64, 64
    device = "cuda"
    input_w13 = (torch.randn(batch, hidden, device=device) * 0.2).to(torch.bfloat16)
    w13_packed = torch.randint(
        0, 256, (experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8, device=device,
    )
    w13_scales = torch.randint(
        120, 135, (experts, 2 * intermediate, hidden // 32),
        dtype=torch.uint8, device=device,
    )
    w13 = _decode(w13_packed, w13_scales)

    input_w2 = (torch.randn(batch, topk, intermediate, device=device) * 0.2) \
        .to(torch.bfloat16)
    w2_packed = torch.randint(
        0, 256, (experts, hidden, intermediate // 2),
        dtype=torch.uint8, device=device,
    )
    w2_scales = torch.randint(
        120, 135, (experts, hidden, intermediate // 32),
        dtype=torch.uint8, device=device,
    )
    w2 = _decode(w2_packed, w2_scales)

    routing = torch.tensor([[1, 1], [2, 2]], dtype=torch.int32, device=device)
    mask = torch.tensor([0, 1, experts], dtype=torch.int32, device=device)
    w13_out = torch.zeros(batch, topk, 2 * intermediate,
                          dtype=torch.bfloat16, device=device)
    w2_out = torch.zeros(batch, topk, hidden,
                         dtype=torch.bfloat16, device=device)

    pk = _make_pk(batch)
    pk.moe_w13_mxfp4_layer(
        pk.attach_input(input_w13, "w13_input"),
        pk.attach_input(w13_packed, "w13_packed"),
        pk.attach_input(w13_scales, "w13_scale"),
        pk.attach_input(routing, "w13_routing"),
        pk.attach_input(mask, "w13_mask"),
        pk.attach_input(w13_out, "w13_output"),
        grid_dim=(5, 1, 1), block_dim=(256, 1, 1),
    )
    pk.moe_w2_mxfp4_layer(
        pk.attach_input(input_w2, "w2_input"),
        pk.attach_input(w2_packed, "w2_packed"),
        pk.attach_input(w2_scales, "w2_scale"),
        pk.attach_input(routing, "w2_routing"),
        pk.attach_input(mask, "w2_mask"),
        pk.attach_input(w2_out, "w2_output"),
        grid_dim=(4, 1, 1), block_dim=(256, 1, 1),
    )
    pk.compile(output_dir=str(tmp_path))
    pk()
    torch.cuda.synchronize()

    ref_w13 = torch.empty_like(w13_out)
    ref_w2 = torch.empty_like(w2_out)
    for token in range(batch):
        for expert in range(experts):
            slot = int(routing[expert, token].item()) - 1
            ref_w13[token, slot] = (
                input_w13[token].float() @ w13[expert].float().T
            ).to(torch.bfloat16)
            ref_w2[token, slot] = (
                input_w2[token, slot].float() @ w2[expert].float().T
            ).to(torch.bfloat16)
    torch.testing.assert_close(w13_out, ref_w13, rtol=0.02, atol=0.02)
    torch.testing.assert_close(w2_out, ref_w2, rtol=0.02, atol=0.02)
    pk.finalize()
