"""Hopper linear_layer with a bias row, at GPT-OSS QKV and router shapes."""

import pytest
import torch

from test_moe_mxfp4_hopper_testmode import _make_pk

BATCH, HIDDEN = 8, 2880


# (output columns, tasks): QKV is 64 q + 2 * 8 kv heads of 64; router is 128
# experts. 64 columns per task is the swapAB kernel's tile.
@pytest.mark.parametrize("out_dim, tasks", [(5120, 80), (128, 2)],
                         ids=["qkv", "router"])
def test_linear_bias_hopper(tmp_path, out_dim, tasks):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90/Hopper GPU")

    torch.manual_seed(0)
    device = "cuda"
    x = (torch.randn(BATCH, HIDDEN, device=device) * 0.05).to(torch.bfloat16)
    w = (torch.randn(out_dim, HIDDEN, device=device) * 0.05).to(torch.bfloat16)
    bias = torch.randn(1, out_dim, device=device).to(torch.bfloat16)
    out = torch.zeros(BATCH, out_dim, dtype=torch.bfloat16, device=device)

    pk = _make_pk(BATCH)
    pk.linear_layer(
        input=pk.attach_input(x, "x"),
        weight=pk.attach_input(w, "w"),
        output=pk.attach_input(out, "out"),
        grid_dim=(tasks, 1, 1),
        block_dim=(256, 1, 1),
        bias=pk.attach_input(bias, "bias"),
    )
    pk.compile(output_dir=str(tmp_path))
    pk()
    torch.cuda.synchronize()

    expected = x.float() @ w.float().T + bias.float()
    torch.testing.assert_close(out.float(), expected, rtol=0.01, atol=0.02)
    pk.finalize()
