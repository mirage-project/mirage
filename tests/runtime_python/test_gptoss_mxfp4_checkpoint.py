"""CPU-only tests for GPT-OSS packed MXFP4 expert checkpoint layout."""

import torch

from demo.gpt_oss_static.mxfp4 import convert_expert_weight


def _row_sentinels(rows: int, width: int) -> torch.Tensor:
    """Give each row and column a distinguishable, non-symmetric pattern."""
    r = torch.arange(rows, dtype=torch.int64).view(1, rows, 1)
    c = torch.arange(width, dtype=torch.int64).view(1, 1, width)
    return ((r * 37 + c * 11 + (r % 7) * c) % 251).to(torch.uint8)


def test_gptoss_gate_up_deinterleaves_blocks_scales_and_bias_together():
    # GPT-OSS-120B has H=I=2880. The checkpoint interleaves gate/up output
    # rows, and stores each MXFP4 row as 90 groups of 32 values (16 packed
    # bytes per group). One expert keeps this realistic shape CPU-friendly.
    hidden = 2880
    rows = 2 * hidden
    block_groups = hidden // 32
    blocks = _row_sentinels(rows, block_groups * 16).reshape(
        1, rows, block_groups, 16
    )
    scales = _row_sentinels(rows, block_groups)
    bias = torch.arange(rows, dtype=torch.float32).reshape(1, rows) + 0.25

    got_blocks, got_scales, got_bias = convert_expert_weight(
        blocks, scales, bias, deinterleave=True
    )

    expected_order = torch.cat(
        (torch.arange(0, rows, 2), torch.arange(1, rows, 2))
    )
    assert got_blocks.shape == (1, rows, hidden // 2)
    assert got_scales.shape == (1, rows, block_groups)
    assert got_bias.dtype == torch.bfloat16
    torch.testing.assert_close(got_blocks, blocks.reshape(1, rows, -1)[:, expected_order])
    torch.testing.assert_close(got_scales, scales[:, expected_order])
    torch.testing.assert_close(got_bias, bias.to(torch.bfloat16)[:, expected_order])
    assert got_blocks.is_contiguous()
    assert got_scales.is_contiguous()
    assert got_bias.is_contiguous()


def test_gptoss_down_projection_is_not_deinterleaved():
    # Down-projection rows are output hidden dimensions, not paired gate/up
    # rows. Preserve the checkpoint order for both packed values and scales.
    hidden = 2880
    block_groups = hidden // 32
    blocks = _row_sentinels(hidden, block_groups * 16).reshape(
        1, hidden, block_groups, 16
    )
    scales = _row_sentinels(hidden, block_groups)
    bias = torch.arange(hidden, dtype=torch.float32).reshape(1, hidden)

    got_blocks, got_scales, got_bias = convert_expert_weight(
        blocks, scales, bias, deinterleave=False
    )

    torch.testing.assert_close(got_blocks, blocks.reshape(1, hidden, -1))
    torch.testing.assert_close(got_scales, scales)
    torch.testing.assert_close(got_bias, bias.to(torch.bfloat16))


def test_small_gate_up_layout_exposes_even_then_odd_row_mapping():
    # A tiny asymmetric case makes the ordering contract easy to inspect and
    # guards against accidentally swapping pairs or interleaving columns.
    blocks = torch.stack(
        [torch.arange(base, base + 16, dtype=torch.uint8) for base in (10, 30, 50, 70)]
    ).unsqueeze(0)
    scales = torch.tensor([[[1], [2], [3], [4]]], dtype=torch.uint8)
    bias = torch.tensor([[100.0, 200.0, 300.0, 400.0]])

    got_blocks, got_scales, got_bias = convert_expert_weight(
        blocks, scales, bias, deinterleave=True
    )

    torch.testing.assert_close(
        got_blocks, blocks[:, [0, 2, 1, 3]]
    )
    torch.testing.assert_close(got_scales, torch.tensor([[[1], [3], [2], [4]]], dtype=torch.uint8))
    torch.testing.assert_close(
        got_bias, torch.tensor([[100.0, 300.0, 200.0, 400.0]], dtype=torch.bfloat16)
    )
