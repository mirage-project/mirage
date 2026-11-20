"""CPU-only checks for padded, tile-major GPT-OSS MXFP4 expert banks."""

import torch

from demo.gpt_oss_static.mxfp4 import (
    MXFP4_NEUTRAL_E8M0,
    prepare_gpt_oss_expert_bank,
)


def _patterns(rows, groups, value_offset):
    row = torch.arange(rows, dtype=torch.int64).view(1, rows, 1, 1)
    group = torch.arange(groups, dtype=torch.int64).view(1, 1, groups, 1)
    byte = torch.arange(16, dtype=torch.int64).view(1, 1, 1, 16)
    return ((row * 29 + group * 13 + byte * 7 + value_offset) % 255 + 1).to(torch.uint8)


def _tcgen05_sf_offset(row, group):
    """Byte offset in one 128-row, four-K-group TCGEN05 scale chunk."""
    return (row % 32) * 16 + (row // 32) * 4 + group


def test_prepare_gptoss_bank_builds_tile_major_tma_order_without_mutation():
    hidden, padded = 192, 256
    groups, padded_groups = hidden // 32, padded // 32
    w13 = _patterns(2 * hidden, groups, 1)
    s13 = ((torch.arange(2 * hidden * groups).reshape(1, 2 * hidden, groups) * 3) % 255).to(torch.uint8)
    b13 = torch.arange(2 * hidden, dtype=torch.float32).reshape(1, -1) + 0.25
    w2 = _patterns(hidden, groups, 11)
    s2 = ((torch.arange(hidden * groups).reshape(1, hidden, groups) * 5 + 1) % 255).to(torch.uint8)
    b2 = torch.arange(hidden, dtype=torch.bfloat16).reshape(1, -1)
    before = tuple(t.clone() for t in (w13, s13, b13, w2, s2, b2))

    got_w13, got_s13, got_w2, got_s2, got_b13, got_b2 = prepare_gpt_oss_expert_bank(
        w13, s13, b13, w2, s2, b2, padded_width=padded
    )

    assert got_w13.shape == (1, 4, 2, 8192)
    assert got_s13.shape == (1, 4, 2, 512)
    assert got_w2.shape == (1, 2, 2, 8192)
    assert got_s2.shape == (1, 2, 2, 512)
    assert got_b13.shape == (1, 2 * padded)
    assert got_b2.shape == (1, padded)
    assert all(t.is_contiguous() for t in (got_w13, got_s13, got_w2, got_s2, got_b13, got_b2))

    # Physical tile index follows expert, output tile, then K tile. Within a
    # tile the weight bytes are [128 output rows][64 packed bytes]. Scales use
    # the TCGEN05 32-row interleave rather than a plain [128, 4] byte view.
    for out_tile in range(4):
        for kt in range(2):
            wt = got_w13[0, out_tile, kt].view(128, 64)
            sf = got_s13[0, out_tile, kt]
            for r in range(128):
                logical_row = out_tile * 64 + (r if r < 64 else r - 64)
                source_row = 2 * logical_row + (r >= 64)
                for kg in range(4):
                    source_group = kt * 4 + kg
                    if logical_row < hidden and source_group < groups:
                        torch.testing.assert_close(
                            wt[r, kg * 16 : (kg + 1) * 16],
                            w13[0, source_row, source_group],
                        )
                        assert sf[_tcgen05_sf_offset(r, kg)] == s13[0, source_row, source_group]
                    else:
                        assert torch.count_nonzero(wt[r, kg * 16 : (kg + 1) * 16]) == 0
                        assert sf[_tcgen05_sf_offset(r, kg)] == MXFP4_NEUTRAL_E8M0

    # W2 output tile order is unchanged; the K tile is the inner tile index.
    for out_tile in range(2):
        for kt in range(2):
            wt = got_w2[0, out_tile, kt].view(128, 64)
            sf = got_s2[0, out_tile, kt]
            for r in range(128):
                logical_row = out_tile * 128 + r
                for kg in range(4):
                    source_group = kt * 4 + kg
                    if logical_row < hidden and source_group < groups:
                        torch.testing.assert_close(
                            wt[r, kg * 16 : (kg + 1) * 16],
                            w2[0, logical_row, source_group],
                        )
                        assert sf[_tcgen05_sf_offset(r, kg)] == s2[0, logical_row, source_group]
                    else:
                        assert torch.count_nonzero(wt[r, kg * 16 : (kg + 1) * 16]) == 0
                        assert sf[_tcgen05_sf_offset(r, kg)] == MXFP4_NEUTRAL_E8M0

    # Bias is row-major, with independent W13 gate/up padding.
    torch.testing.assert_close(got_b13[0, :hidden], b13[0, 0::2].to(torch.bfloat16))
    assert torch.count_nonzero(got_b13[0, hidden:padded]) == 0
    torch.testing.assert_close(got_b13[0, padded : padded + hidden], b13[0, 1::2].to(torch.bfloat16))
    assert torch.count_nonzero(got_b13[0, padded + hidden :]) == 0
    torch.testing.assert_close(got_b2[0, :hidden], b2[0])
    assert torch.count_nonzero(got_b2[0, hidden:]) == 0

    for actual, snapshot in zip((w13, s13, b13, w2, s2, b2), before):
        torch.testing.assert_close(actual, snapshot)
