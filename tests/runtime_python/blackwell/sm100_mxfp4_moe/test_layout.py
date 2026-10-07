"""Packing contract for the GPT-OSS MXFP4 loader. No GPU, no checkpoint."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "python")))

from mirage.mpk.models.gpt_oss.mxfp4 import deinterleave_gate_up, unpack_mxfp4


def test_nibble_and_scale():
    # one row, K=32. byte 0 = 0x1A -> even -1, odd +0.5. scale 128 -> x2.
    blocks = torch.zeros(1, 16, dtype=torch.uint8)
    blocks[0, 0] = 0x1A
    scales = torch.tensor([[128]], dtype=torch.uint8)
    out = unpack_mxfp4(blocks, scales)
    assert out.shape == (1, 32)
    assert out[0, 0].item() == -2.0
    assert out[0, 1].item() == 1.0
    assert out[0, 2].item() == 0.0


def test_block_shape_matches_checkpoint():
    # gate_up blocks [E, 2I, K/32, 16] dequant to [E, 2I, K]
    e, rows, k = 2, 4, 64
    blocks = torch.randint(0, 256, (e, rows, k // 32, 16), dtype=torch.uint8)
    scales = torch.randint(120, 135, (e, rows, k // 32), dtype=torch.uint8)
    out = unpack_mxfp4(blocks, scales)
    assert out.shape == (e, rows, k)


def test_deinterleave_splits_gate_and_up():
    rows = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    out = deinterleave_gate_up(rows)
    expect = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.float32)
    assert torch.equal(out[0, :, 0], expect)


def test_120b_grid_slices():
    hidden, inter = 2880, 2880
    # w13 N = 2I divides the 128 MMA tile. w2 N = H does not; the slice is 64.
    assert math.gcd(2 * inter, 128) == 128
    assert (2 * inter) // 128 == 45
    assert math.gcd(hidden, 128) == 64
    assert hidden // 64 == 45


if __name__ == "__main__":
    test_nibble_and_scale()
    test_block_shape_matches_checkpoint()
    test_deinterleave_splits_gate_and_up()
    test_120b_grid_slices()
    print("PASS gpt-oss mxfp4 layout")
