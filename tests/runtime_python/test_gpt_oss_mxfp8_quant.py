"""CPU-side contract tests for the GPT-OSS MXFP8 activation path."""

import math
import struct
from types import MethodType, SimpleNamespace

import mirage.mpk.static_megakernel as static_mk
import pytest
import torch
from mirage.core import bfloat16, float16, uint8


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.num_dims = len(shape)

    def dim(self, axis):
        return self.shape[axis]


def _declare_quant(tokens, dtype=float16):
    graph = object.__new__(static_mk.StaticMegakernel)
    graph.__finalized__ = True
    graph.tokens = None
    recorded = {}

    def add_node(self, *args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        recorded["out_bytes"] = kwargs["out_buffer"](None)[0]

    graph._add_node = MethodType(add_node, graph)
    graph._tensor_slot = MethodType(lambda self, tensor: 17, graph)
    graph._dtypes = MethodType(lambda self, layer, **tensors: None, graph)

    import mirage.mpk.compiler as compiler

    old_graph_nodes, old_producer = compiler.graph_nodes, compiler.producer_of
    compiler.graph_nodes = lambda _: []
    compiler.producer_of = lambda _tensor, _nodes: None
    try:
        h2 = _Tensor((tokens, 2880), dtype)
        q = _Tensor((tokens, 2944), uint8)
        graph.gpt_oss_mxfp8_quant_layer(h2, q)
    finally:
        compiler.graph_nodes, compiler.producer_of = old_graph_nodes, old_producer
    return recorded


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _ceil_log2_bits(value):
    bits = struct.unpack("<I", struct.pack("<f", _f32(value)))[0]
    return ((bits >> 23) & 0xFF) - 127 + int((bits & 0x7FFFFF) != 0)


def _cpu_reference(h2, tokens):
    """Return q bytes and the 23 packed 512-byte K32 scale chunks."""
    pad = 2944
    q = bytearray(tokens * pad)
    chunks = bytearray((pad // 128) * 512)
    inv448 = _f32(1.0 / 448.0)
    for token in range(tokens):
        for start in range(0, pad, 32):
            values = [h2[token][col] if col < 2880 else 0.0 for col in range(start, start + 32)]
            amax = max((abs(value) for value in values), default=0.0)
            ratio = _f32(max(amax, 1.0e-30) * inv448)
            exponent = max(-127, min(127, _ceil_log2_bits(ratio)))
            scale = _f32(math.ldexp(1.0, -exponent))
            quantized = torch.tensor([_f32(value * scale) for value in values], dtype=torch.float32)
            quantized = quantized.to(torch.float8_e4m3fn).view(torch.uint8).tolist()
            q[token * pad + start : token * pad + start + 32] = bytes(quantized)

            tile, group = start // 128, (start // 32) & 3
            chunks[tile * 512 + token * 16 + group] = exponent + 127
    return q, chunks


@pytest.mark.parametrize("tokens", (8, 16, 32))
@pytest.mark.parametrize("dtype", (float16, bfloat16))
def test_gpt_oss_mxfp8_quant_declares_packed_exchange_format(tokens, dtype):
    record = _declare_quant(tokens, dtype)
    args, kwargs = record["args"], record["kwargs"]
    assert args[0] == "gpt_oss_mxfp8_quant"
    assert args[1] == (23, 1, 1)
    assert args[4] == [(1, -1, -1), (1, -1, -1)]
    assert args[5] == [2880, 2944, int(dtype == float16)]
    assert kwargs["exchange"] is True
    assert kwargs["counters"] == 1
    assert kwargs["buf_slots"] == [17]
    assert record["out_bytes"] == tokens * 2944 + 23 * 512
    assert kwargs["out_buffer"](None) == (tokens * 2944 + 23 * 512, 0xFF, False)


def test_cpu_reference_exponents_and_final_padded_tile_layout():
    h2 = [[0.0] * 2880 for _ in range(8)]
    h2[0][0] = 1.0
    h2[0][32] = 448.0
    h2[0][64] = 449.0
    h2[0][2879] = 1.0

    q, scales = _cpu_reference(h2, tokens=8)
    # Group 0 scales 1.0 by 2^8; group 1 is exactly the E4M3 maximum;
    # group 2 crosses that maximum and raises the exponent by one.
    assert scales[0] == 119  # ceil(log2(1 / 448)) = -8, then +127
    assert scales[1] == 127  # amax = 448
    assert scales[2] == 128  # amax = 449
    assert q[0] == 0x78      # E4M3(256)
    assert q[32] == 0x7E     # E4M3(448)
    assert q[64] == 0x76     # E4M3(449 / 2)

    final_tile = 22
    final_chunk = scales[final_tile * 512 : (final_tile + 1) * 512]
    assert final_chunk[0] == 19    # token 0, group 0: all-zero live values
    assert final_chunk[1] == 119   # token 0, group 1 contains col 2879
    assert final_chunk[2] == 19    # token 0, entirely padded K32 group
    assert final_chunk[3] == 19    # token 0, entirely padded K32 group
    for token in range(8):
        row = final_chunk[token * 16 : (token + 1) * 16]
        exponents = [19, 119, 19, 19] if token == 0 else [19] * 4
        assert row == bytes(exponents + [0] * 12)
    for token in range(8):
        assert q[token * 2944 + 2880 : (token + 1) * 2944] == bytes(64)
    # Only the first four bytes of each 16-byte token row carry scales.
    assert final_chunk[4:16] == bytes(12)
    assert final_chunk[8 * 16 :] == bytes(512 - 8 * 16)


def test_cpu_reference_handles_zero_input_and_exact_power_of_two_boundary():
    h2 = [[0.0] * 2880 for _ in range(8)]
    h2[0][0] = 448.0
    q, scales = _cpu_reference(h2, tokens=8)
    assert scales[0] == 127  # exactly 2^0 scale
    assert q[0] == 0x7E
    assert scales[1] == 19   # all-zero group follows the same clamped exponent rule
    assert q[32] == 0
    assert all(byte == 0 for byte in q[8 * 2944 :])
