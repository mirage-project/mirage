"""
    Test cases for nvfp4_util.py
"""

import torch
import numpy as np
from nvfp4_util import (
    _E2M1_LUT, _UE4M3_LUT, encode_ue4m3,
    unpack_e2m1, decode_scale_factors, apply_block_scaling,
    nvfp4_block_scaled_matmul, make_unit_scale_factors,
)

def test_E2M1():
    assert _E2M1_LUT[0b0000] ==  0.0
    assert _E2M1_LUT[0b0001] ==  0.5
    assert _E2M1_LUT[0b0010] ==  1.0
    assert _E2M1_LUT[0b0011] ==  1.5
    assert _E2M1_LUT[0b0100] ==  2.0
    assert _E2M1_LUT[0b0101] ==  3.0
    assert _E2M1_LUT[0b0110] ==  4.0
    assert _E2M1_LUT[0b0111] ==  6.0
    assert _E2M1_LUT[0b1010] == -1.0
    assert _E2M1_LUT[0b1111] == -6.0

def test_UE4M3():
    assert _UE4M3_LUT[0] == 0.0
    assert _UE4M3_LUT[56] == 1.0
    assert _UE4M3_LUT[64] == 2.0
    assert _UE4M3_LUT[127] == 480.0
    assert np.all(_UE4M3_LUT >= 0)

def test_encode_roundtrip():
    for b in range(256):
        val = float(_UE4M3_LUT[b])
        assert _UE4M3_LUT[encode_ue4m3(val)] == val
        
def test_unpack():
    cases = [
        (0,   2, [[0.0, 0.0]]),
        (82,  2, [[1.0, 3.0]]),     # low=0x2(1.0), high=0x5(3.0)
        (250, 2, [[-1.0, -6.0]]),   # low=0xA(-1.0), high=0xF(-6.0)
    ]
    for byte_val, k, expected in cases:
        result = unpack_e2m1(torch.tensor([[byte_val]], dtype=torch.uint8), k)
        assert result.tolist() == expected, f"byte {byte_val}: {result.tolist()}"
        
def test_block_scaling():
    vals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    sf = torch.tensor([[2.0, 0.5]])
    assert apply_block_scaling(vals, sf, scale_vector_size=2).tolist() == [[2.0, 4.0, 1.5, 2.0]]

def test_matmul():
    # A=[[1.0,0.5,2.0,1.5],[3.0,0.0,1.0,4.0]], B=[[0.5,1.0,1.5,2.0]]
    # B @ A^T = [[7.0, 11.0]]
    a = torch.tensor([[18, 52], [5, 98]], dtype=torch.uint8)
    b = torch.tensor([[33, 67]], dtype=torch.uint8)
    sf_a = torch.full((2, 1), 56, dtype=torch.uint8)
    sf_b = torch.full((1, 1), 56, dtype=torch.uint8)
    kw = dict(reduction_size=4, scale_vector_size=4)

    out = nvfp4_block_scaled_matmul(a, sf_a, b, sf_b, **kw)
    torch.testing.assert_close(out, torch.tensor([[7.0, 11.0]]), atol=1e-5, rtol=1e-5)

    # with residual
    out = nvfp4_block_scaled_matmul(a, sf_a, b, sf_b, **kw, residual=torch.tensor([[100.0, 200.0]]))
    torch.testing.assert_close(out, torch.tensor([[107.0, 211.0]]), atol=1e-5, rtol=1e-5)

def test_zero_scales_zero_output():
    a = torch.randint(0, 256, (4, 8), dtype=torch.uint8)
    b = torch.zeros(2, 8, dtype=torch.uint8)
    sf_zero = torch.zeros(4, 1, dtype=torch.uint8)
    sf_b = torch.full((2, 1), 56, dtype=torch.uint8)
    out = nvfp4_block_scaled_matmul(a, sf_zero, b, sf_b, reduction_size=16, scale_vector_size=16)
    assert (out == 0).all()

def test_unit_scale_factors():
    sf = make_unit_scale_factors(2, 3)
    assert (sf == 56).all()
    # byte 1 (torch.ones) is NOT 1.0 in ue4m3
    assert _UE4M3_LUT[1] != 1.0

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn() 
            print(f"  PASS  {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")