"""
Reference implementation for NVFP4 (float_e2m1) block-scaled matrix multiplication.

Implements:
  D = (A × SFA) * (B × SFB)

where A and B are packed FP4 (float_e2m1) tensors, SFA and SFB are
block scale factors (float_ue4m3), and the matmul is computed in float32.

Formats:
  - float_e2m1 (NVFP4): 4-bit float, 1 sign + 2 exponent + 1 mantissa, bias=1
    Two values packed per uint8 byte (low nibble = even index, high nibble = odd index)
  - float_ue4m3 (scale factor): 8-bit unsigned float, 4 exponent + 3 mantissa, bias=7
    One value per uint8 byte
  - Block size (scale vector size): 16 elements along K share one scale factor
"""

import torch
import numpy as np

# Lookup table for all 16 possible e2m1 nibble values (0x0..0xF)
# Format: S(1) E(2) M(1), bias = 1
#   subnormal (E==0): (-1)^S × 0.M × 2^(1-bias) = (-1)^S × M × 0.5
#   normal:           (-1)^S × (1 + M×0.5) × 2^(E-bias)
_E2M1_LUT = np.zeros(16, dtype=np.float32)
for _nib in range(16):
    _s = (_nib >> 3) & 1
    _e = (_nib >> 1) & 3
    _m = _nib & 1
    if _e == 0:
        _val = _m * 0.5                                    # subnormal
    else:
        _val = (1.0 + _m * 0.5) * (2.0 ** (_e - 1))       # normal
    _E2M1_LUT[_nib] = -_val if _s else _val

# Lookup table for all 256 possible ue4m3 byte values (0x00..0xFF)
# Format: E(4) M(3), unsigned, bias = 7
#   subnormal (E==0): (M/8) × 2^(1-bias) = M / 512
#   normal:           (1 + M/8) × 2^(E-bias)
_UE4M3_LUT = np.zeros(256, dtype=np.float32)
for _byte in range(256):
    _e = (_byte >> 3) & 0xF
    _m = _byte & 0x7
    if _e == 0:
        _UE4M3_LUT[_byte] = _m / 512.0                     # subnormal
    else:
        _UE4M3_LUT[_byte] = (1.0 + _m / 8.0) * (2.0 ** (_e - 7))
# Note: E=15 is treated as a normal value (no NaN/Inf for scale factors)


def decode_e2m1(nibbles: np.ndarray) -> np.ndarray:
    """Decode an array of 4-bit e2m1 indices (0..15) to float32 using LUT."""
    return _E2M1_LUT[nibbles.astype(np.intp)]

def decode_ue4m3(bytes_arr: np.ndarray) -> np.ndarray:
    """Decode an array of uint8 ue4m3 values to float32 using LUT."""
    return _UE4M3_LUT[bytes_arr.astype(np.intp)]

def unpack_e2m1(packed: torch.Tensor, reduction_size: int) -> torch.Tensor:
    """
    Unpack a uint8 tensor of packed e2m1 values to a float32 tensor.

    Args:
        packed: uint8 tensor of shape (rows, reduction_size // 2).
                Each byte holds two e2m1 values:
                  low  nibble (bits 0-3) → even-indexed element
                  high nibble (bits 4-7) → odd-indexed element
        reduction_size: the unpacked number of columns (must be even).

    Returns:
        float32 tensor of shape (rows, reduction_size).
    """
    assert packed.dtype == torch.uint8
    rows = packed.shape[0]
    assert packed.shape[1] == reduction_size // 2

    raw = packed.cpu().numpy()                      # (rows, K/2)
    low  = raw & 0x0F                               # even elements
    high = (raw >> 4) & 0x0F                         # odd  elements

    # Interleave: [low0, high0, low1, high1, ...]
    interleaved = np.empty((rows, reduction_size), dtype=np.int32)
    interleaved[:, 0::2] = low
    interleaved[:, 1::2] = high

    decoded = decode_e2m1(interleaved)               # float32
    return torch.from_numpy(decoded).to(torch.float32)

def decode_scale_factors(sf: torch.Tensor) -> torch.Tensor:
    """
    Decode a uint8 tensor of ue4m3 scale factors to float32.

    Args:
        sf: uint8 tensor of shape (rows, num_blocks).

    Returns:
        float32 tensor of the same shape.
    """
    assert sf.dtype == torch.uint8
    raw = sf.cpu().numpy()
    decoded = decode_ue4m3(raw)
    return torch.from_numpy(decoded).to(torch.float32)

def apply_block_scaling(
    values: torch.Tensor,
    scale_factors: torch.Tensor,
    scale_vector_size: int = 16,
) -> torch.Tensor:
    """
    Multiply each block of `scale_vector_size` elements along the last
    dimension by the corresponding scale factor.

    Args:
        values:         float32 tensor of shape (rows, K).
        scale_factors:  float32 tensor of shape (rows, K // scale_vector_size).
        scale_vector_size: number of elements per scale-factor block.

    Returns:
        float32 tensor of shape (rows, K).
    """
    rows, K = values.shape
    assert K % scale_vector_size == 0
    num_blocks = K // scale_vector_size
    assert scale_factors.shape == (rows, num_blocks)

    # Reshape → (rows, num_blocks, block) then broadcast-multiply
    blocked = values.reshape(rows, num_blocks, scale_vector_size)
    sf_expanded = scale_factors.unsqueeze(-1)        # (rows, num_blocks, 1)
    scaled = blocked * sf_expanded
    return scaled.reshape(rows, K)

def nvfp4_block_scaled_matmul(
    packed_a: torch.Tensor,
    sf_a: torch.Tensor,
    packed_b: torch.Tensor,
    sf_b: torch.Tensor,
    reduction_size: int,
    scale_vector_size: int = 16,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reference implementation: D = (A × SFA) * (B × SFB)^T  [+ residual]

    Convention (matching the kernel):
        A = weight matrix, shape (output_size, reduction_size) packed as
            (output_size, reduction_size // 2) uint8
        B = input  matrix, shape (batch_size,  reduction_size) packed as
            (batch_size,  reduction_size // 2) uint8
        SFA = weight scale factors, shape (output_size, reduction_size // SV) uint8
        SFB = input  scale factors, shape (batch_size,  reduction_size // SV) uint8

    Output shape: (batch_size, output_size)

    Args:
        packed_a:  weight, uint8 (output_size,  reduction_size // 2)
        sf_a:      weight SF, uint8 (output_size,  reduction_size // SV)
        packed_b:  input,  uint8 (batch_size,   reduction_size // 2)
        sf_b:      input  SF, uint8 (batch_size,   reduction_size // SV)
        reduction_size: unpacked K dimension
        scale_vector_size: block size for scaling (default 16)
        residual:  optional float32 (batch_size, output_size)

    Returns:
        float32 tensor of shape (batch_size, output_size).
    """
    a_f32 = unpack_e2m1(packed_a, reduction_size)     # (output_size, K)
    b_f32 = unpack_e2m1(packed_b, reduction_size)     # (batch_size,  K)
    sfa_f32 = decode_scale_factors(sf_a)              # (output_size, K/SV)
    sfb_f32 = decode_scale_factors(sf_b)              # (batch_size,  K/SV)

    a_scaled = apply_block_scaling(a_f32, sfa_f32, scale_vector_size)
    b_scaled = apply_block_scaling(b_f32, sfb_f32, scale_vector_size)

    # (batch_size, K) × (K, output_size) → (batch_size, output_size)
    output = torch.matmul(b_scaled, a_scaled.T)
    if residual is not None:
        output = output + residual
    return output

def encode_ue4m3(value: float) -> int:
    """
    Encode a non-negative float to the nearest ue4m3 byte.
    Useful for constructing known scale-factor tensors.

    Returns an int in [0, 255].
    """
    assert value >= 0.0
    if value == 0.0:
        return 0

    best_byte = 0
    best_err = abs(value)
    for b in range(256):
        candidate = float(_UE4M3_LUT[b])
        err = abs(value - candidate)
        if err < best_err:
            best_err = err
            best_byte = b
    return best_byte


def make_unit_scale_factors(rows: int, cols: int) -> torch.Tensor:
    """
    Create a scale-factor tensor where every entry is 1.0 in ue4m3 encoding.
    ue4m3(1.0) = E=7, M=0 → byte value 0x38 = 56.
    """
    UE4M3_ONE = encode_ue4m3(1.0)   # should be 56
    return torch.full((rows, cols), UE4M3_ONE, device="cuda", dtype=torch.uint8)

# # ---------------------------------------------------------------------------
# # Example corrected test script
# # ---------------------------------------------------------------------------

# def run_reference_test():
#     """
#     Demonstrates the correct way to test the NVFP4 kernel against
#     this reference implementation.
#     """
#     batch_size = 1
#     output_size = 128
#     reduction_size = 768
#     scale_vector_size = 16

#     print(f"=== Reference test: batch={batch_size}, out={output_size}, "
#           f"K={reduction_size}, SV={scale_vector_size} ===")

#     # --- Create test data ---
#     torch.manual_seed(42)

#     # Packed FP4 operands (random bytes → random fp4 values)
#     w_packed = torch.randint(
#         0, 256, (output_size, reduction_size // 2),
#         device="cuda", dtype=torch.uint8,
#     )
#     x_packed = torch.randint(
#         0, 256, (batch_size, reduction_size // 2),
#         device="cuda", dtype=torch.uint8,
#     )

#     # Scale factors: use 1.0 so the matmul is just on the raw fp4 values.
#     # IMPORTANT: ue4m3(1.0) = 0x38 = 56, NOT 0x01 = 1 !
#     w_sf = make_unit_scale_factors(output_size, reduction_size // scale_vector_size)
#     x_sf = make_unit_scale_factors(batch_size,  reduction_size // scale_vector_size)

#     # --- Compute reference ---
#     ref_output = nvfp4_block_scaled_matmul(
#         packed_a=w_packed,
#         sf_a=w_sf,
#         packed_b=x_packed,
#         sf_b=x_sf,
#         reduction_size=reduction_size,
#         scale_vector_size=scale_vector_size,
#         residual=None,
#     )
#     print(f"Reference output shape : {ref_output.shape}")
#     print(f"Reference output range : [{ref_output.min().item():.4f}, "
#           f"{ref_output.max().item():.4f}]")
#     print(f"Reference output sample: {ref_output[0, :8]}")

#     print("Done.\n")


# if __name__ == "__main__":
#     _self_test()
#     run_reference_test()