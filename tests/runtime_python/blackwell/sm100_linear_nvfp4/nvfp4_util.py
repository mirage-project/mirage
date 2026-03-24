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


def make_sequential_nvfp4_tensors(
    batch_size: int,
    output_size: int,
    reduction_size: int,
    scale_vector_size: int = 16,
    device: str = "cuda",
):
    """
    Create x (input) and w (weight) packed FP4 tensors with sequential nibble
    patterns for debugging kernel data flow.

    Nibble sequence for x: [1, 3, 5, 7, 9, 11, 13, 15, 1, 3, ...]  (odd, period 8)
    Nibble sequence for w: [0, 2, 4, 6,  8, 10, 12, 14, 0, 2, ...]  (even, period 8)

    Packing: low nibble (bits 0-3) = element at even index,
             high nibble (bits 4-7) = element at odd index.

    So reading the raw bytes for x gives:  0x31, 0x75, 0xB9, 0xFD, 0x31, ...
    And reading the raw bytes for w gives:  0x20, 0x64, 0xA8, 0xEC, 0x20, ...

    Scale factors are set to 1.0 (ue4m3 = 0x38 = 56) for both tensors.

    Returns:
        x_packed: uint8 (batch_size,  reduction_size // 2)
        w_packed: uint8 (output_size, reduction_size // 2)
        x_sf:     uint8 (batch_size,  reduction_size // scale_vector_size)
        w_sf:     uint8 (output_size, reduction_size // scale_vector_size)
    """
    assert reduction_size % 2 == 0
    half_k = reduction_size // 2
    num_sf_cols = reduction_size // scale_vector_size

    # Build the nibble sequence for one row (length = reduction_size)
    indices = np.arange(reduction_size, dtype=np.int32)

    # x: odd nibbles — position i → (2i + 1) mod 16
    x_nibs = ((2 * indices + 1) % 16).astype(np.uint8)
    # w: even nibbles — position i → (2i) mod 16
    w_nibs = ((2 * indices) % 16).astype(np.uint8)

    # Pack pairs: byte j = high(nib[2j+1]) | low(nib[2j])
    x_row = ((x_nibs[1::2] << 4) | x_nibs[0::2]).astype(np.uint8)  # (half_k,)
    w_row = ((w_nibs[1::2] << 4) | w_nibs[0::2]).astype(np.uint8)  # (half_k,)

    # Tile the single row across all rows of each matrix
    x_packed = torch.from_numpy(
        np.tile(x_row, (batch_size, 1))
    ).to(device=device, dtype=torch.uint8)
    w_packed = torch.from_numpy(
        np.tile(w_row, (output_size, 1))
    ).to(device=device, dtype=torch.uint8)

    # Unit scale factors so scaled value = raw fp4 value
    # x_sf = make_unit_scale_factors(batch_size, num_sf_cols)
    # w_sf = make_unit_scale_factors(output_size, num_sf_cols)
    x_sf = make_sequential_scale_factors(batch_size, num_sf_cols)
    w_sf = make_sequential_scale_factors(output_size, num_sf_cols)

    return x_packed, w_packed, x_sf, w_sf


def make_random_nvfp4_tensors(
    batch_size: int,
    output_size: int,
    reduction_size: int,
    scale_vector_size: int = 16,
    device: str = "cuda",
):
    half_k = reduction_size // 2
    num_sf_cols = reduction_size // scale_vector_size

    def random_fp4_bytes(shape):
        lo = torch.randint(0, 16, shape, device=device, dtype=torch.uint8)
        hi = torch.randint(0, 16, shape, device=device, dtype=torch.uint8)
        return (hi << 4) | lo

    def random_valid_fp8(shape):
        exp = torch.randint(0, 15, shape, device=device, dtype=torch.uint8)
        mant = torch.randint(0, 8,  shape, device=device, dtype=torch.uint8)
        return (exp << 3) | mant

    x_packed = random_fp4_bytes((batch_size, half_k))
    w_packed = random_fp4_bytes((output_size, half_k))
    x_sf     = random_valid_fp8((batch_size, num_sf_cols))
    w_sf     = random_valid_fp8((output_size, num_sf_cols))

    return x_packed, w_packed, x_sf, w_sf


def make_sequential_scale_factors(rows: int, cols: int) -> torch.Tensor:
    """
    Generate dense, valid ue4m3 scale factors (no NaN/Inf encodings).

    Strategy:
    - Sweep through all exponent values 0–14
    - Sweep mantissa 0–7
    - Avoid exponent = 15 (NaN/Inf)
    """
    valid_bytes = []
    for exp in range(0, 15):          # 0..14 (safe)
        for mant in range(0, 8):      # 3-bit mantissa
            byte = (exp << 3) | mant
            valid_bytes.append(byte)
    valid_bytes = np.array(valid_bytes, dtype=np.uint8)  # size = 15 * 8 = 120
    total = rows * cols
    tiled = np.tile(valid_bytes, total // len(valid_bytes) + 1)[:total]
    return torch.from_numpy(tiled.reshape(rows, cols)).to("cuda")


def make_unit_scale_factors(rows: int, cols: int) -> torch.Tensor:
    """
    Create a scale-factor tensor where every entry is 1.0 in ue4m3 encoding.
    ue4m3(1.0) = E=7, M=0 → byte value 0x38 = 56.
    """
    UE4M3_ONE = encode_ue4m3(1.0)   # should be 56
    return torch.full((rows, cols), UE4M3_ONE, device="cuda", dtype=torch.uint8)

def interleave_sf_tensor(sf: torch.Tensor) -> torch.Tensor:
    M, SF_K = sf.shape
    REST_M = M // 128
    NUM_K_OUTER = SF_K // 4
    out = sf.reshape(REST_M, 4, 32, NUM_K_OUTER, 4)
    out = out.permute(0, 3, 2, 1, 4).contiguous()
    return out

# def interleave_sf_tensor(sf: torch.Tensor) -> torch.Tensor:
#     """
#     Reorder a scale-factor tensor from row-major (M, SF_K) layout into the
#     physical SMEM layout expected by deduce_smem_layoutSFA/B and the TMA 3D descriptor.

#     The SMEM layout (from Sm1xxBlockScaledBasicChunk with MMA_M=128, MMA_K=64, SV=16)
#     places sf[row, k_group] at byte offset:
#         (row%32)*16 + (row/32)*4 + (k_group%4) + (k_group//4)*512

#     This requires a 5-way reshape and permute:
#         sf[REST_M, row=128, SF_K=16]
#         → sf[REST_M, r_inner=32, r_outer=4, k_outer=4, k_inner=4]   (reshape)
#         → out[REST_M, k_outer, r_inner, r_outer, k_inner]            (permute 0,3,1,2,4)
#         → contiguous → byte = k_outer*512 + r_inner*16 + r_outer*4 + k_inner  ✓

#     The TMA 3D descriptor loads one (1, 1, SF_COL=256) half_t box per issue, advancing
#     the GMEM ROW coordinate by 1 per SMEM_REPEAT_ROW iteration, loading:
#         k_outer=0 → SMEM[0..511]
#         k_outer=1 → SMEM[512..1023]
#         k_outer=2 → SMEM[1024..1535]
#         k_outer=3 → SMEM[1536..2047]

#     Args:
#         sf: uint8 tensor of shape (M, SF_K). M must be a multiple of 128; SF_K a multiple of 4.

#     Returns:
#         Contiguous uint8 tensor in the interleaved layout for TMA.
#     """
#     M, SF_K = sf.shape
#     assert M % 128 == 0, f"M={M} must be a multiple of 128 (MMA_M)"
#     assert SF_K % 4 == 0, f"SF_K={SF_K} must be a multiple of 4 (MMA_K/SV)"
#     REST_M = M // 128
#     NUM_K_OUTER = SF_K // 4  # number of MMA_K blocks (= REDUCTION_SIZE / MMA_K)
#     # sf[REST_M, r_outer=4, r_inner=32, k_outer, k_inner=4]  (row-major: r_outer is outer)
#     out = sf.reshape(REST_M, 4, 32, NUM_K_OUTER, 4)
#     # permute to (REST_M, k_outer, r_inner, r_outer, k_inner)
#     # → byte = k_outer*512 + r_inner*16 + r_outer*4 + k_inner  ✓
#     out = out.permute(0, 3, 2, 1, 4).contiguous()
#     return out


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