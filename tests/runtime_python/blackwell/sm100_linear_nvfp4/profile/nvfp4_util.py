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
    return _E2M1_LUT[nibbles.astype(np.intp)]

def decode_ue4m3(bytes_arr: np.ndarray) -> np.ndarray:
    return _UE4M3_LUT[bytes_arr.astype(np.intp)]

def unpack_e2m1(packed: torch.Tensor, reduction_size: int) -> torch.Tensor:
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
    assert sf.dtype == torch.uint8
    raw = sf.cpu().numpy()
    decoded = decode_ue4m3(raw)
    return torch.from_numpy(decoded).to(torch.float32)

def apply_block_scaling(
    values: torch.Tensor,
    scale_factors: torch.Tensor,
    scale_vector_size: int = 16,
) -> torch.Tensor:
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
) -> tuple[torch.Tensor, float]:
    a_f32 = unpack_e2m1(packed_a, reduction_size)     # (output_size, K)
    b_f32 = unpack_e2m1(packed_b, reduction_size)     # (batch_size,  K)
    sfa_f32 = decode_scale_factors(sf_a)              # (output_size, K/SV)
    sfb_f32 = decode_scale_factors(sf_b)              # (batch_size,  K/SV)

    a_scaled = apply_block_scaling(a_f32, sfa_f32, scale_vector_size)
    b_scaled = apply_block_scaling(b_f32, sfb_f32, scale_vector_size)

    # (batch_size, K) × (K, output_size) → (batch_size, output_size)
    b_scaled_cuda = b_scaled.cuda()
    a_scaled_cuda = a_scaled.cuda()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = torch.matmul(b_scaled_cuda, a_scaled_cuda.T)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    if residual is not None:
        output = output + residual
    return output, elapsed_ms

def _to_blocked_sf(sf: torch.Tensor) -> torch.Tensor:
    """Convert (rows, sf_k) uint8 scale factors to the blocked layout expected
    by torch._scaled_mm (same as to_blocked in the reference kernel)."""
    rows, cols = sf.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    blocks = sf.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().view(torch.float8_e4m3fn)


def _pad_rows_uint8(tensor: torch.Tensor, padded_rows: int, fill_value: int = 0) -> torch.Tensor:
    if tensor.shape[0] == padded_rows:
        return tensor
    padded = torch.full(
        (padded_rows, tensor.shape[1]),
        fill_value,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    padded[: tensor.shape[0]].copy_(tensor)
    return padded


def nvfp4_scaled_mm(
    packed_a: torch.Tensor,
    sf_a: torch.Tensor,
    packed_b: torch.Tensor,
    sf_b: torch.Tensor,
    reduction_size: int,
    residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Reference using torch._scaled_mm with native fp4/fp8 types.
    packed_a: (output_size, K/2) uint8  — weight matrix (w)
    sf_a:     (output_size, K/16) uint8 — weight scale factors
    packed_b: (batch_size,  K/2) uint8  — activation matrix (x)
    sf_b:     (batch_size,  K/16) uint8 — activation scale factors
    """
    logical_rows = packed_b.shape[0]
    # torch._scaled_mm accepts any M, but _to_blocked_sf requires scale factor
    # rows to be a multiple of 128. Pad only the scale factors, not the data.
    padded_rows = ((logical_rows + 127) // 128) * 128
    sf_b_padded = _pad_rows_uint8(sf_b.cuda(), padded_rows, fill_value=encode_ue4m3(1.0))

    # Reinterpret uint8 packed bytes as float4_e2m1fn_x2
    a_fp4 = packed_a.cuda().view(torch.float4_e2m1fn_x2)  # (output_size, K/2)
    b_fp4 = packed_b.cuda().view(torch.float4_e2m1fn_x2)  # (batch_size,  K/2)

    scale_a = _to_blocked_sf(sf_a.cuda())        # flattened blocked float8
    scale_b = _to_blocked_sf(sf_b_padded)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    # (batch_size, K/2) @ (K/2, output_size) → (batch_size, output_size)
    result = torch._scaled_mm(
        b_fp4,
        a_fp4.transpose(0, 1),
        scale_b,
        scale_a,
        bias=None,
        out_dtype=torch.float32,
    )
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)

    if residual is not None:
        result = result + residual
    return result, elapsed_ms


def encode_ue4m3(value: float) -> int:
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
        exp = torch.randint(0, 4, shape, device=device, dtype=torch.uint8)
        mant = torch.randint(0, 8,  shape, device=device, dtype=torch.uint8)
        return (exp << 3) | mant

    x_packed = random_fp4_bytes((batch_size, half_k))
    w_packed = random_fp4_bytes((output_size, half_k))
    x_sf     = random_valid_fp8((batch_size, num_sf_cols))
    w_sf     = random_valid_fp8((output_size, num_sf_cols))

    return x_packed, w_packed, x_sf, w_sf


def make_sequential_scale_factors(rows: int, cols: int) -> torch.Tensor:
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
    UE4M3_ONE = encode_ue4m3(1.0)   # should be 56
    return torch.full((rows, cols), UE4M3_ONE, device="cuda", dtype=torch.uint8)

def interleave_sf_tensor(
    sf: torch.Tensor,
    pad_rows_to_multiple_of: int | None = None,
    pad_value: int | None = None,
) -> torch.Tensor:
    M, SF_K = sf.shape
    if pad_rows_to_multiple_of is None and M < 128:
        pad_rows_to_multiple_of = 128
    if pad_rows_to_multiple_of is not None and M % pad_rows_to_multiple_of != 0:
        padded_m = ((M + pad_rows_to_multiple_of - 1) // pad_rows_to_multiple_of) * pad_rows_to_multiple_of
        if pad_value is None:
            pad_value = encode_ue4m3(1.0)
        padded = torch.full(
            (padded_m, SF_K),
            pad_value,
            device=sf.device,
            dtype=sf.dtype,
        )
        padded[:M].copy_(sf)
        sf = padded
        M = padded_m
    REST_M = M // 128
    NUM_K_OUTER = SF_K // 4
    out = sf.reshape(REST_M, 4, 32, NUM_K_OUTER, 4)
    out = out.permute(0, 3, 2, 1, 4).contiguous()
    return out
