import torch

_E2M1_LUT = []
for nibble in range(16):
    sign = (nibble >> 3) & 1
    exponent = (nibble >> 1) & 0x3
    mantissa = nibble & 0x1
    if exponent == 0:
        value = mantissa * 0.5
    else:
        value = (1.0 + mantissa * 0.5) * (2.0 ** (exponent - 1))
    _E2M1_LUT.append(-value if sign else value)
_E2M1_LUT = torch.tensor(_E2M1_LUT, dtype=torch.float32)


def encode_ue8m0(value: float) -> int:
    encoded = (
        torch.tensor([value], device="cuda", dtype=torch.float32)
        .to(torch.float8_e8m0fnu)
        .view(torch.uint8)
    )
    return int(encoded.item())


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


def _to_blocked_sf(sf: torch.Tensor) -> torch.Tensor:
    rows, cols = sf.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + 3) // 4
    blocks = sf.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().view(torch.float8_e8m0fnu)


def deinterleave_sf_tensor(sf_interleaved: torch.Tensor, logical_rows: int | None = None) -> torch.Tensor:
    padded_rows = sf_interleaved.shape[0] * 128
    sf_k = sf_interleaved.shape[1] * 4
    sf = sf_interleaved.permute(0, 3, 2, 1, 4).reshape(padded_rows, sf_k)
    if logical_rows is not None:
        return sf[:logical_rows]
    return sf


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    rows, half_k = packed.shape
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked = torch.empty((rows, half_k * 2), device=packed.device, dtype=torch.uint8)
    unpacked[:, 0::2] = low
    unpacked[:, 1::2] = high
    return _E2M1_LUT.to(device=packed.device)[unpacked.long()]


def decode_scale_factors(sf: torch.Tensor) -> torch.Tensor:
    return sf.contiguous().view(torch.float8_e8m0fnu).to(torch.float32)


def apply_block_scaling(values: torch.Tensor, scale_factors: torch.Tensor, scale_vector_size: int = 32) -> torch.Tensor:
    rows, reduction_size = values.shape
    num_blocks = reduction_size // scale_vector_size
    blocked = values.reshape(rows, num_blocks, scale_vector_size)
    return (blocked * scale_factors.unsqueeze(-1)).reshape(rows, reduction_size)


def mxfp4_scaled_mm(
    packed_a: torch.Tensor,
    sf_a: torch.Tensor,
    packed_b: torch.Tensor,
    sf_b: torch.Tensor,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    a_values = unpack_e2m1(packed_a.cuda())
    b_values = unpack_e2m1(packed_b.cuda())
    a_scales = decode_scale_factors(sf_a.cuda())
    b_scales = decode_scale_factors(sf_b.cuda())
    a_scaled = apply_block_scaling(a_values, a_scales)
    b_scaled = apply_block_scaling(b_values, b_scales)
    result = torch.matmul(b_scaled, a_scaled.T)
    if residual is not None:
        result = result + residual
    return result
