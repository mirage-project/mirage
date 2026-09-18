import torch


BLOCK_K = 128
SCALE_PACK_SIZE = 4
FP8_MAX = 448.0


def round_up_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def logical_scale_k_for_reduction_size(reduction_size: int) -> int:
    return ceil_div(reduction_size, BLOCK_K)


def packed_scale_k_for_reduction_size(reduction_size: int) -> int:
    return ceil_div(logical_scale_k_for_reduction_size(reduction_size), SCALE_PACK_SIZE)


def get_tma_aligned_size(value: int, element_size: int) -> int:
    return round_up_to_multiple(value, 16 // element_size)


def aligned_scale_outer_dim(outer_dim: int) -> int:
    return get_tma_aligned_size(outer_dim, 4)


def allocate_packed_ue8m0_scale(outer_dim: int, reduction_size: int, device):
    return torch.empty(
        (outer_dim, packed_scale_k_for_reduction_size(reduction_size)),
        device=device,
        dtype=torch.uint32,
    )


def allocate_packed_ue8m0_scale_deepgemm_style(
    outer_dim: int, reduction_size: int, device
):
    return torch.empty_strided(
        (outer_dim, packed_scale_k_for_reduction_size(reduction_size)),
        (1, aligned_scale_outer_dim(outer_dim)),
        device=device,
        dtype=torch.uint32,
    )


def detect_packed_scale_layout(
    packed_scales: torch.Tensor, outer_dim: int, reduction_size: int
) -> str:
    packed_k = packed_scale_k_for_reduction_size(reduction_size)
    if packed_scales.shape != (outer_dim, packed_k):
        return "invalid"
    if tuple(packed_scales.stride()) == (packed_k, 1):
        return "row_major"
    if tuple(packed_scales.stride()) == (1, aligned_scale_outer_dim(outer_dim)):
        return "deepgemm_col_major"
    return "invalid"


def encode_ue8m0(scale: float) -> int:
    scale_tensor = torch.tensor(scale, dtype=torch.float32)
    ue8m0 = int(torch.ceil(torch.log2(torch.clamp(scale_tensor, min=1e-30))).item()) + 127
    return max(0, min(255, ue8m0))


def decode_ue8m0(encoded: int) -> float:
    return 2.0 ** (encoded - 127)


def _quantize_to_fp8_packed_ue8m0(x_bf16: torch.Tensor, layout: str):
    assert x_bf16.dim() == 2
    outer_dim, reduction_size = x_bf16.shape
    assert reduction_size % BLOCK_K == 0

    logical_scale_k = logical_scale_k_for_reduction_size(reduction_size)

    x_fp32 = x_bf16.float()
    x_q = torch.empty_like(x_fp32, dtype=torch.float8_e4m3fn)
    if layout == "row_major":
        packed_scales = allocate_packed_ue8m0_scale(
            outer_dim, reduction_size, x_bf16.device
        )
    elif layout == "deepgemm_col_major":
        packed_scales = allocate_packed_ue8m0_scale_deepgemm_style(
            outer_dim, reduction_size, x_bf16.device
        )
    else:
        raise ValueError(f"Unsupported packed scale layout: {layout}")
    packed_scales.zero_()

    for outer_idx in range(outer_dim):
        for scale_k in range(logical_scale_k):
            k_start = scale_k * BLOCK_K
            k_end = k_start + BLOCK_K
            block = x_fp32[outer_idx, k_start:k_end]
            abs_max = max(block.abs().max().item(), 1e-10)
            raw_scale = abs_max / FP8_MAX
            encoded = encode_ue8m0(raw_scale)
            snapped_scale = decode_ue8m0(encoded)
            x_q[outer_idx, k_start:k_end] = torch.clamp(
                block / snapped_scale, -FP8_MAX, FP8_MAX
            ).to(torch.float8_e4m3fn)

            packed_k = scale_k // SCALE_PACK_SIZE
            packed_shift = (scale_k % SCALE_PACK_SIZE) * 8
            current = int(packed_scales[outer_idx, packed_k].item())
            current |= (encoded & 0xFF) << packed_shift
            packed_scales[outer_idx, packed_k] = current

    return x_q, packed_scales


def quantize_to_fp8_packed_ue8m0(x_bf16: torch.Tensor):
    return _quantize_to_fp8_packed_ue8m0(x_bf16, layout="row_major")


def quantize_to_fp8_deepgemm_style(x_bf16: torch.Tensor):
    return _quantize_to_fp8_packed_ue8m0(x_bf16, layout="deepgemm_col_major")


def dequant_from_packed_ue8m0(x_q: torch.Tensor, packed_scales: torch.Tensor):
    assert x_q.dim() == 2
    outer_dim, reduction_size = x_q.shape
    logical_scale_k = logical_scale_k_for_reduction_size(reduction_size)
    layout = detect_packed_scale_layout(packed_scales, outer_dim, reduction_size)
    assert layout in ("row_major", "deepgemm_col_major")

    x_q_fp32 = x_q.float()
    out = torch.empty_like(x_q_fp32, dtype=torch.float32)
    for outer_idx in range(outer_dim):
        for scale_k in range(logical_scale_k):
            packed = int(packed_scales[outer_idx, scale_k // SCALE_PACK_SIZE].item())
            encoded = (packed >> ((scale_k % SCALE_PACK_SIZE) * 8)) & 0xFF
            scale = decode_ue8m0(encoded)
            k_start = scale_k * BLOCK_K
            k_end = k_start + BLOCK_K
            out[outer_idx, k_start:k_end] = x_q_fp32[outer_idx, k_start:k_end] * scale
    return out


def dequant_from_packed_ue8m0_deepgemm_style(
    x_q: torch.Tensor, packed_scales: torch.Tensor
):
    assert detect_packed_scale_layout(
        packed_scales, x_q.shape[0], x_q.shape[1]
    ) == "deepgemm_col_major"
    return dequant_from_packed_ue8m0(x_q, packed_scales)
