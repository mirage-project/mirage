#pragma once

namespace mirage::blackwell::linear_fp8_sm100 {

enum class MmaKind {
  BF16 = 0,
  MXFP8FP4 = 1,
};

constexpr __host__ __device__ int get_element_size(MmaKind const &mma_kind) {
  switch (mma_kind) {
    case MmaKind::BF16:
      return 2;
    case MmaKind::MXFP8FP4:
      return 1;
    default:
      return 0;
  }
}

enum class GemmType {
  Normal = 0,
  MGroupedContiguous = 1,
  MGroupedMasked = 2,
  KGroupedContiguous = 3,
  Batched = 4,
  MGroupedContiguousWithPsumLayout = 5,
};

constexpr __host__ __device__ bool
    is_m_grouped_contiguous(GemmType const &gemm_type) {
  switch (gemm_type) {
    case GemmType::MGroupedContiguous:
      return true;
    case GemmType::MGroupedContiguousWithPsumLayout:
      return true;
    default:
      return false;
  }
}

enum class KernelType {
  Kernel1D1D = 0,
  Kernel1D2D = 1,
  KernelNoSF = 2,
};

} // namespace mirage::blackwell::linear_fp8_sm100
