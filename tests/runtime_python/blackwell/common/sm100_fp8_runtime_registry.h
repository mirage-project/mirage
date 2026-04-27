#pragma once

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace mirage::blackwell::sm100_fp8_runtime {

inline constexpr int kBlockK = 128;
inline constexpr int kScalePackSize = 4;
inline constexpr std::array<int, 1> kSupportedGroupSizes = {128};
inline constexpr std::array<int, 5> kSupportedBatchSizes = {1, 2, 4, 8, 16};
inline constexpr std::array<int, 1> kSupportedOutputSizes = {128};
inline constexpr std::array<int, 10> kSupportedReductionSizes = {
    128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 7168};

template <size_t N>
inline bool contains(std::array<int, N> const &values, int value) {
  for (int current : values) {
    if (current == value) {
      return true;
    }
  }
  return false;
}

template <size_t N>
inline std::string join(std::array<int, N> const &values) {
  std::ostringstream oss;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  return oss.str();
}

inline constexpr int round_up_to_multiple(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

inline constexpr int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

inline int logical_scale_k_for_reduction_size(int reduction_size) {
  return ceil_div(reduction_size, kBlockK);
}

inline int packed_scale_k_for_reduction_size(int reduction_size) {
  return ceil_div(logical_scale_k_for_reduction_size(reduction_size),
                  kScalePackSize);
}

inline int get_tma_aligned_size(int value, int element_size) {
  return round_up_to_multiple(value, 16 / element_size);
}

inline int aligned_scale_outer_dim(int outer_dim) {
  return get_tma_aligned_size(outer_dim, sizeof(uint32_t));
}

enum class PackedScaleLayout {
  Invalid,
  RowMajor,
  DeepGemmColumnMajor,
};

inline bool is_valid_rowmajor_scale_layout(int outer_dim,
                                           int reduction_size,
                                           int64_t shape0,
                                           int64_t shape1,
                                           int64_t stride0,
                                           int64_t stride1) {
  return shape0 == outer_dim &&
         shape1 == packed_scale_k_for_reduction_size(reduction_size) &&
         stride0 == packed_scale_k_for_reduction_size(reduction_size) &&
         stride1 == 1;
}

inline bool is_valid_deepgemm_scale_layout(int outer_dim,
                                           int reduction_size,
                                           int64_t shape0,
                                           int64_t shape1,
                                           int64_t stride0,
                                           int64_t stride1) {
  return shape0 == outer_dim &&
         shape1 == packed_scale_k_for_reduction_size(reduction_size) &&
         stride0 == 1 && stride1 == aligned_scale_outer_dim(outer_dim);
}

inline PackedScaleLayout detect_scale_layout(int outer_dim,
                                             int reduction_size,
                                             int64_t shape0,
                                             int64_t shape1,
                                             int64_t stride0,
                                             int64_t stride1) {
  if (is_valid_rowmajor_scale_layout(
          outer_dim, reduction_size, shape0, shape1, stride0, stride1)) {
    return PackedScaleLayout::RowMajor;
  }
  if (is_valid_deepgemm_scale_layout(
          outer_dim, reduction_size, shape0, shape1, stride0, stride1)) {
    return PackedScaleLayout::DeepGemmColumnMajor;
  }
  return PackedScaleLayout::Invalid;
}

inline std::vector<int64_t> expected_deepgemm_scale_shape(int outer_dim,
                                                          int reduction_size) {
  return {outer_dim, packed_scale_k_for_reduction_size(reduction_size)};
}

inline std::vector<int64_t> expected_deepgemm_scale_stride(int outer_dim) {
  return {1, aligned_scale_outer_dim(outer_dim)};
}

inline std::vector<int64_t> expected_rowmajor_scale_shape(int outer_dim,
                                                          int reduction_size) {
  return {outer_dim, packed_scale_k_for_reduction_size(reduction_size)};
}

inline std::vector<int64_t> expected_rowmajor_scale_stride(int reduction_size) {
  return {packed_scale_k_for_reduction_size(reduction_size), 1};
}

inline bool is_supported_group_size(int group_size) {
  return contains(kSupportedGroupSizes, group_size);
}

inline bool is_supported_hidden_size(int hidden_size) {
  return contains(kSupportedReductionSizes, hidden_size);
}

inline bool is_supported_dense_gemm_shape(int batch_size,
                                          int output_size,
                                          int reduction_size) {
  return contains(kSupportedBatchSizes, batch_size) &&
         contains(kSupportedOutputSizes, output_size) &&
         contains(kSupportedReductionSizes, reduction_size);
}

inline std::string supported_group_sizes_string() {
  return join(kSupportedGroupSizes);
}

inline std::string supported_hidden_sizes_string() {
  return join(kSupportedReductionSizes);
}

inline std::string supported_batch_sizes_string() {
  return join(kSupportedBatchSizes);
}

inline std::string supported_output_sizes_string() {
  return join(kSupportedOutputSizes);
}

inline std::string supported_reduction_sizes_string() {
  return join(kSupportedReductionSizes);
}

template <size_t N>
inline std::vector<int64_t> to_vector(std::array<int, N> const &values) {
  return std::vector<int64_t>(values.begin(), values.end());
}

inline std::vector<int64_t> supported_group_sizes_vector() {
  return to_vector(kSupportedGroupSizes);
}

inline std::vector<int64_t> supported_hidden_sizes_vector() {
  return to_vector(kSupportedReductionSizes);
}

inline std::vector<std::vector<int64_t>> supported_dense_gemm_shapes_vector() {
  std::vector<std::vector<int64_t>> shapes;
  for (int batch_size : kSupportedBatchSizes) {
    for (int output_size : kSupportedOutputSizes) {
      for (int reduction_size : kSupportedReductionSizes) {
        shapes.push_back(
            {batch_size, output_size, static_cast<int64_t>(reduction_size)});
      }
    }
  }
  return shapes;
}

} // namespace mirage::blackwell::sm100_fp8_runtime
