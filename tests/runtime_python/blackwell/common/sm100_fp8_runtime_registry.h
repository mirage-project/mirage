#pragma once

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace mirage::blackwell::sm100_fp8_runtime {

inline constexpr int kBlockK = 128;
inline constexpr std::array<int, 1> kSupportedGroupSizes = {128};
inline constexpr std::array<int, 1> kSupportedBatchSizes = {1};
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

inline int padded_scale_k_for_reduction_size(int reduction_size) {
  return round_up_to_multiple(reduction_size / kBlockK, 4);
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
