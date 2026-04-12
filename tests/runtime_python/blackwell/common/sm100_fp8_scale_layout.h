#pragma once

#include "sm100_fp8_runtime_registry.h"

#include <ATen/core/TensorBody.h>
#include <torch/extension.h>

namespace mirage::blackwell::sm100_fp8_scale_layout {

inline void check_scale_tensor(torch::Tensor const &scale,
                               int outer_dim,
                               int reduction_size,
                               char const *name) {
  TORCH_CHECK(scale.dim() == 2, name, " must be 2D");
  TORCH_CHECK(scale.scalar_type() == at::kInt ||
                  scale.scalar_type() == at::kUInt32,
              name,
              " must be uint32-compatible");
  TORCH_CHECK(
      sm100_fp8_runtime::is_valid_deepgemm_scale_layout(outer_dim,
                                                        reduction_size,
                                                        scale.size(0),
                                                        scale.size(1),
                                                        scale.stride(0),
                                                        scale.stride(1)),
      name,
      " must use DeepGEMM packed UE8M0 layout with shape [",
      outer_dim,
      ", ",
      sm100_fp8_runtime::packed_scale_k_for_reduction_size(reduction_size),
      "] and stride [1, ",
      sm100_fp8_runtime::aligned_scale_outer_dim(outer_dim),
      "], but got shape [",
      scale.size(0),
      ", ",
      scale.size(1),
      "] and stride [",
      scale.stride(0),
      ", ",
      scale.stride(1),
      "]");
}

} // namespace mirage::blackwell::sm100_fp8_scale_layout
