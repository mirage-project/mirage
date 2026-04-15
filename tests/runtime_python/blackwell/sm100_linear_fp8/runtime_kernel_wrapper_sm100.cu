/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../common/sm100_fp8_runtime_registry.h"
#include "../common/sm100_fp8_scale_layout.h"
#include "blackwell/linear_fp8_sm100.cuh"
#include "blackwell/linear_fp8_sm100_splitk.cuh"
#include "runtime_header.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <map>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <tuple>
#include <vector>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

using bfloat16 = cute::bfloat16_t;
namespace fp8_runtime = mirage::blackwell::sm100_fp8_runtime;
namespace fp8_linear = mirage::blackwell::linear_fp8_sm100;
// sm100_linear_fp8_1d2d
namespace {

struct LinearFp8RuntimeKey {
  int batch_size;
  int output_size;
  int reduction_size;
  int split_k;
  bool has_residual;

  bool operator<(LinearFp8RuntimeKey const &other) const {
    return std::tie(
               batch_size, output_size, reduction_size, split_k, has_residual) <
           std::tie(other.batch_size,
                    other.output_size,
                    other.reduction_size,
                    other.split_k,
                    other.has_residual);
  }
};

struct LinearFp8DescriptorCache {
  CUtensorMap host_input_desc{};
  CUtensorMap host_weight_desc{};
  CUtensorMap host_input_scale_desc{};
  CUtensorMap host_weight_scale_desc{};
  CUtensorMap host_residual_desc{};
  CUtensorMap host_output_desc{};
  at::Tensor internal_input_scale_buffer;
  at::Tensor internal_weight_scale_buffer;
  at::Tensor splitk_accum_buffer;
  void *last_input_ptr = nullptr;
  void *last_weight_ptr = nullptr;
  void *last_input_scale_ptr = nullptr;
  void *last_weight_scale_ptr = nullptr;
  void *last_residual_ptr = nullptr;
  void *last_output_ptr = nullptr;
  bool initialized = false;
  bool kernel_configured = false;
  std::mutex mutex;
};

LinearFp8DescriptorCache &
    get_linear_fp8_descriptor_cache(LinearFp8RuntimeKey const &key) {
  static std::map<LinearFp8RuntimeKey,
                  std::unique_ptr<LinearFp8DescriptorCache>>
      caches;
  static std::mutex caches_mutex;

  std::lock_guard<std::mutex> guard(caches_mutex);
  auto &cache = caches[key];
  if (!cache) {
    cache = std::make_unique<LinearFp8DescriptorCache>();
  }
  return *cache;
}

void check_driver_success(CUresult result, char const *what) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  char const *error_name = nullptr;
  char const *error_string = nullptr;
  cuGetErrorName(result, &error_name);
  cuGetErrorString(result, &error_string);
  TORCH_CHECK(false,
              what,
              " failed with ",
              (error_name ? error_name : "unknown"),
              ": ",
              (error_string ? error_string : "unknown"));
}

void check_last_launch(char const *kernel_name) {
  auto const err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess,
              kernel_name,
              " launch failed: ",
              cudaGetErrorString(err));
}

int read_forced_split_k_override() {
  char const *env_value = std::getenv("MIRAGE_FORCE_SM100_FP8_SPLIT_K");
  if (env_value == nullptr || env_value[0] == '\0') {
    return 0;
  }

  char *end_ptr = nullptr;
  long const parsed_value = std::strtol(env_value, &end_ptr, 10);
  TORCH_CHECK(end_ptr != env_value && *end_ptr == '\0',
              "MIRAGE_FORCE_SM100_FP8_SPLIT_K must be an integer, got: ",
              env_value);
  TORCH_CHECK(parsed_value == 1 || parsed_value == 2 || parsed_value == 4 ||
                  parsed_value == 8,
              "MIRAGE_FORCE_SM100_FP8_SPLIT_K must be one of {1, 2, 4, 8}, got: ",
              parsed_value);
  return static_cast<int>(parsed_value);
}

int choose_split_k(int batch_size, int reduction_size, bool has_residual) {
  if (has_residual) {
    return 1;
  }

  int const forced_split_k = read_forced_split_k_override();
  if (forced_split_k != 0) {
    return forced_split_k;
  }

  if (reduction_size < 7168) {
    return 1;
  }
  return batch_size <= 4 ? 4 : 8;
}

torch::Tensor allocate_legacy_col_major_scale_buffer(torch::Tensor const &source,
                                                     int outer_dim,
                                                     int reduction_size) {
  return torch::empty_strided(
      {outer_dim, fp8_runtime::packed_scale_k_for_reduction_size(reduction_size)},
      {1, fp8_runtime::aligned_scale_outer_dim(outer_dim)},
      source.options());
}

torch::Tensor const &materialize_internal_scale_tensor(
    LinearFp8DescriptorCache &cache,
    torch::Tensor const &scale,
    int outer_dim,
    int reduction_size,
    bool is_input_scale) {
  auto const layout = mirage::blackwell::sm100_fp8_scale_layout::
      detect_scale_tensor_layout(scale, outer_dim, reduction_size);
  TORCH_CHECK(layout != fp8_runtime::PackedScaleLayout::Invalid,
              "Invalid packed scale tensor layout");
  if (layout == fp8_runtime::PackedScaleLayout::DeepGemmColumnMajor) {
    return scale;
  }

  at::Tensor &buffer = is_input_scale ? cache.internal_input_scale_buffer
                                      : cache.internal_weight_scale_buffer;
  int const packed_scale_k =
      fp8_runtime::packed_scale_k_for_reduction_size(reduction_size);
  int const aligned_outer = fp8_runtime::aligned_scale_outer_dim(outer_dim);
  if (!buffer.defined() || buffer.device() != scale.device() ||
      buffer.scalar_type() != scale.scalar_type() || buffer.size(0) != outer_dim ||
      buffer.size(1) != packed_scale_k || buffer.stride(0) != 1 ||
      buffer.stride(1) != aligned_outer) {
    buffer = allocate_legacy_col_major_scale_buffer(scale, outer_dim, reduction_size);
  }
  buffer.copy_(scale);
  return buffer;
}

CUtensorMapDataType aten_dtype_to_tensor_map_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kInt:
      return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case at::kUInt32:
      return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    case at::kFloat:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case at::kBFloat16:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case at::kFloat8_e4m3fn:
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    default:
      TORCH_CHECK(false, "Unsupported tensor map dtype ", dtype);
  }
}

CUtensorMapSwizzle mode_into_tensor_map_swizzle(int mode, int base = 0) {
#if CUDART_VERSION >= 12080
  if (base != 0) {
    TORCH_CHECK(base == 32 && mode == 128,
                "Unsupported swizzle base/mode combination");
    return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
  }
#endif

  TORCH_CHECK(base == 0, "Unsupported non-zero swizzle base");
  switch (mode) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      TORCH_CHECK(false, "Unsupported swizzle mode ", mode);
  }
}

CUtensorMap make_tma_2d_desc(torch::Tensor const &t,
                             int gmem_inner_dim,
                             int gmem_outer_dim,
                             int smem_inner_dim,
                             int smem_outer_dim,
                             int gmem_outer_stride,
                             int swizzle_mode,
                             int swizzle_base = 0) {
  auto const elem_size = static_cast<int>(t.element_size());
  if (swizzle_mode != 0) {
    smem_inner_dim = swizzle_mode / elem_size;
  }

  CUtensorMap tensor_map{};
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim),
                                   static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim),
                                   static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride * elem_size)};
  const cuuint32_t elem_strides[2] = {1, 1};
  check_driver_success(
      cuTensorMapEncodeTiled(
          &tensor_map,
          aten_dtype_to_tensor_map_dtype(t.scalar_type()),
          2,
          t.data_ptr(),
          gmem_dims,
          gmem_strides,
          smem_dims,
          elem_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          mode_into_tensor_map_swizzle(swizzle_mode, swizzle_base),
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tensor_map;
}

CUtensorMap make_tma_a_desc(torch::Tensor const &t,
                            int shape_m,
                            int shape_k,
                            int block_m,
                            int block_k,
                            int outer_stride,
                            int swizzle_mode) {
  return make_tma_2d_desc(
      t, shape_k, shape_m, block_k, block_m, outer_stride, swizzle_mode);
}

CUtensorMap make_tma_b_desc(torch::Tensor const &t,
                            int shape_n,
                            int shape_k,
                            int block_n,
                            int block_k,
                            int outer_stride,
                            int swizzle_mode) {
  return make_tma_2d_desc(
      t, shape_k, shape_n, block_k, block_n, outer_stride, swizzle_mode);
}

CUtensorMap make_tma_cd_desc(torch::Tensor const &t,
                             int shape_m,
                             int shape_n,
                             int block_m,
                             int block_n,
                             int outer_stride,
                             int swizzle_mode) {
  return make_tma_2d_desc(
      t, shape_n, shape_m, block_n, block_m, outer_stride, swizzle_mode);
}

CUtensorMap make_tma_sf_desc(torch::Tensor const &t,
                             int outer_dim,
                             int reduction_size,
                             int block_outer,
                             int gran_k) {
  int aligned_outer = fp8_runtime::aligned_scale_outer_dim(outer_dim);
  int packed_scale_k =
      fp8_runtime::packed_scale_k_for_reduction_size(reduction_size);
  return make_tma_2d_desc(
      t, aligned_outer, packed_scale_k, block_outer, 1, aligned_outer, 0);
}

template <int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int kSwizzleCDMode,
          int kNumStages,
          bool kWithResidual>
constexpr int linear_fp8_dynamic_smem_bytes() {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kLayoutAdM = 128;
  constexpr int kNumTmaStoreStages = kWithResidual ? 1 : 2;
  constexpr int kNumUtccpAlignedElems = 128;
  constexpr int kNumMwaves =
      BLOCK_M / (BLOCK_M < kLayoutAdM ? BLOCK_M : kLayoutAdM);
  constexpr int kStoreBlockM = BLOCK_M < kLayoutAdM ? BLOCK_M : kLayoutAdM;
  constexpr int kSmemCdSizePerStage = kStoreBlockM * kSwizzleCDMode;
  constexpr int kSmemCdSize = kSmemCdSizePerStage * kNumTmaStoreStages;
  constexpr int kSmemResidualSize =
      kWithResidual ? kStoreBlockM * kSwizzleCDMode : 0;
  constexpr int kSfBlockM =
      ((BLOCK_M + kNumUtccpAlignedElems - 1) / kNumUtccpAlignedElems) *
      kNumUtccpAlignedElems;
  constexpr int kSfBlockN =
      ((BLOCK_N + kNumUtccpAlignedElems - 1) / kNumUtccpAlignedElems) *
      kNumUtccpAlignedElems;
  constexpr int kNumSfATmemCols = kSfBlockM / 32;
  constexpr int kNumSfBTmemCols = kSfBlockN / 32;
  constexpr int kNumEpilogueStages =
      (2 * kNumMwaves * BLOCK_N + kNumSfATmemCols + kNumSfBTmemCols) > 512 ? 1
                                                                           : 2;
  constexpr int kSmemASize =
      kNumStages * BLOCK_M * BLOCK_K * sizeof(cutlass::float_e4m3_t);
  constexpr int kSmemBSize =
      kNumStages * BLOCK_N * BLOCK_K * sizeof(cutlass::float_e4m3_t);
  constexpr int kSmemSfASize = kNumStages * kSfBlockM * sizeof(uint32_t);
  constexpr int kSmemSfBSize = kNumStages * kSfBlockN * sizeof(uint32_t);
  constexpr int kNumBarriers =
      kNumStages * 3 + kNumEpilogueStages * 2 + (kWithResidual ? 2 : 0);
  constexpr int kSmemBytesBeforeBarriers = kSmemCdSize + kSmemResidualSize +
                                           kSmemASize + kSmemBSize +
                                           kSmemSfASize + kSmemSfBSize;
  constexpr int kBarrierBytes = kNumBarriers * sizeof(Barrier);
  constexpr int kTmemPtrBytes = sizeof(uint32_t);
  constexpr int kPaddingBytes = 8;
  return kSmemBytesBeforeBarriers + kBarrierBytes + kTmemPtrBytes +
         kPaddingBytes;
}

template <typename KernelPtr, typename... Args>
void launch_kernel_ex_cluster(dim3 grid_dim,
                              dim3 block_dim,
                              dim3 cluster_dim,
                              int smem_bytes,
                              cudaStream_t cuda_stream,
                              KernelPtr kernel_ptr,
                              Args... args) {
  cudaLaunchAttribute launch_attribute{};
  launch_attribute.id = cudaLaunchAttributeClusterDimension;
  launch_attribute.val.clusterDim = {
      cluster_dim.x, cluster_dim.y, cluster_dim.z};

  cudaLaunchConfig_t launch_config{};
  launch_config.gridDim = grid_dim;
  launch_config.blockDim = block_dim;
  launch_config.dynamicSmemBytes = smem_bytes;
  launch_config.stream = cuda_stream;
  launch_config.numAttrs = 1;
  launch_config.attrs = &launch_attribute;

  void *kernel_params[] = {const_cast<void *>(
      reinterpret_cast<void const *>(std::addressof(args)))...};
  if (cluster_dim.x == 1 && cluster_dim.y == 1 && cluster_dim.z == 1) {
    CUTE_CHECK_ERROR(cudaLaunchKernel((void const *)kernel_ptr,
                                      grid_dim,
                                      block_dim,
                                      kernel_params,
                                      smem_bytes,
                                      cuda_stream));
  } else {
    CUTE_CHECK_ERROR(
        cudaLaunchKernelExC(&launch_config, (void *)kernel_ptr, kernel_params));
  }
}

} // namespace

template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          bool kWithResidual>
void launch_linear_fp8_1d1d_sm100(torch::Tensor const &input_q,
                                  torch::Tensor const &input_scale,
                                  torch::Tensor const &weight_q,
                                  torch::Tensor const &weight_scale,
                                  torch::Tensor const *residual,
                                  torch::Tensor &output) {
  static_assert(OUTPUT_SIZE == 128, "Current path requires N=128");
  constexpr int BLOCK_M = 32;
  constexpr int BLOCK_N = 16;
  constexpr int BLOCK_K = 128;
  constexpr int kGranKA = 128;
  constexpr int kGranKB = 128;
  constexpr int kSwizzleAMode = 128;
  constexpr int kSwizzleBMode = 128;
  constexpr int kSwizzleCDMode = 32;
  constexpr int kNumStages = 28;
  constexpr int kNumNonEpilogueThreads = 128;
  constexpr int kNumEpilogueThreads = 128;
  constexpr int kNumMulticast = 1;
  constexpr bool kIsMulticastOnA = false;
  constexpr int kNumSMs = 8;
  constexpr int kSm100MaxDynamicSmemBytes = 228 * 1024;
  constexpr int kSchedulerSmemReservationBytes = 6 * 1024;
  constexpr int kDynamicSmemBytes =
      linear_fp8_dynamic_smem_bytes<BLOCK_M,
                                    BLOCK_N,
                                    BLOCK_K,
                                    kSwizzleCDMode,
                                    kNumStages,
                                    kWithResidual>();
  static_assert(linear_fp8_dynamic_smem_bytes<BLOCK_M,
                                              BLOCK_N,
                                              BLOCK_K,
                                              kSwizzleCDMode,
                                              kNumStages,
                                              false>() == 203468,
                "Unexpected no-residual dynamic shared memory size");
  static_assert(linear_fp8_dynamic_smem_bytes<BLOCK_M,
                                              BLOCK_N,
                                              BLOCK_K,
                                              kSwizzleCDMode,
                                              kNumStages,
                                              true>() == 203484,
                "Unexpected fused-residual dynamic shared memory size");
  static_assert(kDynamicSmemBytes + kSchedulerSmemReservationBytes <=
                    kSm100MaxDynamicSmemBytes,
                "SM100 FP8 fast path must leave scheduler shared memory headroom");

  auto &cache = get_linear_fp8_descriptor_cache(LinearFp8RuntimeKey{
      BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 1, kWithResidual});
  std::lock_guard<std::mutex> lock(cache.mutex);

  auto update_desc = [](CUtensorMap &desc, void *ptr, char const *name) {
    check_driver_success(cuTensorMapReplaceAddress(&desc, ptr), name);
  };

  void *input_ptr = input_q.data_ptr();
  void *weight_ptr = weight_q.data_ptr();
  torch::Tensor const &internal_input_scale = materialize_internal_scale_tensor(
      cache, input_scale, BATCH_SIZE, REDUCTION_SIZE, true);
  torch::Tensor const &internal_weight_scale = materialize_internal_scale_tensor(
      cache, weight_scale, OUTPUT_SIZE, REDUCTION_SIZE, false);
  void *input_scale_ptr = internal_input_scale.data_ptr();
  void *weight_scale_ptr = internal_weight_scale.data_ptr();
  void *residual_ptr = nullptr;
  if constexpr (kWithResidual) {
    TORCH_CHECK(residual != nullptr, "residual tensor is required");
    residual_ptr = residual->data_ptr();
  }
  void *output_ptr = output.data_ptr();

  if (!cache.initialized) {
    cache.host_input_desc = make_tma_a_desc(input_q,
                                            BATCH_SIZE,
                                            REDUCTION_SIZE,
                                            BLOCK_M,
                                            BLOCK_K,
                                            static_cast<int>(input_q.stride(0)),
                                            kSwizzleAMode);
    cache.host_weight_desc =
        make_tma_b_desc(weight_q,
                        OUTPUT_SIZE,
                        REDUCTION_SIZE,
                        BLOCK_N,
                        BLOCK_K,
                        static_cast<int>(weight_q.stride(0)),
                        kSwizzleBMode);
    cache.host_input_scale_desc = make_tma_sf_desc(
        internal_input_scale, BATCH_SIZE, REDUCTION_SIZE, BLOCK_M, kGranKA);
    cache.host_weight_scale_desc = make_tma_sf_desc(
        internal_weight_scale, OUTPUT_SIZE, REDUCTION_SIZE, BLOCK_N, kGranKB);
    if constexpr (kWithResidual) {
      cache.host_residual_desc =
          make_tma_cd_desc(*residual,
                           BATCH_SIZE,
                           OUTPUT_SIZE,
                           BLOCK_M,
                           BLOCK_N,
                           static_cast<int>(residual->stride(0)),
                           kSwizzleCDMode);
    }
    cache.host_output_desc =
        make_tma_cd_desc(output,
                         BATCH_SIZE,
                         OUTPUT_SIZE,
                         BLOCK_M,
                         BLOCK_N,
                         static_cast<int>(output.stride(0)),
                         kSwizzleCDMode);
    cache.last_input_ptr = input_ptr;
    cache.last_weight_ptr = weight_ptr;
    cache.last_input_scale_ptr = input_scale_ptr;
    cache.last_weight_scale_ptr = weight_scale_ptr;
    cache.last_residual_ptr = residual_ptr;
    cache.last_output_ptr = output_ptr;
    cache.initialized = true;
  } else {
    if (input_ptr != cache.last_input_ptr) {
      update_desc(
          cache.host_input_desc, input_ptr, "cuTensorMapReplaceAddress(input)");
      cache.last_input_ptr = input_ptr;
    }
    if (weight_ptr != cache.last_weight_ptr) {
      update_desc(cache.host_weight_desc,
                  weight_ptr,
                  "cuTensorMapReplaceAddress(weight)");
      cache.last_weight_ptr = weight_ptr;
    }
    if (input_scale_ptr != cache.last_input_scale_ptr) {
      update_desc(cache.host_input_scale_desc,
                  input_scale_ptr,
                  "cuTensorMapReplaceAddress(input_scale)");
      cache.last_input_scale_ptr = input_scale_ptr;
    }
    if (weight_scale_ptr != cache.last_weight_scale_ptr) {
      update_desc(cache.host_weight_scale_desc,
                  weight_scale_ptr,
                  "cuTensorMapReplaceAddress(weight_scale)");
      cache.last_weight_scale_ptr = weight_scale_ptr;
    }
    if constexpr (kWithResidual) {
      if (residual_ptr != cache.last_residual_ptr) {
        update_desc(cache.host_residual_desc,
                    residual_ptr,
                    "cuTensorMapReplaceAddress(residual)");
        cache.last_residual_ptr = residual_ptr;
      }
    }
    if (output_ptr != cache.last_output_ptr) {
      update_desc(cache.host_output_desc,
                  output_ptr,
                  "cuTensorMapReplaceAddress(output)");
      cache.last_output_ptr = output_ptr;
    }
  }

  using Kernel =
      decltype(&kernel::linear_fp8_sm100_wrapper<cute::UMMA::Major::K,
                                                 cute::UMMA::Major::K,
                                                 kGranKA,
                                                 kGranKB,
                                                 BATCH_SIZE,
                                                 OUTPUT_SIZE,
                                                 REDUCTION_SIZE,
                                                 BLOCK_M,
                                                 BLOCK_N,
                                                 BLOCK_K,
                                                 1,
                                                 kSwizzleAMode,
                                                 kSwizzleBMode,
                                                 kSwizzleCDMode,
                                                 kNumStages,
                                                 kNumNonEpilogueThreads,
                                                 kNumEpilogueThreads,
                                                 kNumMulticast,
                                                 kIsMulticastOnA,
                                                 kNumSMs,
                                                 kWithResidual,
                                                 fp8_linear::GemmType::Normal,
                                                 false,
                                                 cutlass::float_e4m3_t,
                                                 cutlass::float_e4m3_t,
                                                 cutlass::bfloat16_t,
                                                 fp8_linear::EpilogueIdentity>);

  auto *kernel_ptr =
      &kernel::linear_fp8_sm100_wrapper<cute::UMMA::Major::K,
                                        cute::UMMA::Major::K,
                                        kGranKA,
                                        kGranKB,
                                        BATCH_SIZE,
                                        OUTPUT_SIZE,
                                        REDUCTION_SIZE,
                                        BLOCK_M,
                                        BLOCK_N,
                                        BLOCK_K,
                                        1,
                                        kSwizzleAMode,
                                        kSwizzleBMode,
                                        kSwizzleCDMode,
                                        kNumStages,
                                        kNumNonEpilogueThreads,
                                        kNumEpilogueThreads,
                                        kNumMulticast,
                                        kIsMulticastOnA,
                                        kNumSMs,
                                        kWithResidual,
                                        fp8_linear::GemmType::Normal,
                                        false,
                                        cutlass::float_e4m3_t,
                                        cutlass::float_e4m3_t,
                                        cutlass::bfloat16_t,
                                        fp8_linear::EpilogueIdentity>;

  if (!cache.kernel_configured) {
    CUTE_CHECK_ERROR(
        cudaFuncSetAttribute(kernel_ptr,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kDynamicSmemBytes));
    cache.kernel_configured = true;
  }

  dim3 grid_dim(kNumSMs, 1, 1);
  dim3 block_dim(kNumNonEpilogueThreads + kNumEpilogueThreads, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  int *grouped_layout = nullptr;
  uint32_t shape_m = BATCH_SIZE;
  uint32_t shape_n = OUTPUT_SIZE;
  uint32_t shape_k = REDUCTION_SIZE;
  CUtensorMap const &residual_desc =
      kWithResidual ? cache.host_residual_desc : cache.host_output_desc;
  launch_kernel_ex_cluster(grid_dim,
                           block_dim,
                           cluster_dim,
                           kDynamicSmemBytes,
                           cuda_stream,
                           kernel_ptr,
                           grouped_layout,
                           shape_m,
                           shape_n,
                           shape_k,
                           cache.host_input_desc,
                           cache.host_weight_desc,
                           cache.host_input_scale_desc,
                           cache.host_weight_scale_desc,
                           residual_desc,
                           cache.host_output_desc);
  check_last_launch(kWithResidual ? "linear_fp8_1d1d_sm100_with_residual"
                                  : "linear_fp8_1d1d_sm100");
}

template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int SPLIT_K>
void launch_linear_fp8_1d1d_sm100_splitk(torch::Tensor const &input_q,
                                         torch::Tensor const &input_scale,
                                         torch::Tensor const &weight_q,
                                         torch::Tensor const &weight_scale,
                                         torch::Tensor &output) {
  static_assert(OUTPUT_SIZE == 128, "Current path requires N=128");
  static_assert(SPLIT_K == 2 || SPLIT_K == 4 || SPLIT_K == 8,
                "Unsupported split-K factor");
  constexpr int BLOCK_M = 32;
  constexpr int BLOCK_N = 16;
  constexpr int BLOCK_K = 128;
  constexpr int kGranKA = 128;
  constexpr int kGranKB = 128;
  constexpr int kSwizzleAMode = 128;
  constexpr int kSwizzleBMode = 128;
  constexpr int kSwizzleCDMode = 64;
  constexpr int kNumStages = 28;
  constexpr int kNumNonEpilogueThreads = 128;
  constexpr int kNumEpilogueThreads = 128;
  constexpr int kNumMulticast = 1;
  constexpr bool kIsMulticastOnA = false;
  constexpr int kSm100MaxDynamicSmemBytes = 228 * 1024;
  constexpr int kSchedulerSmemReservationBytes = 6 * 1024;
  constexpr int kDynamicSmemBytes =
      linear_fp8_dynamic_smem_bytes<BLOCK_M,
                                    BLOCK_N,
                                    BLOCK_K,
                                    kSwizzleCDMode,
                                    kNumStages,
                                    false>();
  static_assert(linear_fp8_dynamic_smem_bytes<BLOCK_M,
                                              BLOCK_N,
                                              BLOCK_K,
                                              kSwizzleCDMode,
                                              kNumStages,
                                              false>() == 205516,
                "Unexpected split-K dynamic shared memory size");
  static_assert(kDynamicSmemBytes + kSchedulerSmemReservationBytes <=
                    kSm100MaxDynamicSmemBytes,
                "SM100 FP8 split-K path must leave scheduler shared memory headroom");

  auto &cache = get_linear_fp8_descriptor_cache(LinearFp8RuntimeKey{
      BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, SPLIT_K, false});
  std::lock_guard<std::mutex> lock(cache.mutex);

  auto update_desc = [](CUtensorMap &desc, void *ptr, char const *name) {
    check_driver_success(cuTensorMapReplaceAddress(&desc, ptr), name);
  };

  void *input_ptr = input_q.data_ptr();
  void *weight_ptr = weight_q.data_ptr();
  torch::Tensor const &internal_input_scale = materialize_internal_scale_tensor(
      cache, input_scale, BATCH_SIZE, REDUCTION_SIZE, true);
  torch::Tensor const &internal_weight_scale = materialize_internal_scale_tensor(
      cache, weight_scale, OUTPUT_SIZE, REDUCTION_SIZE, false);
  void *input_scale_ptr = internal_input_scale.data_ptr();
  void *weight_scale_ptr = internal_weight_scale.data_ptr();
  if (!cache.splitk_accum_buffer.defined() ||
      cache.splitk_accum_buffer.size(0) != BATCH_SIZE ||
      cache.splitk_accum_buffer.size(1) != OUTPUT_SIZE ||
      cache.splitk_accum_buffer.device() != output.device()) {
    cache.splitk_accum_buffer =
        torch::zeros({BATCH_SIZE, OUTPUT_SIZE},
                     output.options()
                         .dtype(torch::kFloat)
                         .memory_format(c10::MemoryFormat::Contiguous));
  }
  void *output_ptr = cache.splitk_accum_buffer.data_ptr();

  if (!cache.initialized) {
    cache.host_input_desc = make_tma_a_desc(input_q,
                                            BATCH_SIZE,
                                            REDUCTION_SIZE,
                                            BLOCK_M,
                                            BLOCK_K,
                                            static_cast<int>(input_q.stride(0)),
                                            kSwizzleAMode);
    cache.host_weight_desc =
        make_tma_b_desc(weight_q,
                        OUTPUT_SIZE,
                        REDUCTION_SIZE,
                        BLOCK_N,
                        BLOCK_K,
                        static_cast<int>(weight_q.stride(0)),
                        kSwizzleBMode);
    cache.host_input_scale_desc = make_tma_sf_desc(
        internal_input_scale, BATCH_SIZE, REDUCTION_SIZE, BLOCK_M, kGranKA);
    cache.host_weight_scale_desc = make_tma_sf_desc(
        internal_weight_scale, OUTPUT_SIZE, REDUCTION_SIZE, BLOCK_N, kGranKB);
    cache.host_output_desc =
        make_tma_cd_desc(cache.splitk_accum_buffer,
                         BATCH_SIZE,
                         OUTPUT_SIZE,
                         BLOCK_M,
                         BLOCK_N,
                         static_cast<int>(cache.splitk_accum_buffer.stride(0)),
                         kSwizzleCDMode);
    cache.last_input_ptr = input_ptr;
    cache.last_weight_ptr = weight_ptr;
    cache.last_input_scale_ptr = input_scale_ptr;
    cache.last_weight_scale_ptr = weight_scale_ptr;
    cache.last_output_ptr = output_ptr;
    cache.initialized = true;
  } else {
    if (input_ptr != cache.last_input_ptr) {
      update_desc(
          cache.host_input_desc, input_ptr, "cuTensorMapReplaceAddress(input)");
      cache.last_input_ptr = input_ptr;
    }
    if (weight_ptr != cache.last_weight_ptr) {
      update_desc(cache.host_weight_desc,
                  weight_ptr,
                  "cuTensorMapReplaceAddress(weight)");
      cache.last_weight_ptr = weight_ptr;
    }
    if (input_scale_ptr != cache.last_input_scale_ptr) {
      update_desc(cache.host_input_scale_desc,
                  input_scale_ptr,
                  "cuTensorMapReplaceAddress(input_scale)");
      cache.last_input_scale_ptr = input_scale_ptr;
    }
    if (weight_scale_ptr != cache.last_weight_scale_ptr) {
      update_desc(cache.host_weight_scale_desc,
                  weight_scale_ptr,
                  "cuTensorMapReplaceAddress(weight_scale)");
      cache.last_weight_scale_ptr = weight_scale_ptr;
    }
    if (output_ptr != cache.last_output_ptr) {
      update_desc(cache.host_output_desc,
                  output_ptr,
                  "cuTensorMapReplaceAddress(output)");
      cache.last_output_ptr = output_ptr;
    }
  }

  auto *kernel_ptr =
      &kernel::linear_fp8_sm100_splitk_wrapper<cute::UMMA::Major::K,
                                               cute::UMMA::Major::K,
                                               kGranKA,
                                               kGranKB,
                                               BATCH_SIZE,
                                               OUTPUT_SIZE,
                                               REDUCTION_SIZE,
                                               BLOCK_M,
                                               BLOCK_N,
                                               BLOCK_K,
                                               kSwizzleAMode,
                                               kSwizzleBMode,
                                               kSwizzleCDMode,
                                               kNumStages,
                                               kNumNonEpilogueThreads,
                                               kNumEpilogueThreads,
                                               kNumMulticast,
                                               kIsMulticastOnA,
                                               SPLIT_K,
                                               cutlass::float_e4m3_t,
                                               cutlass::float_e4m3_t,
                                               float,
                                               fp8_linear::EpilogueIdentity>;

  if (!cache.kernel_configured) {
    CUTE_CHECK_ERROR(
        cudaFuncSetAttribute(kernel_ptr,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kDynamicSmemBytes));
    cache.kernel_configured = true;
  }

  constexpr int kNumOutputTiles =
      ((BATCH_SIZE + BLOCK_M - 1) / BLOCK_M) *
      ((OUTPUT_SIZE + BLOCK_N - 1) / BLOCK_N);

  cache.splitk_accum_buffer.zero_();

  dim3 grid_dim(kNumOutputTiles * SPLIT_K, 1, 1);
  dim3 block_dim(kNumNonEpilogueThreads + kNumEpilogueThreads, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream().stream();
  uint32_t shape_m = BATCH_SIZE;
  uint32_t shape_n = OUTPUT_SIZE;
  uint32_t shape_k = REDUCTION_SIZE;
  launch_kernel_ex_cluster(grid_dim,
                           block_dim,
                           cluster_dim,
                           kDynamicSmemBytes,
                           cuda_stream,
                           kernel_ptr,
                           shape_m,
                           shape_n,
                           shape_k,
                           cache.host_input_desc,
                           cache.host_weight_desc,
                           cache.host_input_scale_desc,
                           cache.host_weight_scale_desc,
                           cache.host_output_desc);
  check_last_launch("linear_fp8_1d1d_sm100_splitk");
  output.copy_(cache.splitk_accum_buffer);
}

void linear_fp8_1d2d_sm100_kernel(torch::Tensor input_q,
                                  torch::Tensor input_scale,
                                  torch::Tensor weight_q,
                                  torch::Tensor weight_scale,
                                  c10::optional<at::Tensor> residual,
                                  torch::Tensor output) {
  TORCH_CHECK(input_q.dim() == 2, "input_q must be 2D");
  TORCH_CHECK(input_scale.dim() == 2, "input_scale must be 2D");
  TORCH_CHECK(weight_q.dim() == 2, "weight_q must be 2D");
  TORCH_CHECK(weight_scale.dim() == 2, "weight_scale must be 2D");
  TORCH_CHECK(output.dim() == 2, "output must be 2D");
  TORCH_CHECK(input_q.is_contiguous(), "input_q must be contiguous");
  TORCH_CHECK(weight_q.is_contiguous(), "weight_q must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input_q.scalar_type() == at::kFloat8_e4m3fn,
              "input_q must be float8_e4m3fn");
  TORCH_CHECK(weight_q.scalar_type() == at::kFloat8_e4m3fn,
              "weight_q must be float8_e4m3fn");
  TORCH_CHECK(input_scale.scalar_type() == at::kInt ||
                  input_scale.scalar_type() == at::kUInt32,
              "input_scale must be uint32-compatible");
  TORCH_CHECK(weight_scale.scalar_type() == at::kInt ||
                  weight_scale.scalar_type() == at::kUInt32,
              "weight_scale must be uint32-compatible");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");

  int const batch_size = static_cast<int>(input_q.size(0));
  int const reduction_size = static_cast<int>(input_q.size(1));
  int const output_size = static_cast<int>(weight_q.size(0));

  TORCH_CHECK(weight_q.size(1) == reduction_size,
              "weight_q shape mismatch: expected reduction_size ",
              reduction_size,
              " in dim-1 but got ",
              weight_q.size(1));
  TORCH_CHECK(output.size(0) == batch_size && output.size(1) == output_size,
              "output shape mismatch: expected [",
              batch_size,
              ", ",
              output_size,
              "] but got [",
              output.size(0),
              ", ",
              output.size(1),
              "]");
  mirage::blackwell::sm100_fp8_scale_layout::check_scale_tensor(
      input_scale, batch_size, reduction_size, "input_scale");
  mirage::blackwell::sm100_fp8_scale_layout::check_scale_tensor(
      weight_scale, output_size, reduction_size, "weight_scale");

  TORCH_CHECK(fp8_runtime::is_supported_dense_gemm_shape(
                  batch_size, output_size, reduction_size),
              "Unsupported linear_fp8_1d2d_sm100 shape [B=",
              batch_size,
              ", N=",
              output_size,
              ", K=",
              reduction_size,
              "]. Supported batch sizes: {",
              fp8_runtime::supported_batch_sizes_string(),
              "}, output sizes: {",
              fp8_runtime::supported_output_sizes_string(),
              "}, reduction sizes: {",
              fp8_runtime::supported_reduction_sizes_string(),
              "}");

  bool const has_residual = residual.has_value();
  if (has_residual) {
    TORCH_CHECK(residual->dim() == 2, "residual must be 2D");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual->scalar_type() == at::kBFloat16,
                "residual must be bfloat16");
    TORCH_CHECK(residual->size(0) == batch_size &&
                    residual->size(1) == output_size,
                "residual shape mismatch: expected [",
                batch_size,
                ", ",
                output_size,
                "] but got [",
                residual->size(0),
                ", ",
                residual->size(1),
                "]");
  }
  int const split_k = choose_split_k(batch_size, reduction_size, has_residual);
#define DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                               \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, WITH_RESIDUAL)                    \
  case REDUCTION_SIZE:                                                         \
    launch_linear_fp8_1d1d_sm100<BATCH_SIZE,                                   \
                                 OUTPUT_SIZE,                                  \
                                 REDUCTION_SIZE,                               \
                                 WITH_RESIDUAL>(input_q,                       \
                                                input_scale,                   \
                                                weight_q,                      \
                                                weight_scale,                  \
                                                has_residual ? &(*residual)    \
                                                             : nullptr,        \
                                                output);                       \
    break;

#define DISPATCH_LINEAR_FP8_SPLITK_CASE(                                       \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, SPLIT_K_VALUE)                    \
  case SPLIT_K_VALUE:                                                          \
    launch_linear_fp8_1d1d_sm100_splitk<BATCH_SIZE,                            \
                                        OUTPUT_SIZE,                           \
                                        REDUCTION_SIZE,                        \
                                        SPLIT_K_VALUE>(                        \
        input_q, input_scale, weight_q, weight_scale, output);                 \
    break;

#define DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(                   \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)                                   \
  case REDUCTION_SIZE:                                                         \
    switch (split_k) {                                                         \
      DISPATCH_LINEAR_FP8_SPLITK_CASE(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 2) \
      DISPATCH_LINEAR_FP8_SPLITK_CASE(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 4) \
      DISPATCH_LINEAR_FP8_SPLITK_CASE(BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 8) \
      default:                                                                 \
        TORCH_CHECK(false, "Unsupported split_k dispatch");                   \
    }                                                                          \
    break;

#define DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(BATCH_SIZE, WITH_RESIDUAL)         \
  case BATCH_SIZE:                                                             \
    switch (reduction_size) {                                                  \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 128, WITH_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 256, WITH_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 384, WITH_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 512, WITH_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 768, WITH_RESIDUAL)                                 \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 1024, WITH_RESIDUAL)                                \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 1536, WITH_RESIDUAL)                                \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 2048, WITH_RESIDUAL)                                \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 4096, WITH_RESIDUAL)                                \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE(                                 \
          BATCH_SIZE, 128, 7168, WITH_RESIDUAL)                                \
      default:                                                                 \
        TORCH_CHECK(false, "Unsupported reduction_size dispatch");             \
    }                                                                          \
    break;

#define DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE_WITH_SPLITK(BATCH_SIZE)            \
  case BATCH_SIZE:                                                             \
    switch (reduction_size) {                                                  \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 128) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 256) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 384) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 512) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 768) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 1024) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 1536) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 2048) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 4096) \
      DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK(BATCH_SIZE, 128, 7168) \
      default:                                                                 \
        TORCH_CHECK(false, "Unsupported reduction_size dispatch");             \
    }                                                                          \
    break;

  if (has_residual) {
    switch (batch_size) {
      DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(1, true)
      DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(2, true)
      DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(4, true)
      DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(8, true)
      DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(16, true)
      default:
        TORCH_CHECK(false, "Unsupported batch_size dispatch");
    }
  } else {
    if (split_k == 1) {
      switch (batch_size) {
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(1, false)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(2, false)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(4, false)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(8, false)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE(16, false)
        default:
          TORCH_CHECK(false, "Unsupported batch_size dispatch");
      }
    } else {
      switch (batch_size) {
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE_WITH_SPLITK(1)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE_WITH_SPLITK(2)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE_WITH_SPLITK(4)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE_WITH_SPLITK(8)
        DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE_WITH_SPLITK(16)
        default:
          TORCH_CHECK(false, "Unsupported batch_size dispatch");
      }
    }
  }

#undef DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE_WITH_SPLITK
#undef DISPATCH_LINEAR_FP8_BATCH_SIZE_CASE
#undef DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE_WITH_SPLITK
#undef DISPATCH_LINEAR_FP8_SPLITK_CASE
#undef DISPATCH_LINEAR_FP8_REDUCTION_SIZE_CASE
}

std::vector<std::vector<int64_t>> supported_dense_gemm_shapes() {
  return fp8_runtime::supported_dense_gemm_shapes_vector();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fp8_1d2d_sm100",
        &linear_fp8_1d2d_sm100_kernel,
        "Linear kernel SM100 FP8 1D2D");
  m.def("supported_dense_gemm_shapes",
        &supported_dense_gemm_shapes,
        "Supported SM100 FP8 dense GEMM shapes");
}
