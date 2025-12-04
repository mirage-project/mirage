/* Copyright 2025 CMU
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
#include "runtime_header.h"
#include "blackwell/task_header.cuh"
#include "hopper/tma_4d.cuh"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h> // F16 data type
#include <cutlass/util/print_error.hpp>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp> // CuTe tensor implementation

using bfloat16 = cute::bfloat16_t;

// mixed_attn_sm100_kernel

template <typename T,
          class QTensor,
          class OTensor,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM_QK,
          int HEAD_DIM_V,
          int MAX_NUM_PAGES,
          int PAGE_SIZE,
          bool CAUSAL=true,
          int NUM_BUCKETS=128,
          int MIN_KV_CHUNK_SIZE=512,
          int P_Q_TILE_SIZE=128,
          int P_KV_TILE_SIZE=128,
          int D_Q_TILE_SIZE=16,
          int D_KV_TILE_SIZE=128>
__global__ __launch_bounds__(256) void mixed_attn_sm100_kernel_wrapper(
  QTensor mQ,
  void *tma_k_desc_ptr,
  void *tma_v_desc_ptr,
  OTensor mO,
  void const *decode_work_indptr,
  void const *prefill_work_indptr,
  void const *worker_batch_indices_ptr,
  void const *worker_kv_head_indices_ptr,
  void const *worker_packed_qo_indices_ptr,
  void const *worker_kv_start_ptr,
  void const *worker_kv_end_ptr,
  void const *paged_kv_indices_buffer_ptr,
  void const *paged_kv_indptr_buffer_ptr,
  void const *paged_kv_last_page_len_buffer_ptr,
  int worker_idx) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size

  constexpr size_t k_smem_repeat_col =
      (HEAD_DIM_QK + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  constexpr size_t v_smem_repeat_col =
      (HEAD_DIM_V + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  using TMA_K =
      kernel::tma::tma_4d<
        bfloat16,
        B,
        M,
        S,
        MAX_NUM_PAGES,                          /* GMEM_OUTERMOST_ */
        PAGE_SIZE,                              /* GMEM_DEPTH   */
        NUM_KV_HEADS,                           /* GMEM_ROW   */
        HEAD_DIM_QK,                            /* GMEM_COL   */
        1,                                      /* SMEM_OUTERMOST_ */
        P_KV_TILE_SIZE,                         /* SMEM_DEPTH   */
        1,                                      /* SMEM_ROW   */
        TMA_CP_ASYNC_SIZE,                      /* SMEM_COL   */
        PAGE_SIZE * HEAD_DIM_QK * NUM_KV_HEADS, /* GMEM_STRIDE_OUTERMOST_ */
        PAGE_SIZE * HEAD_DIM_QK,                /* GMEM_STRIDE_DEPTH */
        HEAD_DIM_QK,                            /* GMEM_STRIDE_ROW   */
        1,                                      /* GMEM_STRIDE_COL   */
        1,                                      /* SMEM_REPEAT_ROW   */
        k_smem_repeat_col,                      /* SMEM_REPEAT_COL   */
        P_KV_TILE_SIZE * TMA_CP_ASYNC_SIZE,     /* SMEM_STRIDE       */
        true>;

  using TMA_V =
      kernel::tma::tma_4d<
        bfloat16,
        B,
        M,
        S,
        MAX_NUM_PAGES,                          /* GMEM_OUTERMOST_ */
        PAGE_SIZE,                              /* GMEM_DEPTH   */
        NUM_KV_HEADS,                           /* GMEM_ROW   */
        HEAD_DIM_V,                            /* GMEM_COL   */
        1,                                      /* SMEM_OUTERMOST_ */
        P_KV_TILE_SIZE,                         /* SMEM_DEPTH   */
        1,                                      /* SMEM_ROW   */
        TMA_CP_ASYNC_SIZE,                      /* SMEM_COL   */
        PAGE_SIZE * HEAD_DIM_V * NUM_KV_HEADS, /* GMEM_STRIDE_OUTERMOST_ */
        PAGE_SIZE * HEAD_DIM_V,                /* GMEM_STRIDE_DEPTH */
        HEAD_DIM_V,                            /* GMEM_STRIDE_ROW   */
        1,                                      /* GMEM_STRIDE_COL   */
        1,                                      /* SMEM_REPEAT_ROW   */
        v_smem_repeat_col,                      /* SMEM_REPEAT_COL   */
        P_KV_TILE_SIZE * TMA_CP_ASYNC_SIZE,     /* SMEM_STRIDE       */
        true>;
  
  TMA_K tma_k(static_cast<CUtensorMap *>(tma_k_desc_ptr));
  TMA_V tma_v(static_cast<CUtensorMap *>(tma_v_desc_ptr));

  kernel::mixed_attention_sm100_task_impl<
    T,
    QTensor,
    TMA_K,
    TMA_V,
    OTensor,
    NUM_QO_HEADS,
    NUM_KV_HEADS,
    HEAD_DIM_QK,
    HEAD_DIM_V,
    PAGE_SIZE,
    CAUSAL,
    NUM_BUCKETS,
    MIN_KV_CHUNK_SIZE,
    P_Q_TILE_SIZE,
    P_KV_TILE_SIZE,
    D_Q_TILE_SIZE,
    D_KV_TILE_SIZE>(
        mQ,
        tma_k,
        tma_v,
        mO,
        static_cast<int const *>(decode_work_indptr),
        static_cast<int const *>(prefill_work_indptr),
        static_cast<int const *>(worker_batch_indices_ptr),
        static_cast<int const *>(worker_kv_head_indices_ptr),
        static_cast<int const *>(worker_packed_qo_indices_ptr),
        static_cast<int const *>(worker_kv_start_ptr),
        static_cast<int const *>(worker_kv_end_ptr),
        static_cast<int const *>(paged_kv_indices_buffer_ptr),
        static_cast<int const *>(paged_kv_indptr_buffer_ptr),
        static_cast<int const *>(paged_kv_last_page_len_buffer_ptr),
        worker_idx
    );
}

void mixed_attn_sm100_kernel(
  torch::Tensor q_tensor, 
  torch::Tensor paged_k, 
  torch::Tensor paged_v, 
  torch::Tensor output_tensor,
  torch::Tensor decode_work_indptr, 
  torch::Tensor prefill_work_indptr, 
  torch::Tensor worker_batch_indices,
  torch::Tensor worker_kv_head_indices,
  torch::Tensor worker_packed_qo_indices,
  torch::Tensor worker_kv_start,
  torch::Tensor worker_kv_end,
  torch::Tensor paged_kv_indices_buffer,
  torch::Tensor paged_kv_indptr_buffer,
  torch::Tensor paged_kv_last_page_len_buffer,
  int worker_idx
) {

  using T = bfloat16;

  void *q_tensor_ptr = q_tensor.data_ptr();
  void *paged_k_ptr = paged_k.data_ptr();
  void *paged_v_ptr = paged_v.data_ptr();
  void *output_tensor_ptr = output_tensor.data_ptr();

  void *decode_work_indptr_ptr = decode_work_indptr.data_ptr();
  void *prefill_work_indptr_ptr = prefill_work_indptr.data_ptr();

  void *worker_batch_indices_ptr = worker_batch_indices.data_ptr();
  void *worker_kv_head_indices_ptr = worker_kv_head_indices.data_ptr();
  void *worker_packed_qo_indices_ptr = worker_packed_qo_indices.data_ptr();
  void *worker_kv_start_ptr = worker_kv_start.data_ptr();
  void *worker_kv_end_ptr = worker_kv_end.data_ptr();
  void *paged_kv_indices_buffer_ptr = paged_kv_indices_buffer.data_ptr();
  void *paged_kv_indptr_buffer_ptr = paged_kv_indptr_buffer.data_ptr();
  void *paged_kv_last_page_len_buffer_ptr = paged_kv_last_page_len_buffer.data_ptr();

  constexpr int NUM_QO_HEADS=32;
  constexpr int NUM_KV_HEADS=8;
  constexpr int HEAD_DIM_QK=128;
  constexpr int HEAD_DIM_V=128;
  constexpr int MAX_NUM_PAGES=1024;
  constexpr int PAGE_SIZE=128;
  constexpr bool CAUSAL=true;
  constexpr int NUM_BUCKETS=128;
  constexpr int MIN_KV_CHUNK_SIZE=512;
  constexpr int P_Q_TILE_SIZE=128;
  constexpr int P_KV_TILE_SIZE=128;
  constexpr int D_Q_TILE_SIZE=16;
  constexpr int D_KV_TILE_SIZE=128;
  constexpr int NUM_QO_PER_KV = NUM_QO_HEADS / NUM_KV_HEADS;

  constexpr int BATCH_SIZE = 20;
  constexpr int QO_LEN = 128;

  static_assert(P_KV_TILE_SIZE == D_KV_TILE_SIZE, "P_KV_TILE_SIZE must equal D_KV_TILE_SIZE");

  assert(decode_work_indptr.size(0) == NUM_BUCKETS + 1);
  assert(prefill_work_indptr.size(0) == NUM_BUCKETS + 1);
  assert(worker_batch_indices.size(0) == NUM_BUCKETS);
  assert(worker_kv_head_indices.size(0) == NUM_BUCKETS);
  assert(worker_packed_qo_indices.size(0) == NUM_BUCKETS);
  assert(worker_kv_start.size(0) == NUM_BUCKETS);
  assert(worker_kv_end.size(0) == NUM_BUCKETS);
  assert(q_tensor.size(0) == BATCH_SIZE * QO_LEN && q_tensor.size(1) == NUM_QO_HEADS && q_tensor.size(2) == HEAD_DIM_QK);
  assert(output_tensor.size(0) == BATCH_SIZE * QO_LEN && output_tensor.size(1) == NUM_QO_HEADS && output_tensor.size(2) == HEAD_DIM_V);
  assert(paged_k.size(0) == MAX_NUM_PAGES && paged_k.size(1) == PAGE_SIZE && paged_k.size(2) == NUM_KV_HEADS && paged_k.size(3) == HEAD_DIM_QK);
  assert(paged_v.size(0) == MAX_NUM_PAGES && paged_v.size(1) == PAGE_SIZE && paged_v.size(2) == NUM_KV_HEADS && paged_v.size(3) == HEAD_DIM_V);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  dim3 cluster_dim(1, 1, 1);
  int smemBytes = 227 * 1024;

  // Q Tensor
  cute::Layout layout_Q = cute::make_layout(
      cute::make_shape(cute::make_shape(cute::Int<NUM_QO_PER_KV>{}, cute::Int<BATCH_SIZE * QO_LEN>{}), cute::Int<HEAD_DIM_QK>{}, cute::Int<NUM_KV_HEADS>{}),
      cute::make_stride(cute::make_stride(cute::Int<HEAD_DIM_QK>{}, cute::Int<HEAD_DIM_QK * NUM_QO_HEADS>{}), cute::Int<1>{}, cute::Int<HEAD_DIM_QK * NUM_QO_PER_KV>{}));
  cute::Tensor mQ = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T *>(q_tensor_ptr)), layout_Q);

  // Output Tensor
  cute::Layout layout_O = cute::make_layout(
      cute::make_shape(cute::make_shape(cute::Int<NUM_QO_PER_KV>{}, cute::Int<BATCH_SIZE * QO_LEN>{}), cute::Int<HEAD_DIM_V>{}, cute::Int<NUM_KV_HEADS>{}),
      cute::make_stride(cute::make_stride(cute::Int<HEAD_DIM_V>{}, cute::Int<HEAD_DIM_V * NUM_QO_HEADS>{}), cute::Int<1>{}, cute::Int<HEAD_DIM_V * NUM_QO_PER_KV>{}));
  cute::Tensor mO = cute::make_tensor(
      cute::make_gmem_ptr(static_cast<T *>(output_tensor_ptr)), layout_O); 


  // TMA desc setup on Host side
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size

  CUtensorMap host_k_desc, host_v_desc;
  CUtensorMap *desc_k_ptr, *desc_v_ptr;

  // TMA_K
  uint64_t k_gmem_shape[4] = {static_cast<uint64_t>(MAX_NUM_PAGES),
                              static_cast<uint64_t>(PAGE_SIZE),
                              static_cast<uint64_t>(NUM_KV_HEADS),
                              static_cast<uint64_t>(HEAD_DIM_QK)};
  uint64_t k_gmem_stride[4] = {1,
                              static_cast<uint64_t>(HEAD_DIM_QK),
                              static_cast<uint64_t>(NUM_KV_HEADS * HEAD_DIM_QK),
                              static_cast<uint64_t>(PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM_QK)};
  uint32_t k_smem_shape[4] = {1u,
                              static_cast<uint32_t>(P_KV_TILE_SIZE),
                              1u,
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t k_smem_repeat_col =
      (HEAD_DIM_QK + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 4>(
      &host_k_desc,
      static_cast<bfloat16 *>(paged_k_ptr),
      k_gmem_shape,
      k_gmem_stride,
      k_smem_shape,
      1,
      k_smem_repeat_col);

  // TMA_V
  uint64_t v_gmem_shape[4] = {static_cast<uint64_t>(MAX_NUM_PAGES),
                              static_cast<uint64_t>(PAGE_SIZE),
                              static_cast<uint64_t>(NUM_KV_HEADS),
                              static_cast<uint64_t>(HEAD_DIM_V)};
  uint64_t v_gmem_stride[4] = {1,
                              static_cast<uint64_t>(HEAD_DIM_V),
                              static_cast<uint64_t>(NUM_KV_HEADS * HEAD_DIM_V),
                              static_cast<uint64_t>(PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM_V)};
  uint32_t v_smem_shape[4] = {1u,
                              static_cast<uint32_t>(P_KV_TILE_SIZE),
                              1u,
                              static_cast<uint32_t>(TMA_CP_ASYNC_SIZE)};
  size_t v_smem_repeat_col =
      (HEAD_DIM_V + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;
  mirage::runtime::fill_tma_desc<bfloat16, B, M, S, 4>(
      &host_v_desc,
      static_cast<bfloat16 *>(paged_v_ptr),
      v_gmem_shape,
      v_gmem_stride,
      v_smem_shape,
      1,
      v_smem_repeat_col);

  cudaMalloc(&desc_k_ptr, sizeof(CUtensorMap));
  cudaMalloc(&desc_v_ptr, sizeof(CUtensorMap));
  cudaMemcpy(
      desc_k_ptr, &host_k_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  cudaMemcpy(
      desc_v_ptr, &host_v_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

  void *tma_desc_k, *tma_desc_v;
  tma_desc_k = desc_k_ptr;
  tma_desc_v = desc_v_ptr;

  // Kernel launch
  auto *kernel_ptr = &mixed_attn_sm100_kernel_wrapper<
      T,
      decltype(mQ),
      decltype(mO),
      NUM_QO_HEADS,
      NUM_KV_HEADS,
      HEAD_DIM_QK,
      HEAD_DIM_V,
      MAX_NUM_PAGES,
      PAGE_SIZE,
      CAUSAL,
      NUM_BUCKETS,
      MIN_KV_CHUNK_SIZE,
      P_Q_TILE_SIZE,
      P_KV_TILE_SIZE,
      D_Q_TILE_SIZE,
      D_KV_TILE_SIZE>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
  cutlass::ClusterLaunchParams params = {
      grid_dim, block_dim, cluster_dim, smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, mQ, tma_desc_k, tma_desc_v, mO, decode_work_indptr_ptr, 
      prefill_work_indptr_ptr, worker_batch_indices_ptr, worker_kv_head_indices_ptr, worker_packed_qo_indices_ptr, worker_kv_start_ptr, worker_kv_end_ptr, 
      paged_kv_indices_buffer_ptr, paged_kv_indptr_buffer_ptr, paged_kv_last_page_len_buffer_ptr, worker_idx);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mixed_attn_sm100",
        &mixed_attn_sm100_kernel,
        "Mixed Attention kernel SM100");
}