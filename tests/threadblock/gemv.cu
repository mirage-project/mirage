#include "mirage/kernel/graph.h"
#include "mirage/threadblock/cuda/input_loader.h"
#include "mirage/threadblock/cuda/matmul.h"
#include "mirage/threadblock/cuda/output_saver.h"
#include "mirage/threadblock/graph.h"

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

#include "common.h"

using namespace mirage::threadblock;
using namespace mirage::kernel;

__global__ void launch_gemv_kernel(
    DTensor D_A, STensor A, DTensor D_B, STensor B, DTensor D_C, STensor C) {
  extern __shared__ char smem_buffer[];

  //   copy_global_to_shared(smem_buffer, A, D_A.data_ptr);
  //   copy_global_to_shared(smem_buffer, B, D_B.data_ptr);
  //   copy_global_to_shared(smem_buffer, C, D_C.data_ptr);

  int tb_offset_row = 0;
  int tb_offset_column = 0;
  int global_offset = 0;

  cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};

  // load A & B
  mirage::threadblock::GenericInputLoader loader_A(smem_buffer,
                                                D_A,
                                                A,
                                                threadIdx.x,
                                                blockDim.x,
                                                matrix_offset,
                                                global_offset);

  mirage::threadblock::GenericInputLoader loader_B(smem_buffer,
                                                D_B,
                                                B,
                                                threadIdx.x,
                                                blockDim.x,
                                                matrix_offset,
                                                global_offset);
  __syncthreads();

  GenericGemvExecutor executor(smem_buffer,
                               A,
                               B,
                               C,
                               threadIdx.x,
                               cutlass::canonical_warp_idx_sync(),
                               cutlass::canonical_lane_idx());

  __syncthreads();

  mirage::threadblock::GenericOutputSaver saver(smem_buffer,
                                             D_C,
                                             C,
                                             threadIdx.x,
                                             blockDim.x,
                                             matrix_offset,
                                             global_offset);
  __syncthreads();
}

TEST(threadblock_tests, matmul) {

  mirage::kernel::Graph kgraph;

  // [1, 64] x [64, 64]
  constexpr int m = 1, n = 64, k = 64;
  mirage::threadblock::Graph bgraph = create_single_threadblock_graph(128);

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> A(
      cutlass::MatrixCoord(m, k)),
      C_ours(cutlass::MatrixCoord(m, n)), C_ref(cutlass::MatrixCoord(m, n));
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(
      cutlass::MatrixCoord(k, n));
  random_fill_tensor(A, 'A');
  random_fill_tensor(B, 'B');
  zero_fill_tensor(C_ours);

  // Copy C_ours into C_ref because we do accumulation.
  cutlass::device_memory::copy_device_to_device(
      C_ref.device_data(), C_ours.device_data(), C_ours.capacity());
  C_ref.sync_host();

  mirage::kernel::DTensor D_A = kgraph.new_input(
      {m, k}, mirage::type::DT_FLOAT16, mirage::layout::DmemLayout::DmemRowMajor);
  mirage::kernel::DTensor D_B = kgraph.new_input(
      {k, n}, mirage::type::DT_FLOAT16, mirage::layout::DmemLayout::DmemColumnMajor);
  mirage::kernel::DTensor D_C_ours = kgraph.new_input(
      {m, n}, mirage::type::DT_FLOAT16, mirage::layout::DmemLayout::DmemRowMajor);

  // copy inputs
  cutlass::device_memory::copy_device_to_device(
      static_cast<cutlass::half_t *>(D_A.data_ptr),
      A.device_data(),
      A.capacity());
  cutlass::device_memory::copy_device_to_device(
      static_cast<cutlass::half_t *>(D_B.data_ptr),
      B.device_data(),
      B.capacity());

  mirage::threadblock::STensor A_s =
      bgraph.new_input(D_A, {0, -1, -1}, -1, mirage::layout::SmemRowMajor);
  mirage::threadblock::STensor B_s =
      bgraph.new_input(D_B, {0, -1, -1}, -1, mirage::layout::SmemColumnMajor);
  mirage::threadblock::STensor C_s =
      bgraph.new_input(D_C_ours, {0, -1, -1}, -1, mirage::layout::SmemRowMajor);

  int smem_size = 48 * 1024; // 48 KB
  launch_gemv_kernel<<<bgraph.grid_dim, bgraph.block_dim, smem_size>>>(
      D_A, A_s, D_B, B_s, D_C_ours, C_s);

  cutlass::device_memory::copy_device_to_device(
      C_ours.device_data(),
      static_cast<cutlass::half_t *>(D_C_ours.data_ptr),
      C_ours.capacity());

  cudaDeviceSynchronize();
  A.sync_host();
  B.sync_host();
  C_ours.sync_host();

  cutlass::reference::host::Gemm<cutlass::half_t,
                                 cutlass::layout::RowMajor,
                                 cutlass::half_t,
                                 cutlass::layout::ColumnMajor,
                                 cutlass::half_t,
                                 cutlass::layout::RowMajor,
                                 cutlass::half_t,
                                 float>
      gemm_ref;
  gemm_ref(
      {m, n, k}, 1.0_hf, A.host_ref(), B.host_ref(), 1.0_hf, C_ref.host_ref());
  if (!cutlass::reference::host::TensorRelativelyEquals(
          C_ref.host_view(), C_ours.host_view(), 0.2_hf, 0.1_hf)) {
    char const *filename = "errors_gemv.csv";
    std::cerr << "Error - Our kernel differs from reference. Wrote computed "
                 "and reference results to '"
              << filename << "'" << std::endl;
    std::ofstream file(filename);
    // file << "\n\nA =\n" << A.host_view() << std::endl;
    // file << "\n\nB =\n" << B.host_view() << std::endl;
    file << "\n\nOurs =\n" << C_ours.host_view() << std::endl;
    file << "\n\nReference =\n" << C_ref.host_view() << std::endl;
    file.close();
    ASSERT_TRUE(false && "Our kernel differs from reference");
  }
}
