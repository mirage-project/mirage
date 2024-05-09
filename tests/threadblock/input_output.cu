#include "mirage/kernel/graph.h"
#include "mirage/threadblock/cuda/input_loader.h"
#include "mirage/threadblock/cuda/output_saver.h"
#include "mirage/threadblock/graph.h"

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

#include "common.h"

using namespace mirage::threadblock;
using namespace mirage::kernel;

__global__ void
    launch_input_output_kernel(DTensor D_In, DTensor D_Out, STensor S_tensor) {
  extern __shared__ char smem_buffer[];
  // save to shared memory and copy back

  int tb_offset_row = 0;
  int tb_offset_column = 0;
  int global_offset = blockIdx.x * (64 * 64);

  cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
  mirage::threadblock::GenericInputLoader loader(smem_buffer,
                                              D_In,
                                              S_tensor,
                                              threadIdx.x,
                                              blockDim.x,
                                              matrix_offset,
                                              global_offset);
  __syncthreads();
  mirage::threadblock::GenericOutputSaver saver(smem_buffer,
                                             D_Out,
                                             S_tensor,
                                             threadIdx.x,
                                             blockDim.x,
                                             matrix_offset,
                                             global_offset);
  __syncthreads();
}

TEST(threadblock_tests, input_output) {
  mirage::kernel::Graph kgraph;

  // single thread block test
  mirage::threadblock::Graph bgraph({16, 1, 1}, {128, 1, 1}, 1);
  mirage::kernel::DTensor Input =
      kgraph.new_input({16, 64, 64},
                       mirage::type::DT_FLOAT16,
                       mirage::layout::DmemLayout::DmemRowMajor);
  mirage::kernel::DTensor Output =
      kgraph.new_input({16, 64, 64},
                       mirage::type::DT_FLOAT16,
                       mirage::layout::DmemLayout::DmemRowMajor);
  mirage::kernel::DTensor Output_Ref =
      kgraph.new_input({16, 64, 64},
                       mirage::type::DT_FLOAT16,
                       mirage::layout::DmemLayout::DmemRowMajor);

  int const num_threads_per_blk = 1024;
  int num_blocks =
      (Input.num_elements() + num_threads_per_blk - 1) / num_threads_per_blk;

  random_fill_device_tensor<cutlass::half_t>
      <<<num_blocks, num_threads_per_blk>>>(Input, Input.num_elements());
  cudaMemcpy(Output_Ref.data_ptr,
             Input.data_ptr,
             Input.num_elements() * sizeof(cutlass::half_t),
             cudaMemcpyDeviceToDevice);

  mirage::threadblock::STensor Input_S =
      bgraph.new_input(Input, {0, -1, -1}, -1, mirage::layout::SmemRowMajor);

  int smem_size = 48 * 1024; // 48 KB
  launch_input_output_kernel<<<bgraph.grid_dim, bgraph.block_dim, smem_size>>>(
      Input, Output, Input_S);

  cudaDeviceSynchronize();

  // check Output and Output_Ref
  int h_notEqual = 0;
  int *d_notEqual;

  cudaMalloc(&d_notEqual, sizeof(int));
  cudaMemcpy(d_notEqual, &h_notEqual, sizeof(int), cudaMemcpyHostToDevice);

  checkTensorsEqual<cutlass::half_t><<<num_blocks, num_threads_per_blk>>>(
      Output.data_ptr, Output_Ref.data_ptr, d_notEqual, Output.num_elements());

  cudaMemcpy(&h_notEqual, d_notEqual, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "Unequal number of elements: " << h_notEqual << std::endl;
  ASSERT_TRUE(h_notEqual == 0);
  cudaFree(d_notEqual);
}
