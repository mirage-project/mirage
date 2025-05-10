#include "cutlass/fast_math.h"
#include "mirage/config.h"
#include "mirage/kernel/chunk.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>
#include <iostream>

namespace mirage {
namespace kernel {

using namespace mirage::type;
using namespace mirage::config;
using namespace mirage::utils;

template <typename DT>
__global__ void execute_chunk(DT *input_ptr,
                              DT *output1_ptr,
                              DT *output2_ptr,
                              int3 input_shape,
                              int3 output_shape,
                              int chunk_size,
                              int chunk_dim,
                              int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    int input_i = i / (input_shape.y * input_shape.z);
    int input_j = (i % (input_shape.y * input_shape.z)) / input_shape.z;
    int input_k = i % input_shape.z;
    if (chunk_dim == 0) {
      if (input_i < output_shape.x) {
        output1_ptr[i] = input_ptr[i];
      } else {
        int i2 =
            ((input_i - output_shape.x) * (output_shape.y * output_shape.z)) +
            (input_j * output_shape.z) + input_k;
        output2_ptr[i2] = input_ptr[i];
      }
    } else if (chunk_dim == 1) {
      if (input_j < output_shape.y) {
        output1_ptr[i] = input_ptr[i];
      } else {
        int i2 = (input_i * (output_shape.y * output_shape.z)) +
                 ((input_j - output_shape.y) * output_shape.z) + input_k;
        output2_ptr[i2] = input_ptr[i];
      }
    } else if (chunk_dim == 2) {
      if (input_k < output_shape.z) {
        output1_ptr[i] = input_ptr[i];
      } else {
        int i2 = (input_i * (output_shape.y * output_shape.z)) +
                 (input_j * output_shape.z) + (input_k - output_shape.z);
        output2_ptr[i2] = input_ptr[i];
      }
    } else { // chunk_dim == 3
      assert(false && "unimplemented");
    }
  }
}

bool KNChunkOp::profile(ProfileResult &result) {
  checkCUDA(cudaSetDevice(0));
  assert(input_tensors[0].data_type == DT_FLOAT16);
  assert(output_tensors[1].data_type == DT_FLOAT16);
  assert(output_tensors[2].data_type == DT_FLOAT16);
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  cutlass::half_t *input_ptr = reinterpret_cast<cutlass::half_t *>(
      dmm->data_base_ptr[0] + input_tensors[0].data_offset);
  cutlass::half_t *output1_ptr = reinterpret_cast<cutlass::half_t *>(
      dmm->data_base_ptr[0] + output_tensors[0].data_offset);
  cutlass::half_t *output2_ptr = reinterpret_cast<cutlass::half_t *>(
      dmm->data_base_ptr[0] + output_tensors[1].data_offset);

  int num_elements = input_tensors[0].num_elements();
  int3 input_shape = {input_tensors[0].dim[0],
                      input_tensors[0].dim[1],
                      input_tensors[0].dim[2]};
  int3 output_shape = {
      chunk_dim == 0 ? input_shape.x / chunk_size : input_shape.x,
      chunk_dim == 1 ? input_shape.y / chunk_size : input_shape.y,
      chunk_dim == 2 ? input_shape.z / chunk_size : input_shape.z};
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < 16; i++) {
    execute_chunk<<<num_blocks, num_threads_per_blk>>>(input_ptr,
                                                       output1_ptr,
                                                       output2_ptr,
                                                       input_shape,
                                                       output_shape,
                                                       chunk_size,
                                                       chunk_dim,
                                                       num_elements);
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / 16;
  printf("Chunk: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

__global__ void compute_chunk_fingerprint(char *dmem_fp_ptr,
                                          mirage::kernel::DTensor input,
                                          mirage::kernel::DTensor output1,
                                          mirage::kernel::DTensor output2,
                                          int3 input_shape,
                                          int3 output_shape,
                                          int chunk_size,
                                          int chunk_dim,
                                          int num_elements) {
  mirage::type::FPType *input_fp_ptr =
      reinterpret_cast<mirage::type::FPType *>(dmem_fp_ptr + input.fp_offset);
  mirage::type::FPType *output1_fp_ptr =
      reinterpret_cast<mirage::type::FPType *>(dmem_fp_ptr + output1.fp_offset);
  mirage::type::FPType *output2_fp_ptr =
      reinterpret_cast<mirage::type::FPType *>(dmem_fp_ptr + output2.fp_offset);

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    int input_i = i / (input_shape.y * input_shape.z);
    int input_j = (i % (input_shape.y * input_shape.z)) / input_shape.z;
    int input_k = i % input_shape.z;
    if (chunk_dim == 0) {
      if (input_i < output_shape.x) {
        output1_fp_ptr[i] = input_fp_ptr[i];
      } else {
        int i2 =
            ((input_i - output_shape.x) * (output_shape.y * output_shape.z)) +
            (input_j * output_shape.z) + input_k;
        output2_fp_ptr[i2] = input_fp_ptr[i];
      }
    } else if (chunk_dim == 1) {
      if (input_j < output_shape.y) {
        output1_fp_ptr[i] = input_fp_ptr[i];
      } else {
        int i2 = (input_i * (output_shape.y * output_shape.z)) +
                 ((input_j - output_shape.y) * output_shape.z) + input_k;
        output2_fp_ptr[i2] = input_fp_ptr[i];
      }
    } else if (chunk_dim == 2) {
      if (input_k < output_shape.z) {
        // printf("0: i=%d, coords=(%d, %d, %d)\n", i, input_i, input_j, input_k);
        output1_fp_ptr[i] = input_fp_ptr[i];
      } else {
        int i2 = (input_i * (output_shape.y * output_shape.z)) +
                 (input_j * output_shape.z) + (input_k - output_shape.z);
        // printf("1: i=%d, i2=%d, coords=(%d, %d, %d)\n", i, i2, input_i, input_j, input_k);
        output2_fp_ptr[i2] = input_fp_ptr[i];
      }
    } else { // chunk_dim == 3
      assert(false && "unimplemented");
    }
  }
}

bool KNChunkOp::fingerprint(void) {
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);

  assert(input_tensors[0].num_dims == output_tensors[0].num_dims);
  assert(input_tensors[0].num_dims == output_tensors[1].num_dims);

  int num_elements = input_tensors[0].num_elements();
  int3 input_shape;
  if (input_tensors[0].num_dims == 1) {
    input_shape.x = 1;
    input_shape.y = 1;
    input_shape.z = input_tensors[0].dim[0];
  } else if (input_tensors[0].num_dims == 2) {
    input_shape.x = 1;
    input_shape.y = input_tensors[0].dim[0];
    input_shape.z = input_tensors[0].dim[1];
  } else { // num_dims = 3
    input_shape.x = input_tensors[0].dim[0];
    input_shape.y = input_tensors[0].dim[1];
    input_shape.z = input_tensors[0].dim[2];
  }

  int adjusted_chunk_dim = chunk_dim + (3 - input_tensors[0].num_dims);
  int3 output_shape;
  output_shape.x = adjusted_chunk_dim == 0 ? input_shape.x / chunk_size : input_shape.x;
  output_shape.y = adjusted_chunk_dim == 1 ? input_shape.y / chunk_size : input_shape.y;
  output_shape.z = adjusted_chunk_dim == 2 ? input_shape.z / chunk_size : input_shape.z;

  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;

  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  // Use GPU 0 for computing fingerprint
  checkCUDA(cudaSetDevice(0));
  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    compute_chunk_fingerprint<<<num_blocks, num_threads_per_blk>>>(
        dmm->fp_base_ptr[gpu_id],
        input_tensors[0],
        output_tensors[0],
        output_tensors[1],
        input_shape,
        output_shape,
        chunk_size,
        adjusted_chunk_dim,
        num_elements);
    checkCUDA(cudaDeviceSynchronize());
  }
  return true;
}

} // namespace kernel
} // namespace mirage