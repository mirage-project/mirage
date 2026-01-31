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
#include "tasks/common/common_header.cuh"
#include "tasks/hopper/allreduce.cuh"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>
#include "mpi.h"
#include <torch/extension.h>
#include <vector>
#include <cuda_bf16.h>

#include <nvshmem.h>
#include <nvshmemx.h>

using bfloat16 = type::bfloat16_t;

#ifndef NVSHMEM_CHECK
#define NVSHMEM_CHECK(stmt)                                                    \
    do {                                                                       \
        int result = (stmt);                                                   \
        if (NVSHMEMX_SUCCESS != result) {                                      \
            fprintf(stderr, "[%s:%d] NVSHMEM failed with error %d\n",          \
                    __FILE__, __LINE__, result);                               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#endif    

static void* nvshmem_teams_ptr_global = nullptr;
static void* input_ptr_global = nullptr;
static void* output_ptr_global = nullptr;
// One CTA/team per hidden-dimension tile (4096 / 64 = 64 tiles).
constexpr int NUM_TEAMS = 64;
constexpr size_t MALLOC_SIZE = 1024 * 1024 * 1024; // 1GB

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__global__ void allreduce_kernel_wrapper(void *input_ptr,
                                         void *output_ptr,
                                         void *nvshmem_teams,
                                         int task_offset) {
  // One CTA per tile: map blockIdx.x to tile/team
  task_offset = blockIdx.x;
  // Offset to this tile's slice along hidden dimension
  input_ptr = reinterpret_cast<void*>(
      reinterpret_cast<char*>(input_ptr) + blockIdx.x * OUTPUT_SIZE * sizeof(T));
  output_ptr = reinterpret_cast<void*>(
      reinterpret_cast<char*>(output_ptr) + blockIdx.x * OUTPUT_SIZE * sizeof(T));
  kernel::nvshmem_tile_allreduce<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_STRIDE>(
      input_ptr, output_ptr, nvshmem_teams, task_offset);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
void launch_allreduce(void *input_ptr,
                      void *output_ptr,
                      void *nvshmem_teams,
                      int task_offset = 0) {
  dim3 grid_dim(NUM_TEAMS, 1, 1);
  dim3 block_dim(256, 1, 1);

  allreduce_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_STRIDE>
      <<<grid_dim, block_dim>>>(
          input_ptr, output_ptr, nvshmem_teams, task_offset);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

void init_nvshmem() {
  // Choose device based on local MPI rank before initializing NVSHMEM so the
  // symmetric heap is created on the correct GPU.
  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  int local_device = (device_count > 0) ? (world_rank % device_count) : 0;

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);

  cudaSetDevice(mype_node);

  input_ptr_global = nvshmem_align(4096, MALLOC_SIZE);
  output_ptr_global = nvshmem_align(4096, MALLOC_SIZE);

  cudaMemset(output_ptr_global, -1, MALLOC_SIZE);

  std::vector<nvshmem_team_t> teams_host(NUM_TEAMS, 0);
  for (int i = 0; i < NUM_TEAMS; i++) {
    NVSHMEM_CHECK(nvshmem_team_split_strided(
        NVSHMEM_TEAM_WORLD, 0, 1, npes, nullptr, 0, &teams_host[i]));
    if (mype == 0)
    printf("Team %d : %d\n", i, teams_host[i]);
  }
  cudaMalloc(&nvshmem_teams_ptr_global, NUM_TEAMS * sizeof(nvshmem_team_t));
  cudaMemcpy(nvshmem_teams_ptr_global,
            teams_host.data(),
            NUM_TEAMS * sizeof(nvshmem_team_t),
            cudaMemcpyHostToDevice);
}

void allreduce_kernel(torch::Tensor input,
                      torch::Tensor output,
                      int task_offset) {
  void *input_ptr = input.data_ptr();
  void *output_ptr = output.data_ptr();
  void *nvshmem_teams_ptr = nvshmem_teams_ptr_global;

  if (input_ptr == nullptr || output_ptr == nullptr ||
      nvshmem_teams_ptr == nullptr) {
    printf("NVSHMEM allreduce kernel not initialized properly.\n");
    assert(0);
    return;
  }

  // Copy data from tensor to nvshmem allocated memory
  size_t num_bytes = input.numel() * sizeof(bfloat16);
  cudaMemcpy(input_ptr_global, input_ptr, num_bytes, cudaMemcpyDeviceToDevice);

  // Reduce hidden dimension using 64-element tiles (one CTA per tile).
  launch_allreduce<__nv_bfloat16, 8, 64, 4096>(input_ptr_global, output_ptr_global, nvshmem_teams_ptr);

  cudaMemcpy(output_ptr, output_ptr_global, num_bytes, cudaMemcpyDeviceToDevice);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_nvshmem", &init_nvshmem, "Initialize NVSHMEM");
  m.def("allreduce", &allreduce_kernel, "NVSHMEM tile allreduce kernel");
}
