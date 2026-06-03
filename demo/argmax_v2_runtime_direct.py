import os
import tempfile

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#define MODE_OFFLINE
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "mirage/persistent_kernel/runtime_v2.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/argmax_sm100.cuh"

using namespace mirage::runtime;

__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig const &config) {
  if (config.step[0] == 0) {
    config.qo_indptr_buffer[0] = 0;
    config.qo_indptr_buffer[1] = 1;
    config.step[0] = config.max_seq_length - 1;
  }
  return true;
}

namespace mirage {
namespace runtime_v2 {

__device__ __forceinline__ void
_execute_loader_task_v2(TaskDesc const *task_desc,
                        RuntimeConfig const &config,
                        RuntimeSMEM *runtime_smem,
                        int instruction_index) {
  switch (task_desc->task_type) {
  case TASK_ARGMAX_PARTIAL_SM100_V2:
  case TASK_ARGMAX_REDUCE_SM100_V2:
    ::kernel::v2::ArgmaxTask::loader::run(
        task_desc, config, runtime_smem, instruction_index);
    break;
  default:
    break;
  }
}

__device__ __forceinline__ void
_execute_launcher_task_v2(TaskDesc const *task_desc,
                          RuntimeConfig const &config,
                          RuntimeSMEM *runtime_smem,
                          int instruction_index) {
  switch (task_desc->task_type) {
  case TASK_ARGMAX_PARTIAL_SM100_V2:
  case TASK_ARGMAX_REDUCE_SM100_V2:
    ::kernel::v2::ArgmaxTask::launcher::run(
        task_desc, config, runtime_smem, instruction_index);
    break;
  default:
    break;
  }
}

__device__ __forceinline__ void
_execute_consumer_task_v2(TaskDesc const *task_desc,
                          RuntimeConfig const &runtime_config,
                          RuntimeSMEM *runtime_smem,
                          int) {
  switch (task_desc->task_type) {
  case TASK_ARGMAX_PARTIAL_SM100_V2:
    if (task_desc->variant_id == 0) {
      ::kernel::v2::argmax_partial_sm100_kernel<::kernel::bfloat16, 1, 128, 1>(
          task_desc->input_ptrs[0],
          task_desc->output_ptrs[0],
          task_desc->output_ptrs[1],
          task_desc,
          runtime_config.qo_indptr_buffer[1],
          runtime_smem);
    }
    break;
  case TASK_ARGMAX_REDUCE_SM100_V2:
    if (task_desc->variant_id == 0) {
      ::kernel::v2::argmax_reduce_sm100_kernel<::kernel::bfloat16, 1, 128, 1>(
          task_desc->input_ptrs[0],
          task_desc->input_ptrs[1],
          task_desc->output_ptrs[0],
          task_desc,
          runtime_config.qo_indptr_buffer[1],
          runtime_smem);
    }
    break;
  default:
    break;
  }
}

__device__ __forceinline__ void
_execute_storer_task_v2(TaskDesc const *task_desc,
                        RuntimeConfig const &config,
                        RuntimeSMEM *runtime_smem,
                        int instruction_index) {
  switch (task_desc->task_type) {
  case TASK_ARGMAX_PARTIAL_SM100_V2:
  case TASK_ARGMAX_REDUCE_SM100_V2:
    ::kernel::v2::ArgmaxTask::storer::run(
        task_desc, config, runtime_smem, instruction_index);
    break;
  default:
    break;
  }
}

} // namespace runtime_v2
} // namespace mirage

static void check_cuda(cudaError_t err, char const *where) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " +
                             cudaGetErrorString(err));
  }
}

torch::Tensor run_argmax_v2_direct(torch::Tensor logits) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  TORCH_CHECK(logits.scalar_type() == torch::kBFloat16, "logits must be bf16");
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) == 1 && logits.size(1) == 128,
              "expected logits shape [1, 128]");

  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(logits.device());
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());
  auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(logits.device());

  torch::Tensor output = torch::full({1, 1}, -1, opts_i64);
  torch::Tensor part_val = torch::empty({1, 1}, opts_bf16);
  torch::Tensor part_idx = torch::empty({1, 1}, opts_i64);
  torch::Tensor step = torch::zeros({1}, opts_i32);
  torch::Tensor qo = torch::zeros({2}, opts_i32);

  TaskDesc h_tasks[2];
  h_tasks[0].task_type = TASK_ARGMAX_PARTIAL_SM100_V2;
  h_tasks[0].variant_id = 0;
  h_tasks[0].dependent_event = EVENT_INVALID_ID;
  h_tasks[0].trigger_event = EVENT_INVALID_ID;
  h_tasks[0].input_ptrs[0] = logits.data_ptr();
  h_tasks[0].output_ptrs[0] = part_val.data_ptr();
  h_tasks[0].output_ptrs[1] = part_idx.data_ptr();
  h_tasks[0].num_smem_regions = 2;
  h_tasks[0].smem_regions[0] = {0, 1, 0};
  h_tasks[0].smem_regions[1] = {0, 1, 256};

  h_tasks[1].task_type = TASK_ARGMAX_REDUCE_SM100_V2;
  h_tasks[1].variant_id = 0;
  h_tasks[1].dependent_event = EVENT_INVALID_ID;
  h_tasks[1].trigger_event = EVENT_INVALID_ID;
  h_tasks[1].input_ptrs[0] = part_val.data_ptr();
  h_tasks[1].input_ptrs[1] = part_idx.data_ptr();
  h_tasks[1].output_ptrs[0] = output.data_ptr();
  h_tasks[1].num_smem_regions = 2;
  h_tasks[1].smem_regions[0] = {1, 1, 0};
  h_tasks[1].smem_regions[1] = {1, 1, 256};

  TaskDesc *d_tasks = nullptr;
  size_t *d_offsets = nullptr;
  size_t *d_positions = nullptr;
  unsigned long long *d_sync = nullptr;
  unsigned long long *d_go = nullptr;
  check_cuda(cudaMalloc(&d_tasks, sizeof(h_tasks)), "cudaMalloc tasks");
  check_cuda(cudaMemcpy(d_tasks, h_tasks, sizeof(h_tasks), cudaMemcpyHostToDevice),
             "cudaMemcpy tasks");
  size_t h_offsets[2] = {0, 2};
  size_t h_positions[2] = {0, 1};
  check_cuda(cudaMalloc(&d_offsets, sizeof(h_offsets)), "cudaMalloc offsets");
  check_cuda(cudaMalloc(&d_positions, sizeof(h_positions)), "cudaMalloc positions");
  check_cuda(cudaMemcpy(d_offsets, h_offsets, sizeof(h_offsets), cudaMemcpyHostToDevice),
             "cudaMemcpy offsets");
  check_cuda(cudaMemcpy(d_positions, h_positions, sizeof(h_positions), cudaMemcpyHostToDevice),
             "cudaMemcpy positions");
  check_cuda(cudaMalloc(&d_sync, sizeof(unsigned long long)), "cudaMalloc sync");
  check_cuda(cudaMalloc(&d_go, sizeof(unsigned long long)), "cudaMalloc go");
  check_cuda(cudaMemset(d_sync, 0, sizeof(unsigned long long)), "cudaMemset sync");
  check_cuda(cudaMemset(d_go, 0, sizeof(unsigned long long)), "cudaMemset go");

  RuntimeConfig config{};
  config.num_workers = 1;
  config.num_events = 0;
  config.all_tasks = d_tasks;
  config.step = static_cast<int *>(step.data_ptr());
  config.qo_indptr_buffer = static_cast<int *>(qo.data_ptr());
  config.max_seq_length = 2;
  config.v2_per_sm_task_offsets = d_offsets;
  config.v2_per_sm_task_positions = d_positions;
  config.v2_iter_sync_counter = d_sync;
  config.v2_iter_go_counter = d_go;
  config.v2_max_iters = 1;
  config.v2_enabled = true;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  mirage::runtime_v2::launch_worker_v2(config, 1, stream);
  check_cuda(cudaGetLastError(), "launch_worker_v2");
  check_cuda(cudaStreamSynchronize(stream), "sync");

  cudaFree(d_tasks);
  cudaFree(d_offsets);
  cudaFree(d_positions);
  cudaFree(d_sync);
  cudaFree(d_go);
  return output;
}

"""


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(tempfile.gettempdir(), "mirage_argmax_v2_direct")
    os.makedirs(build_dir, exist_ok=True)
    mod = load_inline(
        name="mirage_argmax_v2_direct",
        cpp_sources="torch::Tensor run_argmax_v2_direct(torch::Tensor logits);",
        cuda_sources=CUDA_SRC,
        functions=["run_argmax_v2_direct"],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DMODE_OFFLINE",
            "-DMIRAGE_BACKEND_USE_CUDA",
            "-DMPK_MAX_NUM_BATCHED_REQUESTS=1",
            "-DMPK_MAX_NUM_BATCHED_TOKENS=1",
            "-DMPK_MAX_NUM_PAGES=4",
            "-DMPK_PAGE_SIZE=16",
            "-DMPK_MAX_SEQ_LENGTH=2",
            "-DUSE_RUNTIME_V2",
        ],
        extra_include_paths=[
            os.path.join(root, "include"),
            os.path.join(root, "deps", "cutlass", "include"),
            os.path.join(root, "deps", "json", "include"),
        ],
        build_directory=build_dir,
        verbose=False,
    )
    logits = torch.randn((1, 128), dtype=torch.bfloat16, device="cuda")
    expected = int(torch.argmax(logits[0]).item())
    output = mod.run_argmax_v2_direct(logits)
    got = int(output[0, 0].item())
    print(f"expected={expected} got={got}")
    if got != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
