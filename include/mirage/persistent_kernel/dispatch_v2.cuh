// v2 task headers.
//
// runtime_v2.cuh calls codegen-emitted role dispatchers. This file only keeps
// the v2 task implementations visible to generated CUDA.
//
// blackwell_v2/ headers are included below so the codegen's emitted calls
// (e.g., kernel::linear_v2::linear_{loader,launcher,consumer}_task,
// kernel::v2::multitoken_paged_attention_sm100_task_impl) resolve to the v2
// kernel implementations.

#pragma once

#include "mirage/persistent_kernel/runtime_header.h"

// v2 kernels — each file declares its own distinct namespace so there's no
// collision with v1 (kernel::) versions that are still pulled in via
// persistent_kernel.cuh for non-v2 dispatch paths and for shared helpers.
#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_sm100_v2.cuh"  // kernel::linear_v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_sm100_v3.cuh"  // kernel::linear_v3
#include "mirage/persistent_kernel/tasks/blackwell_v2/rmsnorm_v2.cuh"       // kernel::rmsnorm_v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/rotary_embedding_v2.cuh" // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/norm_sm100.cuh"          // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/attention_sm100.cuh"     // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/argmax_sm100.cuh"        // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/silu_mul_v2.cuh"         // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/embedding_v2.cuh"        // kernel::v2

// Legacy wrapper kept for non-role v2 call sites during the transition. The new
// runtime path uses generated _execute_{loader,launcher,consumer,storer}_task_v2.
__device__ __forceinline__ void
_execute_task(mirage::runtime::TaskDesc const *task_desc,
              mirage::runtime::RuntimeConfig const &config);

namespace mirage {
namespace runtime_v2 {

using namespace mirage::runtime;

__device__ __forceinline__ void
_execute_task_v2(TaskDesc const *task_desc,
                 RuntimeConfig const &config) {
    ::_execute_task(task_desc, config);
}

} // namespace runtime_v2
} // namespace mirage
