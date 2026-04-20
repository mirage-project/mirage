// v2 task dispatch.
// Called by compute warps (W0-7) from runtime_v2.cuh compute_warp_loop.
//
// All Qwen3 tasks now use TASK_X_V2 enums and are dispatched through the
// codegen-emitted ::_execute_task switch (auto-generated per model). This
// file just forwards to that switch.
//
// blackwell_v2/ headers are included below so the codegen's emitted calls
// (e.g., kernel::linear_v2::linear_task, kernel::v2::multitoken_paged_attention_sm100_task_impl)
// resolve to the v2 kernel implementations.

#pragma once

#include "mirage/persistent_kernel/runtime_header.h"

// v2 kernels — each file declares its own distinct namespace so there's no
// collision with v1 (kernel::) versions that are still pulled in via
// persistent_kernel.cuh for non-v2 dispatch paths and for shared helpers.
#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_sm100_v2.cuh"  // kernel::linear_v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/rmsnorm_v2.cuh"       // kernel::rmsnorm_v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/rotary_embedding_v2.cuh" // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/norm_sm100.cuh"          // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/attention_sm100.cuh"     // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/argmax_sm100.cuh"        // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/silu_mul_v2.cuh"         // kernel::v2
#include "mirage/persistent_kernel/tasks/blackwell_v2/embedding_v2.cuh"        // kernel::v2

// Forward decl: v1 codegen emits ::_execute_task in the generated .cu at global
// scope. The generated switch contains only _V2 enum cases when Python calls
// _v2 layer variants (which happens when use_v2_runtime=True).
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
