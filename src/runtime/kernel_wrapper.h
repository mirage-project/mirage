#include "mirage/runtime/runtime.h"
#include "rms_norm.cuh"


namespace mirage {
namespace runtime {
 template <typename Kernel>
 __device__ void generic_wrapper_kernel(
    TensorDesc* inputs,
    TensorDesc* outputs,
    int4 *tensor_offsets,
    int forloop_range
) {

    auto params = Kernel::pack_parameters(inputs, outputs);
    auto layouts = Kernel::create_layouts(inputs, outputs, tensor_offsets, forloop_range);
    Kernel::execute(params, layouts);
}
}; // namespace runtime
}; // namespace mirage