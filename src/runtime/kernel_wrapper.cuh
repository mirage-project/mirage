#include "mirage/runtime/runtime.h"
#include "mirage/runtime/rms_norm.cuh"

 template <typename Kernel>
 __global__ void generic_wrapper_kernel(
    TensorDesc* inputs,
    TensorDesc* outputs,
    int4 *tensor_offsets,
    int forloop_range
) {

    auto params = Kernel::pack_parameters(inputs, outputs);
    auto layouts = Kernel::create_layouts(inputs, outputs, tensor_offsets, forloop_range);
    Kernel::execute(params, layouts);
}