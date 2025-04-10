#include "mirage/runtime/runtime.h"
#include "mirage/runtime/rms_norm.cuh"

 template <typename Kernel>
 __global__ void generic_wrapper_kernel(
    TensorDesc* inputs,  // array of input descriptors
    int num_inputs,
    TensorDesc* outputs, // array of output descriptors
    int num_outputs
) {

    auto params = Kernel::pack_parameters(inputs, outputs);
    auto layouts = Kernel::create_layouts(inputs, outputs);
    Kernel::execute(params, layouts);
}