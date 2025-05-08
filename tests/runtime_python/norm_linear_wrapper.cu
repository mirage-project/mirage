#include <torch/extension.h>
#include <cuda_runtime.h>
#include "norm_linear.cuh"

using mirage::runtime::norm_linear_kernel;

template <typename T>
__global__ void norm_linear_kernel_wrapper(void const * input_ptr, void const * weight_ptr, void * output_ptr, int batch_size, int hidden_size) {
   norm_linear_kernel<T, 16, 64, 128>(input_ptr, weight_ptr, output_ptr);
}

void launch_norm_linear(torch::Tensor input, torch::Tensor weight, torch::Tensor output) {
    
    void const * input_ptr = input.data_ptr();
    void const * weight_ptr = weight.data_ptr();
    void * output_ptr = output.data_ptr();
    norm_linear_kernel_wrapper<nv_bfloat16><<<(1, 1, 1), (128, 1, 1)>>>(
        input_ptr, weight_ptr, output_ptr,
        input.size(0), input.size(1)
    );
}

// pybind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("norm_linear", &launch_norm_linear, "RMSNorm Linear kernel");
}