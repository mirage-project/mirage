#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "mirage/transpiler/runtime/runtime.h"
#include "mirage/runtime/runtime.h"
#include "layout_infer.h"

using namespace cute;

namespace mirage {
namespace runtime {


struct EmbeddingKernel {

  static constexpr int input_nums = 2;
  static constexpr int output_nums = 1;

  static __device__ __forceinline__ void execute(TensorDesc* inputs, TensorDesc* outputs, int4 *tensor_offsets, int forloop_range);
};



template <typename Input0Layout, typename Input0LayoutDevice,
          typename Input1Layout, typename Input1LayoutDevice,
          typename Output0Layout, typename Output0LayoutDevice>
__device__ __forceinline__ void embedding_kernel_impl(bfloat16_t* __restrict__ output,
    const uint16_t* __restrict__ input_ids,
    const bfloat16_t* __restrict__ embedding);

#define EMBEDDING_KERNEL(BATCH_SIZE, OUT_DIM) \
  embedding_kernel_impl<BATCH_SIZE, OUT_DIM>(dtensor10000005_ptr, dtensor10000003_ptr, dtensor10000004_ptr)

__device__ __forceinline__ void EmbeddingKernel::execute(TensorDesc* inputs, TensorDesc* outputs, int4 *tensor_offsets, int forloop_range) 
{

  bfloat16_t* __restrict__ output = static_cast<uint16_t*>(outputs[0].base_ptr);
  uint16_t const* __restrict__ input_ids = static_cast<const uint16_t*>(inputs[0].base_ptr);
  bfloat16_t const* __restrict__ embedding= static_cast<bfloat16_t*>(inputs[1].base_ptr);

  if(dim_out[0]==1 && dim_out[1]==3584)
  EMBEDDING_KERNEL(1, 3584);
  }else{
    assert(false && "unsupported layout");
  }
  

template <int BATCH_SIZE, int OUT_DIM>
__device__ __forceinline__ void embedding_kernel_impl(
    bfloat16_t* __restrict__ output,
    const uint16_t* __restrict__ input_ids,
    const bfloat16_t* __restrict__ embedding) 
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < BATCH_SIZE * OUT_DIM; i += blockDim.x * gridDim.x) {
        int idx = i / OUT_DIM;
        int off = i % OUT_DIM;
        uint16_t wordIdx = reinterpret_cast<const uint16_t*>(input_ids)[idx];
        output[i] = embedding[wordIdx * OUT_DIM + off];
    }
}
  
  


} // namespace runtime
} // namespace mirage