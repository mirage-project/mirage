#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "mirage/transpiler/runtime/runtime.h"
#include "mirage/runtime/runtime.h"
#include "layout_infer.h"

using namespace cute;

namespace mirage {
namespace runtime {


struct RmsNormKernel {

  static constexpr int input_nums = 2;
  static constexpr int output_nums = 1;

  // struct Params {
  //   const bfloat16_t* input0;
  //   const bfloat16_t* input1;
  //   bfloat16_t* output0;
  
  //   int4 offset_in0;
  //   int4 offset_in1;
  //   int4 offset_out0;
  
  //   int forloop_range;
  
  //   int input0_dim[2];
  //   int input0_stride[2];
  
  //   int input1_dim[2];
  //   int input1_stride[2];
  
  //   int output0_dim[2];
  //   int output0_stride[2];
  // };

  // static __device__ __forceinline__ auto pack_parameters(TensorDesc* inputs, TensorDesc* outputs, int4 *tensor_offsets, int forloop_range);

  static __device__ __forceinline__ void execute(TaskDesc &task_desc,  int4 *tensor_offsets, int forloop_range);
};

// auto RmsNormKernel::pack_parameters(TensorDesc* inputs, TensorDesc* outputs, int4 *tensor_offsets, int forloop_range) {

//   Params params;

//   params.input0 = static_cast<const bfloat16_t*>(inputs[0].base_ptr);
//   params.input1 = static_cast<const bfloat16_t*>(inputs[1].base_ptr);
//   params.output0 = static_cast<bfloat16_t*>(outputs[0].base_ptr);

//   params.offset_in0 = *tensor_offsets;
//   params.offset_in1 = *(tensor_offsets + 1);
//   params.offset_out0 = *(tensor_offsets + 2);
//   params.forloop_range = forloop_range;

//   return params;
// }


template <typename Input0Layout, typename Input0LayoutDevice,
          typename Input1Layout, typename Input1LayoutDevice,
          typename Output0Layout, typename Output0LayoutDevice>
__device__ __forceinline__ void rms_norm_kernel_impl(bfloat16_t* __restrict__ dtensor10000005_ptr, bfloat16_t const* __restrict__ dtensor10000003_ptr, bfloat16_t const* __restrict__ dtensor10000004_ptr, int4 offset_in0,
  int4 offset_in1,
  int4 offset_out0,
  int forloop_range);

#define RMSNORM_KERNEL(I0, I0D, I1, I1D, O0, O0D) \
  rms_norm_kernel_impl<I0, I0D, I1, I1D, O0, O0D>(dtensor10000005_ptr, dtensor10000003_ptr, dtensor10000004_ptr, tensor_offsets[0], tensor_offsets[1], tensor_offsets[2], forloop_range)

// template <typename Layouts>
__device__ __forceinline__ void RmsNormKernel::execute(TaskDesc &task_desc, int4 *tensor_offsets, int forloop_range) 
{
  // auto& p = *static_cast<Params*>(params);

   TensorDesc* inputs = task_desc.inputs;
   TensorDesc* outputs = task_desc.outputs;

  // Convenience aliases for readability
  const int *dim0 = inputs[0].dim;
  const int *stride0 = inputs[0].stride;

  const int *dim1 = inputs[1].dim;
  const int *stride1 = inputs[1].stride;

  const int *dim_out = outputs[0].dim;
  const int *stride_out = outputs[0].stride;

  const int *dtensor1_stride = inputs[1].dtensor_stride;


  bfloat16_t* __restrict__ dtensor10000005_ptr = static_cast<bfloat16_t*>(outputs[0].base_ptr);
  bfloat16_t const* __restrict__ dtensor10000003_ptr = static_cast<const bfloat16_t*>(inputs[0].base_ptr);
  bfloat16_t const* __restrict__ dtensor10000004_ptr= static_cast<bfloat16_t*>(inputs[1].base_ptr);
  

  if(dim0[0]==1 && dim0[1]==64 && dim1[0]==64 && dim1[1]==64 && dtensor1_stride[0] == 4608){
  using Input0Layout        = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
  using Input0LayoutDevice  = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<4096>>>;
  using Input1Layout        = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Input1LayoutDevice  = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<4608>>>;
  using Output0Layout       = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
  using Output0LayoutDevice = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
  RMSNORM_KERNEL(Input0Layout, Input0LayoutDevice, Input1Layout, Input1LayoutDevice, Output0Layout, Output0LayoutDevice);
  }else if(dim0[0]==1 && dim0[1]==64 && dim1[0]==64 && dim1[1]==64 && dtensor1_stride[0] == 7168){
    using Input0Layout        = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
  using Input0LayoutDevice  = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<4096>>>;
  using Input1Layout        = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Input1LayoutDevice = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<7168>>>;
  using Output0Layout       = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
  using Output0LayoutDevice = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
  RMSNORM_KERNEL(Input0Layout, Input0LayoutDevice, Input1Layout, Input1LayoutDevice, Output0Layout, Output0LayoutDevice);
  }
  else{
    assert(false && "unsupported layout");

  }
  
}

template <typename Input0Layout, typename Input0LayoutDevice, typename Input1Layout, typename Input1LayoutDevice, typename Output0Layout,  typename Output0LayoutDevice>
__device__ __forceinline__ void rms_norm_kernel_impl(bfloat16_t* __restrict__ dtensor10000005_ptr, bfloat16_t const* __restrict__ dtensor10000003_ptr, bfloat16_t const* __restrict__ dtensor10000004_ptr, int4 offset_in0,
  int4 offset_in1,
  int4 offset_out0,
  int forloop_range) {  
    int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  bfloat16_t *stensor20000031_ptr = (bfloat16_t*)(buf + 128);
  bfloat16_t *stensor20000029_ptr = (bfloat16_t*)(buf + 384);
  bfloat16_t *stensor20000030_ptr = (bfloat16_t*)(buf + 256);
  bfloat16_t *stensor30000023_ptr = (bfloat16_t*)(buf + 8704);
  bfloat16_t *stensor20000023_ptr = (bfloat16_t*)(buf + 512);
  bfloat16_t *stensor30000022_ptr = (bfloat16_t*)(buf + 384);
  bfloat16_t *stensor20000022_ptr = (bfloat16_t*)(buf + 256);
  bfloat16_t *stensor20000027_ptr = (bfloat16_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000003 -> stensor 20000022
  const bfloat16_t *dtensor10000003_tile_ptr = dtensor10000003_ptr ;
  using DTensor10000003TileLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<3584>>>;
  using STensor20000022InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>, DTensor10000003TileLayout, NUM_THREADS>;
  bfloat16_t *stensor20000022_async_copy_buf = stensor30000022_ptr;
  // Copy for G->S: dtensor 10000004 -> stensor 20000023
  const bfloat16_t *dtensor10000004_tile_ptr = dtensor10000004_ptr  + blockIdx.x*64*1;
  using DTensor10000004TileLayout = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<7168>>>;
  using STensor20000023InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{})), DTensor10000004TileLayout, NUM_THREADS>;
  bfloat16_t *stensor20000023_async_copy_buf = stensor30000023_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000031 -> dtensor 10000005
  bfloat16_t *dtensor10000005_tile_ptr = dtensor10000005_ptr  + blockIdx.x*64*1;
  using DTensor10000005TileLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
  using STensor20000031OutputAtom = tb::OutputChunkedSyncCopy<bfloat16_t, DTensor10000005TileLayout, Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<bfloat16_t, 64, NUM_THREADS>::run(stensor20000027_ptr, thread_idx);
  
  using Matmul20000030LayoutA = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
  using Matmul20000030LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000030LayoutC = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
  using Matmul20000030LayoutAAligned = Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>;
  using Matmul20000030LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000030Kernel = tb::Matmul<bfloat16_t, SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000030LayoutA, Matmul20000030LayoutB, Matmul20000030LayoutC, Matmul20000030LayoutAAligned, Matmul20000030LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000030_accum = Matmul20000030Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000023InputAtom::run(stensor20000023_async_copy_buf, dtensor10000004_tile_ptr, thread_idx);
    STensor20000022InputAtom::run(stensor20000022_async_copy_buf, dtensor10000003_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 56; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 56) {
        STensor20000023InputAtom::run(stensor20000023_ptr, dtensor10000004_tile_ptr + 458752*(for_idx+1), thread_idx);
        STensor20000022InputAtom::run(stensor20000022_ptr, dtensor10000003_tile_ptr + 64*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000023_ptr, stensor20000023_async_copy_buf);
      SWAP(stensor20000022_ptr, stensor20000022_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      Matmul20000030Kernel::run(matmul_20000030_accum, stensor20000022_ptr, stensor20000023_ptr, (char*)(buf+0), thread_idx);
    }
    {
      // OP type: tb_square_op
      using InLayout = Layout<Shape<Int<1>, Int<64>>, Stride<Int<64>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<1>, Int<64>>, Stride<Int<1>, Int<1>>>;
      using Kernel = tb::ElementUnaryKernel<bfloat16_t, tb::ElementUnaryOpType::SQUARE, OutLayout, InLayout, NUM_THREADS, tb::EpilogueMulScalar<bfloat16_t, tb::EpilogueStoreAccum<bfloat16_t>>>;
      const float scalars[] = {0.000279f, 0.0f};
      Kernel::run(stensor20000027_ptr, stensor20000022_ptr, thread_idx, 0.000000, scalars);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000030Kernel::write_back_mma_rC(stensor20000030_ptr, matmul_20000030_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_1_op
    using InLayout = Layout<Shape<Int<1>, Int<64>>, Stride<Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<1>>, Stride<Int<1>, Int<1>>>;
    using Kernel = tb::ReductionKernel<bfloat16_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueSqrt<bfloat16_t, tb::EpilogueStore<bfloat16_t>>>;
    const float scalars[] = {0.000000f, 0.0f};
    Kernel::run(stensor20000029_ptr, stensor20000027_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<1>, Int<64>>, Stride<Int<64>, Int<1>>>;
    using In1Layout = Layout<Shape<Int<1>, Int<1>>, Stride<Int<1>, Int<1>>>;
    using OutLayout = Layout<Shape<Int<1>, Int<64>>, Stride<Int<1>, Int<1>>>;
    using Kernel = tb::ElementBinaryKernel<bfloat16_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<bfloat16_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000031_ptr, stensor20000030_ptr, stensor20000029_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000031OutputAtom::run(dtensor10000005_tile_ptr, stensor20000031_ptr, thread_idx);
  }
}

} // namespace runtime
} // namespace mirage