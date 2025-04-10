#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "mirage/transpiler/runtime/runtime.h"
#include "layout_infer.h"
using namespace cute;


struct RmsNormKernel {

  static constexpr int input_nums = 2;
  static constexpr int output_nums = 1;

  static void* pack_parameters(TensorDesc* inputs, TensorDesc* outputs);
  static auto create_layouts(TensorDesc* inputs, TensorDesc* outputs);

  template <typename Layouts>
  static __device__ void execute(void* params, Layouts layouts, 
                               char* shared_buf, int tid);
};

void* RmsNormKernel::pack_parameters(TensorDesc* inputs, TensorDesc* outputs) {
  static struct {
      half_t* input0;
      const half_t* input1;
      const half_t* output0;
  } params;
  
  params = {
      static_cast<half_t*>(inputs[0].base_ptr),
      static_cast<const half_t*>(inputs[1].base_ptr),
      static_cast<const half_t*>(outputs[0].base_ptr)
  };
  return &params;
}

auto RmsNormKernel::create_layouts(TensorDesc* inputs, TensorDesc* outputs) {


  return make_tuple(
      // make_swizzled_layout(inputs[1].shape_0, inputs[1].shape_1,
      //                     inputs[1].stride_0, inputs[1].stride_1),
      // make_swizzled_layout(outputs[0].shape_0, outputs[0].shape_1,
      //                     outputs[0].stride_0, outputs[0].stride_1),
      // make_swizzled_layout(inputs[0].shape_0, inputs[0].shape_1,
      //                     inputs[0].stride_0, inputs[0].stride_1)
  );
}



template <typename Layouts>
__device__ void RmsNormKernel::execute(void* params, Layouts layouts,
                                      char* shared_buf, int tid) 
{
    auto& p = *static_cast<decltype(pack_parameters(nullptr,nullptr))*>(params);
    rms_norm_kernl_impl(
        p.input0, p.input1, p.output0,
        get<0>(layouts),
        get<1>(layouts),
        get<2>(layouts),
        shared_buf, tid
    );
}

template <typename Input0Layout, typename Input1Layout, typename Output0Layout>
__global__ void __launch_bounds__(128) rms_norm_kernl_impl(half_t* __restrict__ input_0, half_t const* __restrict__ input_1, half_t const* __restrict__ output_0) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000031_ptr = (half_t*)(buf + 128);
  half_t *stensor20000029_ptr = (half_t*)(buf + 4224);
  half_t *stensor20000030_ptr = (half_t*)(buf + 2176);
  half_t *stensor30000023_ptr = (half_t*)(buf + 14464);
  half_t *stensor20000023_ptr = (half_t*)(buf + 6272);
  half_t *stensor30000022_ptr = (half_t*)(buf + 4224);
  half_t *stensor20000022_ptr = (half_t*)(buf + 2176);
  half_t *stensor20000027_ptr = (half_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000003 -> stensor 20000022
  const half_t *dtensor10000003_tile_ptr = input_1 ;
  using DTensor10000003TileLayout = Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<4096>>>;
  using STensor20000022InputAtom = tb::InputChunkedAsyncCopy<half_t, Input0Layout, DTensor10000003TileLayout, NUM_THREADS>;
  half_t *stensor20000022_async_copy_buf = stensor30000022_ptr;
  // Copy for G->S: dtensor 10000004 -> stensor 20000023
  const half_t *dtensor10000004_tile_ptr = output_0  + blockIdx.x*64*1;
  using DTensor10000004TileLayout = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<4096>>>;
  using STensor20000023InputAtom = tb::InputChunkedAsyncCopy<half_t, Input1Layout, DTensor10000004TileLayout, NUM_THREADS>;
  half_t *stensor20000023_async_copy_buf = stensor30000023_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000031 -> dtensor 10000005
  half_t *dtensor10000005_tile_ptr = input_0  + blockIdx.x*64*16;
  using DTensor10000005TileLayout = Layout<Shape<Int<16>, Int<64>>, Stride<Int<1>, Int<16>>>;
  using STensor20000031OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000005TileLayout, Output0Layout, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20000027_ptr, thread_idx);
  
  using Matmul20000030LayoutA = Input0Layout;
  using Matmul20000030LayoutB = Input1Layout;
  using Matmul20000030LayoutC = layout::LayoutInfer<type::TB_MATMUL_OP, Input0Layout, Input1Layout>::LayoutOut;
  using Matmul20000030LayoutAAligned = Matmul20000030LayoutA;
  using Matmul20000030LayoutBAligned = Matmul20000030LayoutB;
  using Matmul20000030Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000030LayoutA, Matmul20000030LayoutB, Matmul20000030LayoutC, Matmul20000030LayoutAAligned, Matmul20000030LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000030_accum = Matmul20000030Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  {
    STensor20000023InputAtom::run(stensor20000023_async_copy_buf, dtensor10000004_tile_ptr, thread_idx);
    STensor20000022InputAtom::run(stensor20000022_async_copy_buf, dtensor10000003_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 64; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 64) {
        STensor20000023InputAtom::run(stensor20000023_ptr, dtensor10000004_tile_ptr + 262144*(for_idx+1), thread_idx);
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
      using InLayout = Input1Layout;
      using ElementUnaryOutLayput = layout::LayoutInfer<type::TB_SQUARE_OP, InLayout>::LayoutOut;
      using Kernel = tb::ElementUnaryKernel<half_t, tb::ElementUnaryOpType::SQUARE, ElementUnaryOutLayput, InLayout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueStoreAccum<half_t>>>;
      const float scalars[] = {0.000244f, 0.0f};
      Kernel::run(stensor20000027_ptr, stensor20000022_ptr, thread_idx, 0.000000, scalars);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000030Kernel::write_back_mma_rC(stensor20000030_ptr, matmul_20000030_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();

    // OP type: tb_reduction_1_op
    using ReductionInLayout = ElementUnaryOutLayput;
    using ReductionOutLayout = layout::LayoutInfer<type::TB_REDUCTION_1_OP, ReductionInLayout>::LayoutOut;
    using Kernel = tb::ReductionKernel<half_t, ReductionOutLayout, ReductionInLayout, 1, NUM_THREADS, tb::EpilogueSqrt<half_t, tb::EpilogueStore<half_t>>>;
    const float scalars[] = {0.000000f, 0.0f};
    Kernel::run(stensor20000029_ptr, stensor20000027_ptr, thread_idx, scalars);

  __syncthreads();

    // OP type: tb_div_op
    using ElementBinaryIn0Layout = Matmul20000030LayoutC;
    using ElementBinaryIn1Layout = ReductionOutLayout;
    using EleBinaryOutLayout = layout::LayoutInfer<type::TB_DIV_OP, ElementBinaryIn0Layout, ElementBinaryIn1Layout>::LayoutOut;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, EleBinaryOutLayout, ElementBinaryIn0Layout, ElementBinaryIn1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000031_ptr, stensor20000030_ptr, stensor20000029_ptr, thread_idx, scalars);

  __syncthreads();

    // OP type: tb_output_op
    STensor20000031OutputAtom::run(dtensor10000005_tile_ptr, stensor20000031_ptr, thread_idx);
}
