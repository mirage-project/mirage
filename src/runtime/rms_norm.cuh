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

  struct Params {
    half_t* input0;
    const half_t* input1;
    const half_t* output0;
    int4  offset_in0;
    int4 offset_in1;
    int4 offset_out0;
    int forloop_range;
};

  static void* pack_parameters(TensorDesc* inputs, TensorDesc* outputs, int4 *tensor_offsets);
  static auto create_layouts(TensorDesc* inputs, TensorDesc* outputs);

  template <typename Layouts>
  static __device__ void execute(void* params, Layouts layouts);
};

void* RmsNormKernel::pack_parameters(TensorDesc* inputs, TensorDesc* outputs, int4 *tensor_offsets) {

  static Params params;
  
  params.input0 = static_cast<half_t*>(inputs[0].base_ptr);
  params.input1 = static_cast<const half_t*>(inputs[1].base_ptr);
  params.output0 = static_cast<const half_t*>(outputs[0].base_ptr);

  params.offset_in0 = *tensor_offsets;
  params.offset_in1 = *(tensor_offsets + 1);
  params.offset_out0 = *(tensor_offsets + 2);
  // params.forloop_range = forloop_range;

  return &params;
}

auto RmsNormKernel::create_layouts(TensorDesc* inputs, TensorDesc* outputs) {

  auto make_dtensor_layout = [](const TensorDesc& desc) {
    return make_layout(
      make_shape(desc.dim[1], desc.dim[0]),
      make_stride(desc.dtensor_stride[1], desc.dtensor_stride[0])
    );
  };

  auto make_stensor_layout = [](const TensorDesc& desc) {
    int d0 = desc.dim[0];
    int d1 = desc.dim[1];
    int s0 = desc.stride[0];
    int s1 = desc.stride[1];

    int innermost_dim_size = desc.dim[desc.innermost_dim];
    int base = innermost_dim_size / 8;
    auto layout = make_layout(make_shape(d1, d0), make_stride(s1, s0));
    if (base > 0 && (base & (base - 1)) == 0) {
      int log2_base = __builtin_ctz(innermost_dim_size);
      switch (log2_base) {
        case 3: return composition(Swizzle<3, 3, 3>{}, layout);
        case 4: return composition(Swizzle<3, 3, 4>{}, layout);
        default: assert(false);
    }
    } else {
      return layout;
    }
  };

  // Input layouts
  auto input0_smem_layout = make_stensor_layout(inputs[0]);
  auto input0_dtensor_layout = make_dtensor_layout(inputs[0]);



  auto input1_dtensor_layout  = make_dtensor_layout(inputs[1]);
  auto input1_smem_layout = make_stensor_layout(inputs[1]);

  auto output0_dtensor_layout  = make_dtensor_layout(outputs[0]);
  auto output0_smem_layout = make_stensor_layout(outputs[0]);

  return make_tuple(
    input0_smem_layout,
    input0_dtensor_layout,
    input1_smem_layout,
    input1_dtensor_layout,
    output0_smem_layout,
    output0_dtensor_layout
  );
}

template <typename Input0Layout, typename Input0LayoutDevice,
          typename Input1Layout, typename Input1LayoutDevice,
          typename Output0Layout, typename Output0LayoutDevice>
__device__ void rms_norm_kernel_impl(RmsNormKernel::Params const &params);

template <typename Layouts>
__device__ void RmsNormKernel::execute(void* params, Layouts layouts) 
{
  auto& p = *static_cast<Params*>(params);

  using Input0Layout        = typename std::tuple_element<0, Layouts>::type;
  using Input0LayoutDevice  = typename std::tuple_element<1, Layouts>::type;
  using Input1Layout        = typename std::tuple_element<2, Layouts>::type;
  using Input1LayoutDevice  = typename std::tuple_element<3, Layouts>::type;
  using Output0Layout       = typename std::tuple_element<4, Layouts>::type;
  using Output0LayoutDevice = typename std::tuple_element<5, Layouts>::type;
  rms_norm_kernel_impl<Input0Layout, Input0LayoutDevice,
                      Input1Layout, Input1LayoutDevice,
                      Output0Layout, Output0LayoutDevice>(p);
}

template <typename Input0Layout, typename Input0LayoutDevice, typename Input1Layout, typename Input1LayoutDevice, typename Output0Layout,  typename Output0LayoutDevice>
__device__ void rms_norm_kernel_impl(RmsNormKernel::Params const &params) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  half_t* __restrict__ input_0 = params.input0;
  half_t const* __restrict__ input_1 = params.input1;
  half_t const* __restrict__ output_0 = params.output0;
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
  const half_t *dtensor10000003_tile_ptr = input_1 + blockIdx.x * params.offset_in0.x + blockIdx.y * params.offset_in0.y + blockIdx.z * params.offset_in0.z;
  using DTensor10000003TileLayout = Input0LayoutDevice;
  using STensor20000022InputAtom = tb::InputChunkedAsyncCopy<half_t, Input0Layout, DTensor10000003TileLayout, NUM_THREADS>;
  half_t *stensor20000022_async_copy_buf = stensor30000022_ptr;
  // Copy for G->S: dtensor 10000004 -> stensor 20000023
  const half_t *dtensor10000004_tile_ptr = output_0  + blockIdx.x * params.offset_in1.x + blockIdx.y * params.offset_in1.y + blockIdx.z * params.offset_in1.z;
  using DTensor10000004TileLayout = Input1LayoutDevice;
  using STensor20000023InputAtom = tb::InputChunkedAsyncCopy<half_t, Input1Layout, DTensor10000004TileLayout, NUM_THREADS>;
  half_t *stensor20000023_async_copy_buf = stensor30000023_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000031 -> dtensor 10000005
  half_t *dtensor10000005_tile_ptr = input_0 + blockIdx.x * params.offset_out0.x + blockIdx.y * params.offset_out0.y + blockIdx.z * params.offset_out0.z;
  using DTensor10000005TileLayout = Output0LayoutDevice;
  using STensor20000031OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000005TileLayout, Output0Layout, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 1024, NUM_THREADS>::run(stensor20000027_ptr, thread_idx);
  
  using Matmul20000030LayoutA = Input0Layout;
  using Matmul20000030LayoutB = Input1Layout;
  using Matmul20000030LayoutC = typename LayoutInfer<type::TB_MATMUL_OP, Input0Layout, Input1Layout>::LayoutOut;
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
  for (int for_idx = 0; for_idx < params.forloop_range; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != params.forloop_range) {
        STensor20000023InputAtom::run(stensor20000023_ptr, dtensor10000004_tile_ptr + params.offset_in0.w*(for_idx+1), thread_idx);
        STensor20000022InputAtom::run(stensor20000022_ptr, dtensor10000003_tile_ptr + params.offset_in1.w*(for_idx+1), thread_idx);
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
      using ElementUnaryOutLayput = typename LayoutInfer<type::TB_SQUARE_OP, InLayout>::LayoutOut;
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
    using ReductionInLayout = Input0Layout;
    using ReductionOutLayout = typename LayoutInfer<type::TB_REDUCTION_1_OP, ReductionInLayout>::LayoutOut;
    using Kernel = tb::ReductionKernel<half_t, ReductionOutLayout, ReductionInLayout, 1, NUM_THREADS, tb::EpilogueSqrt<half_t, tb::EpilogueStore<half_t>>>;
    const float scalars_red[] = {0.000000f, 0.0f};
    Kernel::run(stensor20000029_ptr, stensor20000027_ptr, thread_idx, scalars_red);

  __syncthreads();

    // OP type: tb_div_op
    using ElementBinaryIn0Layout = Matmul20000030LayoutC;
    using ElementBinaryIn1Layout = ReductionOutLayout;
    using EleBinaryOutLayout = typename LayoutInfer<type::TB_DIV_OP, ElementBinaryIn0Layout, ElementBinaryIn1Layout>::LayoutOut;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, EleBinaryOutLayout, ElementBinaryIn0Layout, ElementBinaryIn1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars_div[] = {0.0f};
    Kernel::run(stensor20000031_ptr, stensor20000030_ptr, stensor20000029_ptr, thread_idx, scalars_div);

  __syncthreads();

    // OP type: tb_output_op
    STensor20000031OutputAtom::run(dtensor10000005_tile_ptr, stensor20000031_ptr, thread_idx);
}

} // namespace runtime
} // namespace mirage