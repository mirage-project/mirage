/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "mirage/transpiler/runtime/runtime.h"
#include "mirage/runtime/runtime.h"
#include "layout_infer.h"

using namespace cute;

namespace mirage {
namespace runtime {


struct SiluMulMatmulKernel {

  static constexpr int input_nums = 2;
  static constexpr int output_nums = 1;

  static __device__ __forceinline__ void execute(TaskDesc &task_desc, int4 *tensor_offsets, int forloop_range);
};


template <typename Input0Layout, typename Input0LayoutDevice,
          typename Input1Layout, typename Input1LayoutDevice,
          typename Output0Layout, typename Output0LayoutDevice>
__device__ __forceinline__ void silu_mul_matmul_kernel_impl(bfloat16_t* __restrict__ dtensor10000007_ptr, bfloat16_t const* __restrict__ dtensor10000004_ptr, bfloat16_t const* __restrict__ dtensor10000006_ptr, int tile_x);

#define SILUMATMUL_KERNEL(I0, I0D, I1, I1D, O0, O0D) \
  silu_mul_matmul_kernel_impl<I0, I0D, I1, I1D, O0, O0D>(dtensor10000005_ptr, dtensor10000003_ptr, dtensor10000004_ptr, tile_x)

// template <typename Layouts>
__device__ __forceinline__ void SiluMulMatmulKernel::execute(TaskDesc &task_desc, int4 *tensor_offsets, int forloop_range) 
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

  int tile_x = task_desc.task_id /(task_desc.task_partition.y * task_desc.task_partition.z);


  bfloat16_t* __restrict__ dtensor10000005_ptr = static_cast<bfloat16_t*>(outputs[0].base_ptr);
  bfloat16_t const* __restrict__ dtensor10000003_ptr = static_cast<const bfloat16_t*>(inputs[0].base_ptr);
  bfloat16_t const* __restrict__ dtensor10000004_ptr= static_cast<bfloat16_t*>(inputs[1].base_ptr);
  using Input0Layout        = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
  using Input0LayoutDevice  = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<4096>>>;
  using Input1Layout        = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Input1LayoutDevice  = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>;
  using Output0Layout       = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
  using Output0LayoutDevice = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;

  if(dim0[0]==1 && dim0[1]==128){
  SILUMATMUL_KERNEL(Input0Layout, Input0LayoutDevice, Input1Layout, Input1LayoutDevice, Output0Layout, Output0LayoutDevice);
  }else{
    assert(false && "unsupported layout");
  }
  
}

template <typename Input0Layout, typename Input0LayoutDevice, typename Input1Layout, typename Input1LayoutDevice, typename Output0Layout,  typename Output0LayoutDevice>
__device__ __forceinline__ void silu_mul_matmul_kernel_impl(bfloat16_t* __restrict__ dtensor10000007_ptr, bfloat16_t const* __restrict__ dtensor10000004_ptr, bfloat16_t const* __restrict__ dtensor10000006_ptr, int tile_x) {
  
    int thread_idx = threadIdx.x;
    static constexpr int NUM_THREADS = 128;
    // STensors
    extern __shared__ char buf[];
    bfloat16_t *stensor20000027_ptr = (bfloat16_t*)(buf + 128);
    bfloat16_t *stensor20000025_ptr = (bfloat16_t*)(buf + 17152);
    bfloat16_t *stensor20000024_ptr = (bfloat16_t*)(buf + 17024);
    bfloat16_t *stensor30000022_ptr = (bfloat16_t*)(buf + 16896);
    bfloat16_t *stensor20000023_ptr = (bfloat16_t*)(buf + 8704);
    bfloat16_t *stensor20000022_ptr = (bfloat16_t*)(buf + 8576);
    bfloat16_t *stensor30000021_ptr = (bfloat16_t*)(buf + 8448);
    bfloat16_t *stensor30000023_ptr = (bfloat16_t*)(buf + 256);
    bfloat16_t *stensor20000021_ptr = (bfloat16_t*)(buf + 128);
    *((uint128_t*)buf) = 0ul;
    
    // G->S copy atoms
    // Copy for G->S: dtensor 10000004 -> stensor 20000021
    const bfloat16_t *dtensor10000004_tile_ptr = dtensor10000004_ptr ;
    using DTensor10000004TileLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<3584>>>;
    using STensor20000021InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>, DTensor10000004TileLayout, NUM_THREADS>;
    bfloat16_t *stensor20000021_async_copy_buf = stensor30000021_ptr;
    // Copy for G->S: dtensor 10000005 -> stensor 20000022
    const bfloat16_t *dtensor10000005_tile_ptr = dtensor10000004_ptr + 3584 ;
    using DTensor10000005TileLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<3584>>>;
    using STensor20000022InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>, DTensor10000005TileLayout, NUM_THREADS>;
    bfloat16_t *stensor20000022_async_copy_buf = stensor30000022_ptr;
    // Copy for G->S: dtensor 10000006 -> stensor 20000023
    const bfloat16_t *dtensor10000006_tile_ptr = dtensor10000006_ptr  + tile_x*64*1;
    using DTensor10000006TileLayout = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<3584>>>;
    using STensor20000023InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{})), DTensor10000006TileLayout, NUM_THREADS>;
    bfloat16_t *stensor20000023_async_copy_buf = stensor30000023_ptr;
    
    
    // S->G copy atoms
    // Copy for S->G: stensor 20000027 -> dtensor 10000007
    bfloat16_t *dtensor10000007_tile_ptr = dtensor10000007_ptr  + tile_x*64*1;
    using DTensor10000007TileLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<3584>>>;
    using STensor20000027OutputAtom = tb::OutputChunkedSyncCopy<bfloat16_t, DTensor10000007TileLayout, Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>, NUM_THREADS>;
    
    
    using Matmul20000027LayoutA = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
    using Matmul20000027LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
    using Matmul20000027LayoutC = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
    using Matmul20000027LayoutAAligned = Layout<Shape<Int<64>, Int<16>>, Stride<Int<16>, Int<1>>>;
    using Matmul20000027LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
    using Matmul20000027Kernel = tb::Matmul<bfloat16_t, SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000027LayoutA, Matmul20000027LayoutB, Matmul20000027LayoutC, Matmul20000027LayoutAAligned, Matmul20000027LayoutBAligned,NUM_THREADS, 0, false>;
    auto matmul_20000027_accum = Matmul20000027Kernel::get_mma_rC(thread_idx);
    
    {
      STensor20000023InputAtom::run(stensor20000023_async_copy_buf, dtensor10000006_tile_ptr, thread_idx);
      STensor20000022InputAtom::run(stensor20000022_async_copy_buf, dtensor10000005_tile_ptr, thread_idx);
      STensor20000021InputAtom::run(stensor20000021_async_copy_buf, dtensor10000004_tile_ptr, thread_idx);
      cute::cp_async_fence();
    }
    
    // The main loop
    for (int for_idx = 0; for_idx < 56; for_idx++) {
      {
        // Issue async copies for the next round
        if (for_idx+1 != 56) {
          STensor20000023InputAtom::run(stensor20000023_ptr, dtensor10000006_tile_ptr + 229376*(for_idx+1), thread_idx);
          STensor20000022InputAtom::run(stensor20000022_ptr, dtensor10000005_tile_ptr + 64*(for_idx+1), thread_idx);
          STensor20000021InputAtom::run(stensor20000021_ptr, dtensor10000004_tile_ptr + 64*(for_idx+1), thread_idx);
        }
        cute::cp_async_fence();
        // Wait for the async copies in the last round to finish
        cute::cp_async_wait<1>();
        // Switch buffers
        SWAP(stensor20000023_ptr, stensor20000023_async_copy_buf);
        SWAP(stensor20000022_ptr, stensor20000022_async_copy_buf);
        SWAP(stensor20000021_ptr, stensor20000021_async_copy_buf);
      }
      __syncthreads();
      {
        // OP type: tb_silu_op
        using InLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
        using OutLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
        using Kernel = tb::ElementUnaryKernel<bfloat16_t, tb::ElementUnaryOpType::SILU, OutLayout, InLayout, NUM_THREADS, tb::EpilogueStore<bfloat16_t>>;
        const float scalars[] = {0.0f};
        Kernel::run(stensor20000024_ptr, stensor20000021_ptr, thread_idx, 0.000000, scalars);
      }
      __syncthreads();
      {
        // OP type: tb_mul_op
        using In0Layout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
        using In1Layout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
        using OutLayout = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
        using Kernel = tb::ElementBinaryKernel<bfloat16_t, tb::ElementBinaryOpType::MUL, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<bfloat16_t>>;
        const float scalars[] = {0.0f};
        Kernel::run(stensor20000025_ptr, stensor20000024_ptr, stensor20000022_ptr, thread_idx, scalars);
      }
      __syncthreads();
      {
        // OP type: tb_matmul_op
        Matmul20000027Kernel::run(matmul_20000027_accum, stensor20000025_ptr, stensor20000023_ptr, (char*)(buf+0), thread_idx);
      }
    }
    
    // Write back in-register accumulators
    __syncthreads();
    Matmul20000027Kernel::write_back_mma_rC(stensor20000027_ptr, matmul_20000027_accum, thread_idx);
    // The epilogue (kernels outside the loop)
    __syncthreads();
    {
      // OP type: tb_output_op
      STensor20000027OutputAtom::run(dtensor10000007_tile_ptr, stensor20000027_ptr, thread_idx);
    }
}

} // namespace runtime
} // namespace mirage