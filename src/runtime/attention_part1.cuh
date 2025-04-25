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
 
 
 struct AttentionPart1Kernel {
 
   static constexpr int input_nums = 3;
   static constexpr int output_nums = 1;

   static __device__ __forceinline__ void execute(TaskDesc &task_desc, int4 *tensor_offsets, int forloop_range);
 };
 
 
 template <typename Input0Layout, typename Input0LayoutDevice,
           typename Input1Layout, typename Input1LayoutDevice,
           typename Output0Layout, typename Output0LayoutDevice>
 __device__ __forceinline__ void attention_qkvproj_kernel_impl(bfloat16_t* __restrict__ dtensor10000007_ptr, bfloat16_t const* __restrict__ dtensor10000004_ptr, bfloat16_t const* __restrict__ dtensor10000005_ptr, bfloat16_t const* __restrict__ dtensor10000006_ptr, int4 offset_in0,
   int4 offset_in1,
   int4 offset_out0,
   int forloop_range);
 
 #define ATTENTIONPART1_KERNEL(I0, I0D, I1, I1D, O0, O0D) \
 attention_qkvproj_kernel_impl<I0, I0D, I1, I1D, O0, O0D>(dtensor10000007_ptr, dtensor10000004_ptr, dtensor10000005_ptr, dtensor10000006_ptr, tensor_offsets[0], tensor_offsets[1], tensor_offsets[2], forloop_range)
 
 __device__ __forceinline__ void AttentionPart1Kernel::execute(TaskDesc &task_desc, int4 *tensor_offsets, int forloop_range) 
 { 

  TensorDesc* inputs = task_desc.inputs;

  TensorDesc* outputs = task_desc.outputs;

   // Convenience aliases for readability
   const int *dim0 = inputs[0].dim;
   const int *stride0 = inputs[0].stride;
 
   const int *dim1 = inputs[1].dim;
   const int *stride1 = inputs[1].stride;
 
   const int *dim_out = outputs[0].dim;
   const int *stride_out = outputs[0].stride;

  //  int tile_x = task_desc.task_id /(task_desc.task_partition.y * task_desc.task_partition.z);
  //  int tile_y = (task_desc.task_id / task_desc.task_partition.z) % task_desc.task_partition.y;
  //  int tile_z = task_desc.task_id % task_desc.task_partition.z;
   
 
 
   bfloat16_t* __restrict__ dtensor10000007_ptr = static_cast<bfloat16_t*>(outputs[0].base_ptr);
   bfloat16_t const* __restrict__ dtensor10000004_ptr = static_cast<const bfloat16_t*>(inputs[0].base_ptr);
   bfloat16_t const* __restrict__ dtensor10000005_ptr= static_cast<const bfloat16_t*>(inputs[1].base_ptr);
   bfloat16_t const* __restrict__ dtensor10000006_ptr= static_cast<const bfloat16_t*>(inputs[2].base_ptr);
  
 
   if(dim0[0]==1 && dim0[1]==1152){
    using Input0Layout        = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
    using Input0LayoutDevice  = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<4096>>>;
    using Input1Layout        = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
    using Input1LayoutDevice  = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>;
    using Output0Layout       = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
    using Output0LayoutDevice = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
 
   ATTENTIONPART1_KERNEL(Input0Layout, Input0LayoutDevice, Input1Layout, Input1LayoutDevice, Output0Layout, Output0LayoutDevice);
   }else{
     assert(false && "unsupported layout");
   }
   
 }
 
 template <typename Input0Layout, typename Input0LayoutDevice, typename Input1Layout, typename Input1LayoutDevice, typename Output0Layout,  typename Output0LayoutDevice>
 __device__ __forceinline__ void attention_qkvproj_kernel_impl(bfloat16_t* __restrict__ dtensor10000007_ptr, bfloat16_t const* __restrict__ dtensor10000004_ptr, bfloat16_t const* __restrict__ dtensor10000005_ptr, bfloat16_t const* __restrict__ dtensor10000006_ptr, int4 offset_in0,
   int4 offset_in1,
   int4 offset_out0,
   int forloop_range) {
    return;
   
    int thread_idx = threadIdx.x;
    static constexpr int NUM_THREADS = 128;
    // STensors
    extern __shared__ char buf[];
    bfloat16_t *stensor20000023_ptr = (bfloat16_t*)(buf + 128);
    bfloat16_t *stensor20000021_ptr = (bfloat16_t*)(buf + 67456);
    bfloat16_t *stensor30000020_ptr = (bfloat16_t*)(buf + 51072);
    bfloat16_t *stensor20000020_ptr = (bfloat16_t*)(buf + 34688);
    bfloat16_t *stensor30000019_ptr = (bfloat16_t*)(buf + 18304);
    bfloat16_t *stensor20000019_ptr = (bfloat16_t*)(buf + 1920);
    bfloat16_t *stensor20000018_ptr = (bfloat16_t*)(buf + 128);
    *((uint128_t*)buf) = 0ul;
    
    // G->S copy atoms
    // Copy for G->S: dtensor 10000004 -> stensor 20000018
    const bfloat16_t *dtensor10000004_tile_ptr = dtensor10000004_ptr  + 0*1*3584 + 0*7*128;
    using DTensor10000004TileLayout = Layout<Shape<Int<128>, Int<7>, Int<1>>, Stride<Int<1>, Int<128>, Int<3584>>>;
    using STensor20000018InputAtom = tb::InputChunkedSyncCopy<bfloat16_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<7>, Int<1>>, Stride<Int<1>, Int<128>, Int<896>>>{})), DTensor10000004TileLayout, NUM_THREADS>;
    // Copy for G->S: dtensor 10000005 -> stensor 20000019
    const bfloat16_t *dtensor10000005_tile_ptr = dtensor10000005_ptr  + 0*1*524288 + 0*256*1;
    using DTensor10000005TileLayout = Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<4096>, Int<524288>>>;
    using STensor20000019InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>, Int<1>>, Stride<Int<1>, Int<64>, Int<8192>>>{})), DTensor10000005TileLayout, NUM_THREADS>;
    bfloat16_t *stensor20000019_async_copy_buf = stensor30000019_ptr;
    // Copy for G->S: dtensor 10000006 -> stensor 20000020
    const bfloat16_t *dtensor10000006_tile_ptr = dtensor10000006_ptr  + 0*1*524288 + 0*256*128;
    using DTensor10000006TileLayout = Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<524288>>>;
    using STensor20000020InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>, Int<1>>, Stride<Int<1>, Int<128>, Int<8192>>>{})), DTensor10000006TileLayout, NUM_THREADS>;
    bfloat16_t *stensor20000020_async_copy_buf = stensor30000020_ptr;
    
    STensor20000018InputAtom::run(stensor20000018_ptr, dtensor10000004_tile_ptr, thread_idx);
    
    // S->G copy atoms
    // Copy for S->G: stensor 20000023 -> dtensor 10000007
    bfloat16_t *dtensor10000007_tile_ptr = dtensor10000007_ptr  + 0*7*2048 + 0*128*1;
    using DTensor10000007TileLayout = Layout<Shape<Int<128>, Int<7>, Int<1>>, Stride<Int<1>, Int<2048>, Int<57344>>>;
    using STensor20000023OutputAtom = tb::OutputChunkedSyncCopy<bfloat16_t, DTensor10000007TileLayout, decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<7>, Int<1>>, Stride<Int<1>, Int<128>, Int<896>>>{})), NUM_THREADS>;
    
    
    using Matmul20000021LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<7>>, Stride<Int<1>, Int<128>>>{}));
    using Matmul20000021LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
    using Matmul20000021LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<7>>, Stride<Int<1>, Int<64>>>{}));
    using Matmul20000021LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<16>>, Stride<Int<1>, Int<128>>>{}));
    using Matmul20000021LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
    using Matmul20000021Kernel = tb::Matmul<bfloat16_t, SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000021LayoutA, Matmul20000021LayoutB, Matmul20000021LayoutC, Matmul20000021LayoutAAligned, Matmul20000021LayoutBAligned,NUM_THREADS, 0, false>;
    
    using Matmul20000023LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<7>>, Stride<Int<1>, Int<64>>>{}));
    using Matmul20000023LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
    using Matmul20000023LayoutC = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<7>>, Stride<Int<1>, Int<128>>>{}));
    using Matmul20000023LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<16>>, Stride<Int<1>, Int<64>>>{}));
    using Matmul20000023LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<128>, Int<64>>, Stride<Int<1>, Int<128>>>{}));
    using Matmul20000023Kernel = tb::Matmul<bfloat16_t, SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<Int<1>, Int<4>, _1>>, true, false, Matmul20000023LayoutA, Matmul20000023LayoutB, Matmul20000023LayoutC, Matmul20000023LayoutAAligned, Matmul20000023LayoutBAligned,NUM_THREADS, 0, false>;
    auto matmul_20000023_accum = Matmul20000023Kernel::get_mma_rC(thread_idx);
    
    
    __syncthreads();
    
    {
      STensor20000020InputAtom::run(stensor20000020_async_copy_buf, dtensor10000006_tile_ptr, thread_idx);
      STensor20000019InputAtom::run(stensor20000019_async_copy_buf, dtensor10000005_tile_ptr, thread_idx);
      cute::cp_async_fence();
    }
  
    
    // The main loop
    for (int for_idx = 0; for_idx < 4; for_idx++) {
      {
        // Issue async copies for the next round
        if (for_idx+1 != 4) {
          STensor20000020InputAtom::run(stensor20000020_ptr, dtensor10000006_tile_ptr + 0*(for_idx+1), thread_idx);
          STensor20000019InputAtom::run(stensor20000019_ptr, dtensor10000005_tile_ptr + 0*(for_idx+1), thread_idx);
        }
        cute::cp_async_fence();
        // Wait for the async copies in the last round to finish
        cute::cp_async_wait<1>();
        // Switch buffers
        SWAP(stensor20000020_ptr, stensor20000020_async_copy_buf);
        SWAP(stensor20000019_ptr, stensor20000019_async_copy_buf);
      }
      
      __syncthreads();
      {
        // OP type: tb_matmul_op
        auto mma_rC = Matmul20000021Kernel::get_mma_rC(thread_idx);
        Matmul20000021Kernel::run(mma_rC, stensor20000018_ptr, stensor20000019_ptr, (char*)(buf+0), thread_idx);
        Matmul20000021Kernel::write_back_mma_rC(stensor20000021_ptr, mma_rC, thread_idx);
      }
      __syncthreads();
      {
        // OP type: tb_matmul_op
        Matmul20000023Kernel::run(matmul_20000023_accum, stensor20000021_ptr, stensor20000020_ptr, (char*)(buf+0), thread_idx);
      }
    }
    
    // Write back in-register accumulators
    __syncthreads();
    Matmul20000023Kernel::write_back_mma_rC(stensor20000023_ptr, matmul_20000023_accum, thread_idx);
    // The epilogue (kernels outside the loop)
    __syncthreads();
    {
      // OP type: tb_output_op
      STensor20000023OutputAtom::run(dtensor10000007_tile_ptr, stensor20000023_ptr, thread_idx);
    }
 }
 
 } // namespace runtime
 } // namespace mirage