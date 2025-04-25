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
 
 
 struct AttentionPart2Kernel {
 
   static constexpr int input_nums = 2;
   static constexpr int output_nums = 1;

   static __device__ __forceinline__ void execute(TaskDesc &task_desc, int4 *tensor_offsets, int forloop_range);
 };
 
 
 template <typename Input0Layout, typename Input0LayoutDevice,
           typename Input1Layout, typename Input1LayoutDevice,
           typename Output0Layout, typename Output0LayoutDevice>
 __device__ __forceinline__ void attention_reduction_kernel_impl(bfloat16_t* __restrict__ dtensor10000003_ptr, bfloat16_t const* __restrict__ dtensor10000002_ptr, int tile_x, int tile_y);
 
 #define ATTENTIONPART2_KERNEL(I0, I0D, I1, I1D, O0, O0D) \
   attention_reduction_kernel_impl<I0, I0D, I1, I1D, O0, O0D>(dtensor10000003_ptr, dtensor10000002_ptr, tile_x, tile_y)
 
 __device__ __forceinline__ void AttentionPart2Kernel::execute(TaskDesc &task_desc, int4 *tensor_offsets, int forloop_range) 
 { 
   // Convenience aliases for readability
   TensorDesc* inputs = task_desc.inputs;

   TensorDesc* outputs = task_desc.outputs;
   const int *dim0 = inputs[0].dim;
   const int *stride0 = inputs[0].stride;

   int tile_x = task_desc.task_id /(task_desc.task_partition.y * task_desc.task_partition.z);
   int tile_y = (task_desc.task_id / task_desc.task_partition.z) % task_desc.task_partition.y;
  
   if(threadIdx.x == 0){
    printf("tilex tiley%d, %d, %d, %d, %d\n", tile_x, tile_y,  task_desc.task_id, task_desc.task_partition.y, task_desc.task_partition.z);
   }
   
 
   bfloat16_t* __restrict__ dtensor10000003_ptr = static_cast<bfloat16_t*>(outputs[0].base_ptr);
   bfloat16_t const* __restrict__ dtensor10000002_ptr = static_cast<const bfloat16_t*>(inputs[0].base_ptr);
 
   if(dim0[0]==1 && dim0[1]==16 && dim0[2]==128){
    using Input0Layout        = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<64>>>;
    using Input0LayoutDevice  = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<4096>>>;
    using Input1Layout        = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
    using Input1LayoutDevice  = Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>;
    using Output0Layout       = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
    using Output0LayoutDevice = Layout<Shape<Int<64>, Int<1>>, Stride<Int<1>, Int<1>>>;
 
   ATTENTIONPART2_KERNEL(Input0Layout, Input0LayoutDevice, Input1Layout, Input1LayoutDevice, Output0Layout, Output0LayoutDevice);
   }else{
     assert(false && "unsupported layout");
   }
   
 }
 
 template <typename Input0Layout, typename Input0LayoutDevice, typename Input1Layout, typename Input1LayoutDevice, typename Output0Layout,  typename Output0LayoutDevice>
 __device__ __forceinline__ void attention_reduction_kernel_impl(bfloat16_t* __restrict__ dtensor10000003_ptr, bfloat16_t const* __restrict__ dtensor10000002_ptr, int tile_x, int tile_y) {
   
    int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  bfloat16_t *stensor20000011_ptr = (bfloat16_t*)(buf + 1152);
  bfloat16_t *stensor30000009_ptr = (bfloat16_t*)(buf + 2176);
  bfloat16_t *stensor20000009_ptr = (bfloat16_t*)(buf + 1152);
  bfloat16_t *stensor20000010_ptr = (bfloat16_t*)(buf + 128);
  *((uint128_t*)buf) = 0ul;
  
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000002 -> stensor 20000009
  const bfloat16_t *dtensor10000002_tile_ptr = dtensor10000002_ptr  + tile_x*1*14336 + tile_y*128*1;
  using DTensor10000002TileLayout = Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<3584>, Int<14336>>>;
  using STensor20000009InputAtom = tb::InputChunkedAsyncCopy<bfloat16_t, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, DTensor10000002TileLayout, NUM_THREADS>;
  bfloat16_t *stensor20000009_async_copy_buf = stensor30000009_ptr;
  
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000011 -> dtensor 10000003
  bfloat16_t *dtensor10000003_tile_ptr = dtensor10000003_ptr  + tile_y*128*1;
  using DTensor10000003TileLayout = Layout<Shape<Int<128>, Int<1>, Int<1>>, Stride<Int<1>, Int<3584>, Int<3584>>>;
  using STensor20000011OutputAtom = tb::OutputChunkedSyncCopy<bfloat16_t, DTensor10000003TileLayout, Layout<Shape<Int<128>, Int<1>, Int<1>>, Stride<Int<1>, Int<128>, Int<128>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<bfloat16_t, 512, NUM_THREADS>::run(stensor20000010_ptr, thread_idx);
  
  __syncthreads();
  
  {
    STensor20000009InputAtom::run(stensor20000009_async_copy_buf, dtensor10000002_tile_ptr, thread_idx);
    cute::cp_async_fence();
  }
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // Issue async copies for the next round
      if (for_idx+1 != 1) {
        STensor20000009InputAtom::run(stensor20000009_ptr, dtensor10000002_tile_ptr + 14336*(for_idx+1), thread_idx);
      }
      cute::cp_async_fence();
      // Wait for the async copies in the last round to finish
      cute::cp_async_wait<1>();
      // Switch buffers
      SWAP(stensor20000009_ptr, stensor20000009_async_copy_buf);
    }
    __syncthreads();
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<bfloat16_t, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>, NUM_THREADS>;
      Kernel::run(stensor20000010_ptr, stensor20000009_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_1_op
    using InLayout = Layout<Shape<Int<128>, Int<4>, Int<1>>, Stride<Int<1>, Int<128>, Int<512>>>;
    using OutLayout = Layout<Shape<Int<128>, Int<1>, Int<1>>, Stride<Int<1>, Int<128>, Int<128>>>;
    using Kernel = tb::ReductionKernel<bfloat16_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<bfloat16_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000011_ptr, stensor20000010_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000011OutputAtom::run(dtensor10000003_tile_ptr, stensor20000011_ptr, thread_idx);
  }
 }
 
 } // namespace runtime
 } // namespace mirage