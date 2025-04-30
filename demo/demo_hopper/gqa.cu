#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(128) custom_kernel_0(half_t* __restrict__ dtensor10000010_ptr, half_t* __restrict__ dtensor10000011_ptr, half_t* __restrict__ dtensor10000012_ptr, half_t const* __restrict__ dtensor10000007_ptr, half_t const* __restrict__ dtensor10000008_ptr, half_t const* __restrict__ dtensor10000009_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000044_ptr = (half_t*)(buf + 256);
  half_t *stensor20000046_ptr = (half_t*)(buf + 128);
  half_t *stensor20000042_ptr = (half_t*)(buf + 49280);
  half_t *stensor20000039_ptr = (half_t*)(buf + 8320);
  half_t *stensor20000038_ptr = (half_t*)(buf + 128);
  half_t *stensor20000045_ptr = (half_t*)(buf + 32896);
  half_t *stensor20000040_ptr = (half_t*)(buf + 16512);
  half_t *stensor20000047_ptr = (half_t*)(buf + 24704);
  half_t *stensor20000041_ptr = (half_t*)(buf + 41088);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000007 -> stensor 20000038
  const half_t *dtensor10000007_tile_ptr = dtensor10000007_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000007TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000038InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000007TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000008 -> stensor 20000039
  const half_t *dtensor10000008_tile_ptr = dtensor10000008_ptr  + blockIdx.x*1*65536 + blockIdx.y*64*1;
  using DTensor10000008TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<65536>>>;
  using STensor20000039InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000008TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000009 -> stensor 20000040
  const half_t *dtensor10000009_tile_ptr = dtensor10000009_ptr  + blockIdx.x*1*65536 + blockIdx.y*64*64;
  using DTensor10000009TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<65536>>>;
  using STensor20000040InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), DTensor10000009TileLayout, NUM_THREADS>;
  
  STensor20000038InputAtom::run(stensor20000038_ptr, dtensor10000007_tile_ptr, thread_idx);
  STensor20000039InputAtom::run(stensor20000039_ptr, dtensor10000008_tile_ptr, thread_idx);
  STensor20000040InputAtom::run(stensor20000040_ptr, dtensor10000009_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000044 -> dtensor 10000010
  half_t *dtensor10000010_tile_ptr = dtensor10000010_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*256 + blockIdx.z*64*1;
  using DTensor10000010TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000044OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000010TileLayout, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
  // Copy for S->G: stensor 20000047 -> dtensor 10000012
  half_t *dtensor10000012_tile_ptr = dtensor10000012_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000012TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20000047OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000012TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000046 -> dtensor 10000011
  half_t *dtensor10000011_tile_ptr = dtensor10000011_ptr  + blockIdx.x*1*4096 + blockIdx.y*1*256 + blockIdx.z*64*1;
  using DTensor10000011TileLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20000046OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000011TileLayout, Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20000047_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20000045_ptr, thread_idx);
  
  using Matmul20000041LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000041LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000041LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000041LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000041LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000041Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20000041LayoutA, Matmul20000041LayoutB, Matmul20000041LayoutC, Matmul20000041LayoutAAligned, Matmul20000041LayoutBAligned,NUM_THREADS, 0, false>;
  
  using Matmul20000044LayoutA = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000044LayoutB = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000044LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000044LayoutAAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000044LayoutBAligned = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000044Kernel = tb::Matmul<half_t, SM80_16x8x16_F16F16F16F16_TN, Layout<Shape<Int<2>, Int<2>, _1>>, true, false, Matmul20000044LayoutA, Matmul20000044LayoutB, Matmul20000044LayoutC, Matmul20000044LayoutAAligned, Matmul20000044LayoutBAligned,NUM_THREADS, 0, false>;
  auto matmul_20000044_accum = Matmul20000044Kernel::get_mma_rC(thread_idx);
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 4; for_idx++) {
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000041Kernel::get_mma_rC(thread_idx);
      Matmul20000041Kernel::run(mma_rC, stensor20000038_ptr, stensor20000039_ptr, (char*)(buf+0), thread_idx);
      Matmul20000041Kernel::write_back_mma_rC(stensor20000041_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_exp_op
      using InLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{}));
      using OutLayout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{}));
      using Kernel = tb::ElementUnaryKernel<half_t, tb::ElementUnaryOpType::EXP, OutLayout, InLayout, NUM_THREADS, tb::EpilogueStore<half_t>>;
      const float scalars[] = {0.0f};
      Kernel::run(stensor20000042_ptr, stensor20000041_ptr, thread_idx, 0.000000, scalars);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000047_ptr, stensor20000041_ptr, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      Matmul20000044Kernel::run(matmul_20000044_accum, stensor20000042_ptr, stensor20000040_ptr, (char*)(buf+0), thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000045_ptr, stensor20000042_ptr, thread_idx);
    }
  }
  
  // Write back in-register accumulators
  __syncthreads();
  Matmul20000044Kernel::write_back_mma_rC(stensor20000044_ptr, matmul_20000044_accum, thread_idx);
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
    using OutLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000046_ptr, stensor20000045_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20000044OutputAtom::run(dtensor10000010_tile_ptr, stensor20000044_ptr, thread_idx);
  }
  {
    // OP type: tb_output_op
    STensor20000047OutputAtom::run(dtensor10000012_tile_ptr, stensor20000047_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000046OutputAtom::run(dtensor10000011_tile_ptr, stensor20000046_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000013_ptr, half_t const* __restrict__ dtensor10000010_ptr, half_t const* __restrict__ dtensor10000011_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000061_ptr = (half_t*)(buf + 2208);
  half_t *stensor20000056_ptr = (half_t*)(buf + 32896);
  half_t *stensor20000055_ptr = (half_t*)(buf + 128);
  half_t *stensor20000057_ptr = (half_t*)(buf + 33408);
  half_t *stensor20000059_ptr = (half_t*)(buf + 66176);
  half_t *stensor20000058_ptr = (half_t*)(buf + 128);
  half_t *stensor20000060_ptr = (half_t*)(buf + 2176);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000010 -> stensor 20000055
  const half_t *dtensor10000010_tile_ptr = dtensor10000010_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1;
  using DTensor10000010TileLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<256>, Int<262144>>>;
  using STensor20000055InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, DTensor10000010TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000011 -> stensor 20000056
  const half_t *dtensor10000011_tile_ptr = dtensor10000011_ptr  + blockIdx.x*1*4096 + blockIdx.y*16*1;
  using DTensor10000011TileLayout = Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20000056InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, DTensor10000011TileLayout, NUM_THREADS>;
  
  STensor20000055InputAtom::run(stensor20000055_ptr, dtensor10000010_tile_ptr, thread_idx);
  STensor20000056InputAtom::run(stensor20000056_ptr, dtensor10000011_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000061 -> dtensor 10000013
  half_t *dtensor10000013_tile_ptr = dtensor10000013_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*1;
  using DTensor10000013TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<16384>>>;
  using STensor20000061OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000013TileLayout, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 16384, NUM_THREADS>::run(stensor20000057_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 256, NUM_THREADS>::run(stensor20000059_ptr, thread_idx);
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, NUM_THREADS>;
      Kernel::run(stensor20000057_ptr, stensor20000055_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, NUM_THREADS>;
      Kernel::run(stensor20000059_ptr, stensor20000056_ptr, thread_idx);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_to_dimx_op
    using InLayout = Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000058_ptr, stensor20000057_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000060_ptr, stensor20000059_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000061_ptr, stensor20000058_ptr, stensor20000060_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000061OutputAtom::run(dtensor10000013_tile_ptr, stensor20000061_ptr, thread_idx);
  }
}


static void _init() {
}


static void _execute_mugraph(std::vector<void const *> input_tensors, std::vector<void*> output_tensors, void* buf, cudaStream_t stream, void * profiler_buffer){
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_input_op
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000010 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000011 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000012 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000007 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000008 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000009 = (half_t*)input_tensors.at(2);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 57472;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 57472);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000010, dtensor10000011, dtensor10000012, dtensor10000007, dtensor10000008, dtensor10000009);
  }
  {
    // OP type: kn_output_op
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000013 = (half_t*)output_tensors.at(1);
    half_t *dtensor10000010 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000011 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 66688;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 66688);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000013, dtensor10000010, dtensor10000011);
  }
  {
    // OP type: kn_output_op
  }
}

