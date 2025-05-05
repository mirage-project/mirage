#define NUM_GPUS 1
#define USE_NVSHMEM false
#include "runtime.h"
using namespace cute;

__global__ void __launch_bounds__(256) custom_kernel_0(half_t* __restrict__ dtensor10000015_ptr, half_t* __restrict__ dtensor10000016_ptr, half_t const* __restrict__ dtensor10000009_ptr, half_t const* __restrict__ dtensor10000010_ptr, half_t const* __restrict__ dtensor10000011_ptr, half_t const* __restrict__ dtensor10000012_ptr, half_t const* __restrict__ dtensor10000013_ptr, half_t const* __restrict__ dtensor10000014_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 256;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000147_ptr = (half_t*)(buf + 128);
  half_t *stensor20000141_ptr = (half_t*)(buf + 45264);
  half_t *stensor20000139_ptr = (half_t*)(buf + 41168);
  half_t *stensor20000143_ptr = (half_t*)(buf + 41152);
  half_t *stensor20000135_ptr = (half_t*)(buf + 57520);
  half_t *stensor20000132_ptr = (half_t*)(buf + 41136);
  unknown *stensor20000131_ptr = (unknown*)(buf + 49328);
  unknown *stensor20000118_ptr = (unknown*)(buf + 24736);
  unknown *stensor20000117_ptr = (unknown*)(buf + 24720);
  unknown *stensor20000116_ptr = (unknown*)(buf + 24704);
  unknown *stensor20000115_ptr = (unknown*)(buf + 16512);
  unknown *stensor20000113_ptr = (unknown*)(buf + 128);
  unknown *stensor20000114_ptr = (unknown*)(buf + 8320);
  unknown *stensor20000145_ptr = (unknown*)(buf + 24752);
  unknown *stensor20000146_ptr = (unknown*)(buf + 32944);
  unknown *stensor20000129_ptr = (unknown*)(buf + 57520);
  unknown *stensor20000121_ptr = (unknown*)(buf + 41136);
  unknown *stensor20000138_ptr = (unknown*)(buf + 57536);
  unknown *stensor20000124_ptr = (unknown*)(buf + 49328);
  unknown *stensor20000125_ptr = (unknown*)(buf + 65728);
  unknown *stensor20000126_ptr = (unknown*)(buf + 69824);
  unknown *stensor20000140_ptr = (unknown*)(buf + 73920);
  unknown *stensor20000127_ptr = (unknown*)(buf + 41136);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000009 -> stensor 20000113
  const half_t *dtensor10000009_tile_ptr = dtensor10000009_ptr  + blockIdx.x*1*16384 + blockIdx.z*64*64;
  using DTensor10000009TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<16384>>>;
  using STensor20000113InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, DTensor10000009TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000010 -> stensor 20000114
  const half_t *dtensor10000010_tile_ptr = dtensor10000010_ptr  + blockIdx.x*1*65536 + blockIdx.y*64*1;
  using DTensor10000010TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<65536>>>;
  using STensor20000114InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, DTensor10000010TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000011 -> stensor 20000115
  const half_t *dtensor10000011_tile_ptr = dtensor10000011_ptr  + blockIdx.x*1*65536 + blockIdx.y*64*64;
  using DTensor10000011TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<65536>>>;
  using STensor20000115InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, DTensor10000011TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000012 -> stensor 20000116
  const half_t *dtensor10000012_tile_ptr = dtensor10000012_ptr  + blockIdx.x*1*4 + blockIdx.z*1*1;
  using DTensor10000012TileLayout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<4>>>;
  using STensor20000116InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>, DTensor10000012TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000013 -> stensor 20000117
  const half_t *dtensor10000013_tile_ptr = dtensor10000013_ptr  + blockIdx.x*1*16 + blockIdx.y*1*1;
  using DTensor10000013TileLayout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
  using STensor20000117InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>, DTensor10000013TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000014 -> stensor 20000118
  const half_t *dtensor10000014_tile_ptr = dtensor10000014_ptr  + blockIdx.x*1*16 + blockIdx.y*1*1;
  using DTensor10000014TileLayout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<16>>>;
  using STensor20000118InputAtom = tb::InputNonChunkedSyncCopy<half_t, Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>, DTensor10000014TileLayout, NUM_THREADS>;
  
  STensor20000113InputAtom::run(stensor20000113_ptr, dtensor10000009_tile_ptr, thread_idx);
  STensor20000114InputAtom::run(stensor20000114_ptr, dtensor10000010_tile_ptr, thread_idx);
  STensor20000115InputAtom::run(stensor20000115_ptr, dtensor10000011_tile_ptr, thread_idx);
  STensor20000116InputAtom::run(stensor20000116_ptr, dtensor10000012_tile_ptr, thread_idx);
  STensor20000117InputAtom::run(stensor20000117_ptr, dtensor10000013_tile_ptr, thread_idx);
  STensor20000118InputAtom::run(stensor20000118_ptr, dtensor10000014_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000145 -> dtensor 10000015
  half_t *dtensor10000015_tile_ptr = dtensor10000015_ptr  + blockIdx.x*1*262144 + blockIdx.y*64*1 + blockIdx.z*64*1024;
  using DTensor10000015TileLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20000145OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000015TileLayout, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
  // Copy for S->G: stensor 20000147 -> dtensor 10000016
  half_t *dtensor10000016_tile_ptr = dtensor10000016_ptr  + blockIdx.x*1*4096 + blockIdx.y*1*256 + blockIdx.z*64*1;
  using DTensor10000016TileLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20000147OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000016TileLayout, Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20000146_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 4096, NUM_THREADS>::run(stensor20000145_ptr, thread_idx);
  
  using Matmul20000127LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000127LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000127LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000127LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000127LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000127Kernel = tb::Matmul<cutlass::float_e4m3_t, SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<Int<2>, Int<4>, _1>>, true, false, Matmul20000127LayoutA, Matmul20000127LayoutB, Matmul20000127LayoutC, Matmul20000127LayoutAAligned, Matmul20000127LayoutBAligned,NUM_THREADS, 0, false>;
  
  using Matmul20000141LayoutA = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000141LayoutB = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000141LayoutC = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000141LayoutAAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));
  using Matmul20000141LayoutBAligned = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<64>, Int<1>>>{}));
  using Matmul20000141Kernel = tb::Matmul<cutlass::float_e4m3_t, SM80_16x8x16_F32BF16BF16F32_TN, Layout<Shape<Int<2>, Int<4>, _1>>, true, false, Matmul20000141LayoutA, Matmul20000141LayoutB, Matmul20000141LayoutC, Matmul20000141LayoutAAligned, Matmul20000141LayoutBAligned,NUM_THREADS, 0, false>;
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_div_op
      using In0Layout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueClamp<half_t, tb::EpilogueStore<half_t>>>>;
      const float scalars[] = {0.000000f, 0.000000f, 0.0f};
      Kernel::run(stensor20000121_ptr, stensor20000113_ptr, stensor20000116_ptr, thread_idx, scalars);
    }
    {
      // OP type: tb_div_op
      using In0Layout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueClamp<half_t, tb::EpilogueStore<half_t>>>>;
      const float scalars[] = {0.000000f, 0.000000f, 0.0f};
      Kernel::run(stensor20000124_ptr, stensor20000114_ptr, stensor20000117_ptr, thread_idx, scalars);
    }
    {
      // OP type: tb_mul_op
      using In0Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::MUL, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueStore<half_t>>>;
      const float scalars[] = {0.000000f, 0.0f};
      Kernel::run(stensor20000129_ptr, stensor20000116_ptr, stensor20000117_ptr, thread_idx, scalars);
    }
    {
      // OP type: tb_div_op
      using In0Layout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueClamp<half_t, tb::EpilogueStore<half_t>>>>;
      const float scalars[] = {0.000000f, 0.000000f, 0.0f};
      Kernel::run(stensor20000138_ptr, stensor20000115_ptr, stensor20000118_ptr, thread_idx, scalars);
    }
    __syncthreads();
    {
      // OP type: tb_e4m3_cast_op
      using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using OutLayout = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{}));
      using Kernel = tb::TypeCastKernel<cutlass::float_e4m3_t, half_t,OutLayout, InLayout, NUM_THREADS>;
      Kernel::run(stensor20000125_ptr, stensor20000121_ptr, thread_idx);
    }
    {
      // OP type: tb_e4m3_cast_op
      using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using OutLayout = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{}));
      using Kernel = tb::TypeCastKernel<cutlass::float_e4m3_t, half_t,OutLayout, InLayout, NUM_THREADS>;
      Kernel::run(stensor20000126_ptr, stensor20000124_ptr, thread_idx);
    }
    {
      // OP type: tb_e4m3_cast_op
      using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using OutLayout = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{}));
      using Kernel = tb::TypeCastKernel<cutlass::float_e4m3_t, half_t,OutLayout, InLayout, NUM_THREADS>;
      Kernel::run(stensor20000140_ptr, stensor20000138_ptr, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000127Kernel::get_mma_rC(thread_idx);
      Matmul20000127Kernel::run(mma_rC, stensor20000125_ptr, stensor20000126_ptr, (char*)(buf+0), thread_idx);
      Matmul20000127Kernel::write_back_mma_rC(stensor20000127_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_mul_op
      using In0Layout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{}));
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::MUL, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueExp<half_t, tb::EpilogueStore<half_t>>>;
      const float scalars[] = {0.000000f, 0.0f};
      Kernel::run(stensor20000131_ptr, stensor20000127_ptr, stensor20000129_ptr, thread_idx, scalars);
    }
    __syncthreads();
    {
      // OP type: tb_amax_op
      using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using Kernel = tb::AmaxKernel<half_t,OutLayout, InLayout, NUM_THREADS>;
      Kernel::run(stensor20000132_ptr, stensor20000131_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, decltype(composition(Swizzle<5, 1, 5>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{})), Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>, NUM_THREADS>;
      Kernel::run(stensor20000146_ptr, stensor20000131_ptr, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_div_op
      using In0Layout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueClamp<half_t, tb::EpilogueStore<half_t>>>>;
      const float scalars[] = {0.000000f, 0.000000f, 0.0f};
      Kernel::run(stensor20000135_ptr, stensor20000131_ptr, stensor20000132_ptr, thread_idx, scalars);
    }
    {
      // OP type: tb_mul_op
      using In0Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::MUL, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueMulScalar<half_t, tb::EpilogueStore<half_t>>>;
      const float scalars[] = {0.000000f, 0.0f};
      Kernel::run(stensor20000143_ptr, stensor20000132_ptr, stensor20000118_ptr, thread_idx, scalars);
    }
    __syncthreads();
    {
      // OP type: tb_e4m3_cast_op
      using InLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using OutLayout = decltype(composition(Swizzle<3, 3, 4>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{}));
      using Kernel = tb::TypeCastKernel<cutlass::float_e4m3_t, half_t,OutLayout, InLayout, NUM_THREADS>;
      Kernel::run(stensor20000139_ptr, stensor20000135_ptr, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_matmul_op
      auto mma_rC = Matmul20000141Kernel::get_mma_rC(thread_idx);
      Matmul20000141Kernel::run(mma_rC, stensor20000139_ptr, stensor20000140_ptr, (char*)(buf+0), thread_idx);
      Matmul20000141Kernel::write_back_mma_rC(stensor20000141_ptr, mma_rC, thread_idx);
    }
    __syncthreads();
    {
      // OP type: tb_mul_op
      using In0Layout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>{}));
      using In1Layout = Layout<Shape<Int<1>, Int<1>, Int<1>>, Stride<Int<1>, Int<1>, Int<1>>>;
      using OutLayout = Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<1>, Int<64>, Int<4096>>>;
      using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::MUL, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStoreAccum<half_t>>;
      const float scalars[] = {0.0f};
      Kernel::run(stensor20000145_ptr, stensor20000141_ptr, stensor20000143_ptr, thread_idx, scalars);
    }
  }
  
  // The epilogue (kernels outside the loop)
  __syncthreads();
  {
    // OP type: tb_reduction_2_op
    using InLayout = decltype(composition(Swizzle<5, 1, 5>{}, Layout<Shape<Int<64>, Int<64>, Int<1>>, Stride<Int<64>, Int<1>, Int<4096>>>{}));
    using OutLayout = Layout<Shape<Int<64>, Int<1>, Int<1>>, Stride<Int<1>, Int<64>, Int<64>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000147_ptr, stensor20000146_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_output_op
    STensor20000145OutputAtom::run(dtensor10000015_tile_ptr, stensor20000145_ptr, thread_idx);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000147OutputAtom::run(dtensor10000016_tile_ptr, stensor20000147_ptr, thread_idx);
  }
}

__global__ void __launch_bounds__(128) custom_kernel_1(half_t* __restrict__ dtensor10000017_ptr, half_t const* __restrict__ dtensor10000015_ptr, half_t const* __restrict__ dtensor10000016_ptr) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  // STensors
  extern __shared__ char buf[];
  half_t *stensor20000161_ptr = (half_t*)(buf + 2208);
  half_t *stensor20000156_ptr = (half_t*)(buf + 32896);
  unknown *stensor20000155_ptr = (unknown*)(buf + 128);
  unknown *stensor20000157_ptr = (unknown*)(buf + 33408);
  unknown *stensor20000159_ptr = (unknown*)(buf + 66176);
  unknown *stensor20000158_ptr = (unknown*)(buf + 128);
  unknown *stensor20000160_ptr = (unknown*)(buf + 2176);
  *((uint128_t*)buf) = 0ul;
  
  // G->S copy atoms
  // Copy for G->S: dtensor 10000015 -> stensor 20000155
  const half_t *dtensor10000015_tile_ptr = dtensor10000015_ptr  + blockIdx.x*1*262144 + blockIdx.y*16*1024;
  using DTensor10000015TileLayout = Layout<Shape<Int<1024>, Int<16>, Int<1>>, Stride<Int<1>, Int<1024>, Int<262144>>>;
  using STensor20000155InputAtom = tb::InputChunkedSyncCopy<half_t, decltype(composition(Swizzle<3, 3, 7>{}, Layout<Shape<Int<1024>, Int<16>, Int<1>>, Stride<Int<1>, Int<1024>, Int<16384>>>{})), DTensor10000015TileLayout, NUM_THREADS>;
  // Copy for G->S: dtensor 10000016 -> stensor 20000156
  const half_t *dtensor10000016_tile_ptr = dtensor10000016_ptr  + blockIdx.x*1*4096 + blockIdx.y*16*1;
  using DTensor10000016TileLayout = Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<256>, Int<4096>>>;
  using STensor20000156InputAtom = tb::InputChunkedSyncCopy<half_t, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, DTensor10000016TileLayout, NUM_THREADS>;
  
  STensor20000155InputAtom::run(stensor20000155_ptr, dtensor10000015_tile_ptr, thread_idx);
  STensor20000156InputAtom::run(stensor20000156_ptr, dtensor10000016_tile_ptr, thread_idx);
  
  // S->G copy atoms
  // Copy for S->G: stensor 20000161 -> dtensor 10000017
  half_t *dtensor10000017_tile_ptr = dtensor10000017_ptr  + blockIdx.x*1*16384 + blockIdx.y*16*1;
  using DTensor10000017TileLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<256>, Int<16384>>>;
  using STensor20000161OutputAtom = tb::OutputChunkedSyncCopy<half_t, DTensor10000017TileLayout, Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>, NUM_THREADS>;
  
  tb::ClearAccumlatorKernel<half_t, 16384, NUM_THREADS>::run(stensor20000157_ptr, thread_idx);
  tb::ClearAccumlatorKernel<half_t, 256, NUM_THREADS>::run(stensor20000159_ptr, thread_idx);
  
  __syncthreads();
  
  // The main loop
  for (int for_idx = 0; for_idx < 1; for_idx++) {
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1>, Int<16>, Int<16384>>>, decltype(composition(Swizzle<3, 3, 7>{}, Layout<Shape<Int<16>, Int<1024>, Int<1>>, Stride<Int<1024>, Int<1>, Int<16384>>>{})), NUM_THREADS>;
      Kernel::run(stensor20000157_ptr, stensor20000155_ptr, thread_idx);
    }
    {
      // OP type: tb_forloop_accum_nored_op
      using Kernel = tb::ForloopAccumKernel<half_t, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>, NUM_THREADS>;
      Kernel::run(stensor20000159_ptr, stensor20000156_ptr, thread_idx);
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
    Kernel::run(stensor20000158_ptr, stensor20000157_ptr, thread_idx, scalars);
  }
  {
    // OP type: tb_reduction_2_op
    using InLayout = Layout<Shape<Int<16>, Int<16>, Int<1>>, Stride<Int<1>, Int<16>, Int<256>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using Kernel = tb::ReductionKernel<half_t, OutLayout, InLayout, 1, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000160_ptr, stensor20000159_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_div_op
    using In0Layout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using In1Layout = Layout<Shape<Int<16>, Int<1>, Int<1>>, Stride<Int<1>, Int<16>, Int<16>>>;
    using OutLayout = Layout<Shape<Int<16>, Int<64>, Int<1>>, Stride<Int<1>, Int<16>, Int<1024>>>;
    using Kernel = tb::ElementBinaryKernel<half_t, tb::ElementBinaryOpType::DIV, OutLayout, In0Layout, In1Layout, NUM_THREADS, tb::EpilogueStore<half_t>>;
    const float scalars[] = {0.0f};
    Kernel::run(stensor20000161_ptr, stensor20000158_ptr, stensor20000160_ptr, thread_idx, scalars);
  }
  __syncthreads();
  {
    // OP type: tb_output_op
    STensor20000161OutputAtom::run(dtensor10000017_tile_ptr, stensor20000161_ptr, thread_idx);
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
    half_t *dtensor10000015 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000016 = (half_t*)((char*)buf + 1048576);
    half_t *dtensor10000009 = (half_t*)input_tensors.at(0);
    half_t *dtensor10000010 = (half_t*)input_tensors.at(1);
    half_t *dtensor10000011 = (half_t*)input_tensors.at(2);
    half_t *dtensor10000012 = (half_t*)input_tensors.at(3);
    half_t *dtensor10000013 = (half_t*)input_tensors.at(4);
    half_t *dtensor10000014 = (half_t*)input_tensors.at(5);
    dim3 grid_dim(2, 16, 4);
    dim3 block_dim(256, 1, 1);
    size_t smem_size = 78016;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 78016);
    custom_kernel_0<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000015, dtensor10000016, dtensor10000009, dtensor10000010, dtensor10000011, dtensor10000012, dtensor10000013, dtensor10000014);
  }
  {
    // OP type: kn_customized_op
    half_t *dtensor10000017 = (half_t*)output_tensors.at(0);
    half_t *dtensor10000015 = (half_t*)((char*)buf + 0);
    half_t *dtensor10000016 = (half_t*)((char*)buf + 1048576);
    dim3 grid_dim(2, 16, 1);
    dim3 block_dim(128, 1, 1);
    size_t smem_size = 66688;
    
    // define tmas
    cudaFuncSetAttribute(custom_kernel_1, cudaFuncAttributeMaxDynamicSharedMemorySize, 66688);
    custom_kernel_1<<<grid_dim, block_dim, smem_size, stream>>>( dtensor10000017, dtensor10000015, dtensor10000016);
  }
  {
    // OP type: kn_output_op
  }
}