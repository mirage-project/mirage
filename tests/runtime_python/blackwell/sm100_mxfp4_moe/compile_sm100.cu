// Compile-only instantiation of the SM100 MXFP4 expert GEMM.
// tcgen05 will not run on Hopper; this cubin is the "does it build for B200" gate.
//
//   module load StdEnv/2023 gcc/12.3 cuda/13.2
//   nvcc -O2 -std=c++17 -arch=sm_100a --expt-relaxed-constexpr -use_fast_math \
//     -DMPK_TARGET_CC=100 -DMIRAGE_GRACE_BLACKWELL -DMPK_ENABLE_TMA \
//     -DMODE_OFFLINE -DMIRAGE_BACKEND_USE_CUDA \
//     -I <repo>/include -I <repo>/include/mirage/persistent_kernel \
//     -I <repo>/deps/cutlass/include \
//     -I <repo>/deps/cutlass/tools/util/include \
//     compile_sm100.cu -cubin -o /tmp/moe_mxfp4_sm100.cubin

#include "mirage/persistent_kernel/tasks/blackwell/moe_mxfp4_sm100.cuh"

// 120B W13 slice: batch 8, N-slice 128, full N 5760, K 2880, 128 experts, top-4.
// 120B W2 slice: N-slice 64 (2880 is not a multiple of 128).

__global__ void instantiate_w13(cute::bfloat16_t const *input,
                                uint8_t const *blocks,
                                uint8_t const *scales,
                                int32_t const *routing,
                                int32_t const *mask,
                                cute::bfloat16_t const *bias,
                                cute::bfloat16_t *output) {
  kernel::moe_mxfp4_sm100_task_impl<8, 128, 5760, 2880, 128, 4, 8, 5760, true, false>(
      input, blocks, scales, routing, mask, bias, output, 0);
}

__global__ void instantiate_w2(cute::bfloat16_t const *input,
                               uint8_t const *blocks,
                               uint8_t const *scales,
                               int32_t const *routing,
                               int32_t const *mask,
                               cute::bfloat16_t const *bias,
                               cute::bfloat16_t *output) {
  kernel::moe_mxfp4_sm100_task_impl<8, 64, 2880, 2880, 128, 4, 8, 2880, false, false>(
      input, blocks, scales, routing, mask, bias, output, 0);
}
