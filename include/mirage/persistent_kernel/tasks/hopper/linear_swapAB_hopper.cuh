/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

 #pragma once
 #include "../common.h"
 #include "../dmem_layout.cuh"
 #include "../element_binary.cuh"
 #include "../element_unary.cuh"
 #include "../reduction.cuh"
 #include "../smem_layout.cuh"
 #include "../utils.cuh"
 #include "smem_layout_tma.cuh"
 #include "tma.cuh"
 #include "utils.cuh"
 #include "wgmma.cuh"
 
 namespace kernel {
 
 using namespace tma;
 using bfloat16 = type::bfloat16_t;
 
 template <typename T,
           int BATCH_SIZE,
           int OUTPUT_SIZE,
           int REDUCTION_SIZE,
           int Kstages,
           typename TMA_A,
           typename TMA_B,
           typename TMA_RESIDUAL,
           typename TMA_OUT,
           int OUTPUT_STRIDE = OUTPUT_SIZE>
 __device__ __forceinline__ void
     linear_kernel_swapAB_hopper(const TMA_A &tma_a,
                                 const TMA_B &tma_b,
                                 const TMA_RESIDUAL &tma_residual,
                                 const TMA_OUT &tma_out,
                                 void *output_ptr) {
 
   constexpr int INPUT_TMA_TILE_SIZE  = 64;
   constexpr int WEIGHT_TMA_TILE_SIZE = 64;
   constexpr int OUTPUT_TMA_TILE_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;
 
   constexpr int TILE_SIZE =
       (REDUCTION_SIZE < TMA_A::SMEM_COL * TMA_A::SMEM_REPEAT_COL)
           ? REDUCTION_SIZE
           : (TMA_A::SMEM_COL * TMA_A::SMEM_REPEAT_COL);
 
   constexpr int THREADS_PER_WARPGROUP = 128;
   constexpr int CONSUMER_WARPGROUPS   = 1;
   constexpr int PRODUCER_WARPGROUPS   = 1;
   constexpr int NUM_WARPGROUPS        = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS;
 
   constexpr int TMA_TRANS_BYTES_A        = sizeof(T) * BATCH_SIZE * TILE_SIZE;
   constexpr int TMA_TRANS_BYTES_B        = sizeof(T) * TILE_SIZE * OUTPUT_SIZE;
   constexpr int TMA_TRANS_BYTES_RESIDUAL = sizeof(T) * BATCH_SIZE * OUTPUT_SIZE;
 
   constexpr int NUM_ITER_K = (REDUCTION_SIZE + TILE_SIZE - 1) / TILE_SIZE;
 
   extern __shared__ char __smem[];
   uintptr_t smem = (reinterpret_cast<uintptr_t>(__smem) + 1023) & ~uintptr_t(1023);
 
   // shared memory layout
   constexpr size_t SHARED_INPUT_BUFFER_OFFSET    = 0;
   constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET   =
       (SHARED_INPUT_BUFFER_OFFSET + TMA_TRANS_BYTES_A * Kstages + 1023) & ~size_t(1023);
   constexpr size_t SHARED_RESIDUAL_BUFFER_OFFSET =
       (SHARED_WEIGHT_BUFFER_OFFSET + TMA_TRANS_BYTES_B * Kstages + 1023) & ~size_t(1023);
   constexpr size_t SHARED_BARRIERS_OFFSET =
       (SHARED_RESIDUAL_BUFFER_OFFSET + TMA_TRANS_BYTES_RESIDUAL + 15) & ~size_t(15);
 
   T *shared_input     = reinterpret_cast<T *>(smem + SHARED_INPUT_BUFFER_OFFSET);
   T *shared_weight    = reinterpret_cast<T *>(smem + SHARED_WEIGHT_BUFFER_OFFSET);
   T *shared_residual  = reinterpret_cast<T *>(smem + SHARED_RESIDUAL_BUFFER_OFFSET);
 
   // smem swizzle
   using InputSmem = smem_tma<T, 3, 3, 3, BATCH_SIZE,    INPUT_TMA_TILE_SIZE,  TILE_SIZE / INPUT_TMA_TILE_SIZE>;
   using WeightSmem= smem_tma<T, 3, 3, 3, OUTPUT_SIZE,   WEIGHT_TMA_TILE_SIZE, TILE_SIZE / WEIGHT_TMA_TILE_SIZE>;
   using ResidualSmem = smem_tma<T, 0, 0, 0, BATCH_SIZE, OUTPUT_TMA_TILE_SIZE, OUTPUT_SIZE / OUTPUT_TMA_TILE_SIZE>;
 
   InputSmem    input_smem(shared_input);
   WeightSmem   input_weight_smem(shared_weight);
   ResidualSmem residual_smem(shared_residual);
 
   // descriptors (swap AB: A <- weight, B <- input)
   using A_DESC = wgmma::mma_descriptor<WeightSmem>;
   using B_DESC = wgmma::mma_descriptor<InputSmem>;
 
   // barriers
   Barrier *input_barrier    = reinterpret_cast<Barrier *>(smem + SHARED_BARRIERS_OFFSET);
   Barrier *weight_barrier   = input_barrier    + Kstages;
   Barrier *residual_barrier = weight_barrier   + Kstages;
   Barrier *compute_done     = residual_barrier + 1;
 
   if (threadIdx.x == 0) {
     for (int i = 0; i < Kstages; ++i) {
       initialize_barrier(input_barrier[i], 1);
       initialize_barrier(weight_barrier[i], 1);
       initialize_barrier(compute_done[i], 1);
     }
     initialize_barrier(residual_barrier[0], 1);
   }
   __syncthreads();
 
   const int warp_idx     = warp_id();
   const int idx_in_warp  = threadIdx.x & 31;
   const int warpgroup_id = warp_idx / WARPGROUP_WARPS;
 
   // data movement
   if (warpgroup_id == NUM_WARPGROUPS - 1) {
     if (lane_id() == 0 && warp_idx == (NUM_WARPGROUPS * WARPGROUP_WARPS - 4)) {
       // residual once
       set_barrier_transaction_bytes(residual_barrier[0], TMA_TRANS_BYTES_RESIDUAL);
       tma_residual.tma_cp_async(residual_barrier[0], residual_smem(0, 0), {0, 0});
 
       // prefetch
       const int prefetch = (Kstages < NUM_ITER_K) ? Kstages : NUM_ITER_K;
       for (int i = 0; i < prefetch; ++i) {
         int slot = i % Kstages;
 
         input_smem.set_ptr(shared_input + (slot * TMA_TRANS_BYTES_A / sizeof(T)));
         set_barrier_transaction_bytes(input_barrier[slot], TMA_TRANS_BYTES_A);
         tma_a.tma_cp_async(input_barrier[slot], input_smem(0, 0), {i * TILE_SIZE, 0});
 
         input_weight_smem.set_ptr(shared_weight + (slot * TMA_TRANS_BYTES_B / sizeof(T)));
         set_barrier_transaction_bytes(weight_barrier[slot], TMA_TRANS_BYTES_B);
         tma_b.tma_cp_async(weight_barrier[slot], input_weight_smem(0, 0), {i * TILE_SIZE, 0});
       }
 
       for (int i = prefetch; i < NUM_ITER_K; ++i) {
         int slot  = i % Kstages;
         int phase = (i / Kstages) & 1;
         wait(compute_done[slot], phase ^ 1);
 
         input_smem.set_ptr(shared_input + (slot * TMA_TRANS_BYTES_A / sizeof(T)));
         set_barrier_transaction_bytes(input_barrier[slot], TMA_TRANS_BYTES_A);
         tma_a.tma_cp_async(input_barrier[slot], input_smem(0, 0), {i * TILE_SIZE, 0});
 
         input_weight_smem.set_ptr(shared_weight + (slot * TMA_TRANS_BYTES_B / sizeof(T)));
         set_barrier_transaction_bytes(weight_barrier[slot], TMA_TRANS_BYTES_B);
         tma_b.tma_cp_async(weight_barrier[slot], input_weight_smem(0, 0), {i * TILE_SIZE, 0});
       }
     }
   } else {
     // compute
     float s_frag[OUTPUT_SIZE / 2];
 #pragma unroll
     for (int i = 0; i < OUTPUT_SIZE / 16; ++i) {
       clear_8_floats(s_frag + i * 8);
     }
 
     wait(residual_barrier[0], 0);
 
     for (int i = 0; i < NUM_ITER_K; ++i) {
       int slot  = i % Kstages;
       int phase = (i / Kstages) & 1;
 
       wait(input_barrier[slot],  phase);
       wait(weight_barrier[slot], phase);
 
       input_smem.set_ptr(shared_input + (slot * TMA_TRANS_BYTES_A / sizeof(T)));
       input_weight_smem.set_ptr(shared_weight + (slot * TMA_TRANS_BYTES_B / sizeof(T)));
 
       A_DESC a_desc(input_weight_smem(0, 0)); // A <- weight
       B_DESC b_desc(input_smem(0, 0));        // B <- input
 
       wgmma::warpgroup_arrive();
       wgmma::mma<bfloat16,
                  64,
                  OUTPUT_SIZE,
                  16,
                  WeightSmem,
                  InputSmem,
                  A_DESC,
                  B_DESC,
                  false,
                  false>(s_frag, a_desc, b_desc);
       wgmma::mma_commit_group();
       wgmma::mma_async_wait();
 
       if (idx_in_warp == 0 && (warp_idx % 4) == 0) {
         arrive(compute_done[slot], 1);
       }
     }
 
     // elementwise write-back with residual add
     T *g_out = reinterpret_cast<T *>(output_ptr);
 
 #pragma unroll 1
     for (uint32_t i = 0; i < (OUTPUT_SIZE / 4); ++i) {
       int row = (warp_idx % 4) * 16 + (i % 2) * 8 + (idx_in_warp / 4);
       int col = (i / 2) * 8 + ((idx_in_warp % 4) * 2);
 
       T r0 = residual_smem.at(row, col + 0);
       T r1 = residual_smem.at(row, col + 1);
 
       bfloat16 v0 = bfloat16(s_frag[i * 2 + 0]) + r0;
       bfloat16 v1 = bfloat16(s_frag[i * 2 + 1]) + r1;
 
       // normal layout: [BATCH_SIZE, OUTPUT_SIZE]
       g_out[row * OUTPUT_STRIDE + (col + 0)] = v0;
       g_out[row * OUTPUT_STRIDE + (col + 1)] = v1;
 
       // if you need transpose write-back:
       // g_out[(col + 0) * OUTPUT_STRIDE + row] = v0;
       // g_out[(col + 1) * OUTPUT_STRIDE + row] = v1;
     }
   }
 
   __syncthreads();
 }
 
 } // namespace kernel
 