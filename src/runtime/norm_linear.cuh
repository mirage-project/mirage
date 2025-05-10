
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

 #pragma once

#include <cuda_bf16.h>
#include "utils.cuh"
#include "dmem_layout.cuh"
#include "smem_layout.cuh"
#include "copy_sm80.cuh"
#include "element_unary.cuh"
#include "element_binary.cuh"
#include "mma.cuh"
#include "reduction.cuh"

namespace mirage {
    namespace runtime {

  //kernel for [16, 64] and any BATCH_SIZE < 16 [x, 64]
 template<typename T, int BATCH_SIZE, int HIDDEN_SIZE, int NUM_THREADS>
 __device__ __forceinline__ void norm_linear_kernel(void const * input_ptr, void const * weight_ptr, void * output_ptr){
    constexpr int chunk_size = 16 / sizeof(T);
   
    constexpr int num_chunks = HIDDEN_SIZE / chunk_size;

    //using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
    constexpr int num_n = HIDDEN_SIZE/16;
    constexpr int num_m = BATCH_SIZE/16;
    constexpr int num_k = HIDDEN_SIZE/16;

    int warp_idx = warp_id();
    int lane_idx = lane_id();
    int idx_in_warp = lane_id() % 32;

    assert(num_m > 0 && num_n > 0 && num_k > 0);
  //input_tile [BATCH_SIZE, HIDDEN_SIZE] 
  //weight_tile [HIDDEN_SIZE, HIDDEN_SIZE]
  const __restrict__ T* d_input = static_cast<const T*>(input_ptr);

  const __restrict__ T *d_weight = static_cast<const T*>(weight_ptr) +  blockIdx.x*64*1;
  T __restrict__ *d_output = static_cast<T*>(output_ptr) + blockIdx.x*64*1;

  dmem_row_const<T, 16, 64, 4096> input_dmem(d_input);
  dmem_row_const<T, 64, 64, 64> weight_dmem(d_weight);
  dmem_row<T, 16, 64, 64> output_dmem(d_output);

  extern __shared__ T smem[];

  //copy input
  T *shared_input = (T*)(smem + 2176);
  T *shared_input_buffer = (T*)(smem + 4224);
  //copy weight
  T *shared_weight = (T*)(smem + 6272);
  T *shared_weight_buffer = (T*)(smem + 14464);
  //intermidiate
  T *mm_output = (T*)(smem + 2176);
  T *element_unary_output = (T*)(smem + 128);
  T *reduction_output = (T*)(smem + 4224);
  //out
  T *shared_output = (T*)(smem + 128);
  
  //define the swizzle mode
  smem_row<T, 3, 3, 3, 16, 64, 64> input_smem(shared_input);
  smem_row<T, 3, 3, 3, 16, 64, 64> input_smem_buffer(shared_input_buffer);

  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem(shared_weight);
  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem_buffer(shared_weight_buffer);

  smem_row<T, 1, 1, 1, 16, 64, 64> element_unary_smem(element_unary_output);

  smem_row<T, 3, 3, 3, 16, 64, 64> mm_output_smem(mm_output);
  smem_row<T, 1, 1, 1, 16, 1, 16> reduction_output_smem(reduction_output);

  smem_row<T, 1, 1, 1, 16, 64, 64> output_smem(shared_output);


  //load input
  #pragma unroll
  for(int i = threadIdx.x; i < (BATCH_SIZE * num_chunks); i+=NUM_THREADS){
    //offset
    int row = i / num_chunks;
    int col = (i % num_chunks) * chunk_size;
    load_smem(input_smem(row, col), input_dmem(row, col));
  }
  
  //load weight
  #pragma unroll
  for(int i = threadIdx.x; i < (HIDDEN_SIZE * HIDDEN_SIZE / chunk_size); i+=NUM_THREADS){
    int row = i / (HIDDEN_SIZE / chunk_size);
    int col = (i % num_chunks) * chunk_size;
    // printf("load 00 ,%d, %d, thread %d\n", row, col, threadIdx.x);
    load_smem(input_weight_smem(row, col), weight_dmem(row, col));
  }
  cp_async_fence();

  // if(threadIdx.x == 0){
  //       printf("s_frag[m][n] %f, %f, %f, %f\n", __bfloat162float(input_weight_smem.at(0)), __bfloat162float(input_smem.at(0)), __bfloat162float(input_dmem.at(0)), __bfloat162float(weight_dmem.at(0)));
  // }

//  accumulator
  float s_frag[num_m][num_n][8];

  for(int for_idx = 0; for_idx < 64; for_idx++){
    //copy
    if(for_idx + 1 != 64){
        dmem_row_const<T, 16, 64, 4096> input_dmem_buffer(d_input + 64*(for_idx+1));
        dmem_row_const<T, 64, 64, 64> weight_dmem_buffer(d_weight + 4096*(for_idx+1));
        #pragma unroll
        for(int i = threadIdx.x; i < (BATCH_SIZE * num_chunks); i+=NUM_THREADS){
          //offset
          int row = i / num_chunks;
          int col = (i % num_chunks) * chunk_size;
          load_smem(input_smem_buffer(row, col), input_dmem_buffer(row, col));
        }
        //load weight
        #pragma unroll
        for(int i = threadIdx.x; i < (HIDDEN_SIZE * HIDDEN_SIZE / chunk_size); i+=NUM_THREADS){
          int row = i / (HIDDEN_SIZE / chunk_size);
          int col = (i % num_chunks) * chunk_size;
          load_smem(input_weight_smem_buffer(row, col), weight_dmem_buffer(row, col));
        }
        cp_async_fence();
        cp_async_wait<1>();
        //SWAP the double buffer
        if((for_idx & 1) == 0){
          input_smem.set_ptr(shared_input);
          input_smem_buffer.set_ptr(shared_input_buffer);
          input_weight_smem.set_ptr(shared_weight);
          input_weight_smem_buffer.set_ptr(shared_weight_buffer);
        }else{
          input_smem.set_ptr(shared_input_buffer);
          input_smem_buffer.set_ptr(shared_input);
          input_weight_smem.set_ptr(shared_weight_buffer);
          input_weight_smem_buffer.set_ptr(shared_weight);
         
        }
        __syncthreads();
       
    }    
    uint32_t a_frag[4], b_frag[4];
    //N loop
    //16X64X64 -> 16X8X16

    //A = 16 * 64
    //B = 64 X 64

    // if(threadIdx.x == 0){
    //   printf("mm inputx cddd%f\n",
    //      __bfloat162float(input_weight_smem.at(0)));
    //  }

    //atom 16X8X16 -> tileMMA 16X32X16, m = 16X16, n = 16 X 32, k = 16
    //loop N = 1, K = 4, M = 1
    for(uint32_t n = 0; n < num_n / 4; n++){
      //M loop
      for(uint32_t m = 0; m < num_m; m++){
        //K loop
        for(uint32_t k = 0; k < num_k; k++){
          //16X16X16
           //load A matrix
          //  https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float:~:text=%2C..%2C3%7D-,9.7.14.5.8.%20Matrix%20Fragments%20for%20mma.m16n8k16%20with%20floating%20point%20type,-%EF%83%81
           uint32_t m_offset = k * (16 * 16) + (idx_in_warp % 16 * 16 + idx_in_warp / 16 * 8);
           //load B matrix, 
           uint32_t n_offset = warp_idx * (64*8) + k * (16*8) + (idx_in_warp % 32) * 8;
           ldsm(input_smem[m_offset], &a_frag[0]);
           ldsm(input_weight_smem[n_offset], &b_frag[0]);

          //  if(m == 0 && n == 0 && threadIdx.x == 0){
          //   printf("mm inputx %d, %d, %f, %f\n", m_offset, 
          //     n_offset, __bfloat162float(input_smem.at(m_offset)),
          //      __bfloat162float(input_weight_smem.at(n_offset)));
          //  }
           mma_m16n16k16_bf16bf16bf32(s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
           __syncthreads();
          //  if(m == 0 && n == 0 && threadIdx.x == 127){
          //   printf("s_frag[m][n] %f\n", s_frag[0][0][7]);
          //  }
        }
      }
    }
   
    // sqrt, mulscalar
    const float scalars[] = {0.0f, 0.000244f};
    // perform_element_unary_chain_kernel();
    perform_element_unary_chain_kernel<true,
                decltype(element_unary_smem), 
                decltype(input_smem),
                ElementUnaryOpType::SQUARE,
                ElementUnaryOpType::MULSCALAR>(element_unary_smem, input_smem,  scalars);

    // if(threadIdx.x == 0){
    //   printf("element_unary_smem %f, %f\n", __bfloat162float(element_unary_smem.at(0)), __bfloat162float(input_smem.at(0)));
    // }
  }
  __syncthreads();
  //reg write back to smem
  for(uint32_t n = 0; n < num_n / 4; n++){
    for(uint32_t m = 0; m < num_m; m++){
      // warp_idx
      // int row = threadIdx;
      // int col = ;
      //https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float:~:text=The%20layout%20of%20the%20fragments%20held%20by%20different%20threads%20is%20shown%20in%20Figure%2083.
      //the 8 values in each thread is 
      for(uint32_t i = 0; i < 4; i++){
        // if(i == 0){
        //   int row = lane_idx / 4;
        //   int col = lane_idx % 4 + 16 * warp_idx;
        // }else if(i == 1){
        //   int row = lane_idx / 4 + 8;
        //   int col = lane_idx % 4 + 16 * warp_idx;
        // }else if(i == 2){
        //   int row = lane_idx / 4;
        //   int col = lane_idx % 4 + 16 * warp_idx + 8;
        // }else{
        //   int row = lane_idx / 4 + 8;
        //   int col = lane_idx % 4 + 16 * warp_idx + 8;
        // }
        int row = idx_in_warp / 4 + 8 * (i % 2);
        int col = (idx_in_warp % 4) * 2 + 16 * warp_idx + 8 * (i / 2);
        
        mm_output_smem.at(row, col) = __float2bfloat16(s_frag[m][n][i*2]);
        mm_output_smem.at(row, col+1) = __float2bfloat16(s_frag[m][n][i*2+1]);
        // printf("mm output A%f, B%f, rol%d, col %d, value %f, %f\n", 
        //   s_frag[m][n][i*2], s_frag[m][n][i*2+1], row, col, __bfloat162float(mm_output_smem.at(row, col)), __bfloat162float(mm_output_smem.at(row, col+1)));
      }
  }
}
__syncthreads();
  reduction_sum_col<T>(reduction_output_smem, element_unary_smem);
  __syncthreads();
  
  const float scalars[] = {0.0f};
  perform_element_unary_chain_kernel<false,
  decltype(reduction_output_smem), 
  decltype(reduction_output_smem),
  ElementUnaryOpType::SQRT>(reduction_output_smem, reduction_output_smem,  scalars);
  __syncthreads();

  // //div
  div_col(output_smem, mm_output_smem, reduction_output_smem);
  __syncthreads();

  

 
 // write back to device, 128 bytes as a atom
 #pragma unroll
 //todo, use uint128 to to chunk cooy, 16X64 -> 16X8
  for(int i = threadIdx.x; i < (BATCH_SIZE * HIDDEN_SIZE); i+=NUM_THREADS){
    //offset
    int row = i / HIDDEN_SIZE;
    int col = (i % HIDDEN_SIZE);
    // printf("output %f, %d, row is %d, col is %d\n",  
    //   __bfloat162float(output_smem.at(row, col)), row, col, threadIdx.x);
    output_dmem.at(row, col) = output_smem.at(row, col);
  }
  
 }


}
}