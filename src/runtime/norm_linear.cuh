
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

namespace mirage {
    namespace runtime {

 template<typename T, uint16_t BATCH_SIZE, uint16_t HIDDEN_SIZE, NUM_THREADS>
 __device__ __forceinline__ void norm_linear_kernel(){
    constexpr int chunk_size = 128 / sizeof(T);
    constexpr int num_chunks = HIDDEN_SIZE / chunk_size;

    //using SM80_16x8x16_F16F16F16F16_TN
    constexpr int num_n = HIDDEN_SIZE/16;
    constexpr int num_m = BATCH_SIZE/16;
    constexpr int num_k = HIDDEN_SIZE/8;

    int warp_idx = warp_id();
    int lane_idx = lane_id();
    int idx_in_warp = lane_id() % 32;

    assert(num_m > 0 && num_n > 0 && num_K > 0);
  //input_tile [BATCH_SIZE, HIDDEN_SIZE] 
  //weight_tile [HIDDEN_SIZE, HIDDEN_SIZE]
  const __restrict__ T* d_input = ;
  const __restrict__ T *d_weight = ;
  T __restrict__ *d_output =;

  dmem_row<T, 1, 64, 3584> input_dmem(d_input);
  dmem_row<T, 64, 64, 7168> weight_dmem(d_weight);
  dmem_row<T, 1, 64, 1> output_dmem(d_output);

  extern __shared__ T smem[];

  //copy input
  T *shared_input = (T*)(buf + 256);
  T *shared_input_buffer = (T*)(buf + 384);
  //copy weight
  T *shared_weight = (T*)(buf + 8704);
  T *shared_weight_buffer = (T*)(buf + 512);
  //intermidiate
  T *mm_output = (T*)(buf + 256);
  T *element_unary_output = (T*)(buf + 128);
  T *reduction_output = (T*)(buf + 384);
  //out
  T *shared_output = (T*)(buf + 128);
  
  //define the swizzle mode
  smem_row<T, 1, 1, 1, 1, 16, 64> input_smem(shared_input);
  smem_row<T, 1, 1, 1, 1, 16, 64> input_smem_buffer(shared_input_buffer);

  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem(shared_weight);
  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem_buffer(shared_weight_buffer);

  smem_row<T, 1, 1, 1, 1, 16, 64> element_unary_smem(element_unary_output);

  smem_row<T, 1, 1, 1, 1, 16, 64> mm_output_smem(mm_output);
  smem_row<T, 1, 1, 1, 1, 16, 1> reduction_output_smem(reduction_output);

  smem_row<T, 1, 1, 1, 1, 16, 64> output_smem(shared_output);

  //load input
  #pragma unroll
  for(int i = threadIdx.x; i < (BATCH_SIZE * HIDDEN_SIZE / chunk_size); i+=NUM_THREADS){
    //offset
    size_t row = i / num_chunks;
    size_t col = i % num_chunks;
    load_smem(input_smem[row, col], input_dmem[row, col]);
  }
  
  //load weight
  #pragma unroll
  for(int i = threadIdx.x; i < (HIDDEN_SIZE * HIDDEN_SIZE / chunk_size); i+=NUM_THREADS){
    size_t row = i / num_chunks;
    size_t col = i % num_chunks;
    load_smem(input_smem[row, col], input_dmem[row, col]);
  }
  cp_async_fence();

  //accumulator
  float s_frag[num_m][num_n][8];

  for(int for_idx = 0; for_idx < params.forloop_range; for_idx++){
    //copy
    if(for_idx + 1 != params.forloop_range){
        dmem_row<T, 1, 64, 3584> input_dmem_buffer(d_input + for_idx * params.forloop_stride);
        dmem_row<T, 64, 64, 7168> weight_dmem_buffer(d_weight + for_idx * params.forloop_stride);
        #pragma unroll
        for(int i = threadIdx.x; i < (BATCH_SIZE * HIDDEN_SIZE / chunk_size); i+=NUM_THREADS){
          //offset
          size_t row = i / num_chunks;
          size_t col = i % num_chunks;
          load_smem(input_smem_buffer[row, col], input_dmem_buffer[row, col]);
        }
        //load weight
        #pragma unroll
        for(int i = threadIdx.x; i < (HIDDEN_SIZE * HIDDEN_SIZE / chunk_size); i+=NUM_THREADS){
          size_t row = i / num_chunks;
          size_t col = i % num_chunks;
          load_smem(input_weight_smem_buffer[row, col], weight_dmem_buffer[row, col]);
        }
        cp_async_fence();
        cp_async_wait<1>();
        //SWAP the double buffer
        if((i & 1) == 0){
          input_smem(shared_input_buffer);
          input_smem_buffer(shared_input);
          input_weight_smem(shared_weight_buffer);
          input_weight_smem_buffer(shared_weight);
        }else{
          input_smem(shared_input);
          input_smem_buffer(shared_input_buffer);
          input_weight_smem(shared_weight);
          input_weight_smem_buffer(shared_weight_buffer);
        }
       
    }    
    uint32_t a_frag[4], b_frag[4];
    //N loop
    //16X64X64 -> 16X8X16

    //A = 16 * 64
    //B = 64 X 64

    //atom 16X8X16 -> tileMMA 16X32X16, m = 16X16, n = 16 X 32, k = 16
    //loop N = 1, K = 4, M = 1
    for(uint32_t n = 0; n < num_n / 4; n++){
      //M loop
      for(uint32_t m = 0; m < num_m; m++){
        //K loop
        for(uint32_t k = 0; k < num_k; k++){
          //16X16X16
           //load A matrix
           //https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float:~:
           // text=%2C..%2C3%7D-,9.7.14.5.8.%20Matrix%20Fragments%20for%20mma.m16n8k16%20with%20floating%20point%20type,-%EF%83%81
           uint32_t m_offset = k * (16 * 16)+ (idx_in_warp % 32) * 8;
           //load B matrix, 
           uint32_t n_offset = warp_id * (64*8) + k * (16*8) + (idx_in_warp % 32) * 8;
           ldsm(input_smem(m_offset), a_frag);
           ldsm(input_weight_smem(n_offset), b_frag);
           mma_m16n16k16_bf16bf16bf32(s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
        }
      }
    }
    //MNK = 1, 64, 64
    //matmul
    //sqrt, mulscalar
    perform_element_unary_chain_kernel<true,
                ElementUnaryOpType::SQRT,
                ElementUnaryOpType::MULSCALAR,
              >(input_smem, element_unary_smem, 0.0f, 1.0f);
  }

  //write back to smem
  for(uint32_t n = 0; n < num_n / 4; n++){
    for(uint32_t m = 0; m < num_m; m++){
      for(uint32_t i = 0; i < 8; i++){
        mm_output_smem() = s_frag[m][n][i];
      }
  }
}
  reduction_sum_col(element_unary_smem, reduction_output_smem);

  //div
  div_col(output_smem, mm_output_smem, reduction_output_smem);

  //write back to device
  store_smem(output_smem, output_dmem);
 }


}
}