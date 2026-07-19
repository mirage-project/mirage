/* Copyright 2026 CMU
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
#include "tasks/common/common_header.cuh"

// Inkling depthwise short convolution (InklingShortConvolution).
//
// Semantics (matches HF reference exactly, decode / short-extend path):
//   x            : [SEQ, HIDDEN] (bf16) input activations (SEQ tokens, in order)
//   weight       : [HIDDEN, K] (fp32) depthwise conv taps, w[.,0] = oldest tap
//   conv_state   : [K-1, HIDDEN] (fp32) previous K-1 token activations,
//                  row 0 = oldest; updated IN PLACE to the last K-1 tokens
//   out          : [SEQ, HIDDEN] (bf16)
//
//   out[t, c] = x[t, c]                      // residual (added inside module)
//             + sum_{i=0..K-1} w[c, i] * win[i]
//   where win = [state..., x[<=t]] sliding causal window, all math in fp32.
//
// The channel dimension is embarrassingly parallel: grid.x partitions HIDDEN.
// Per-task pointer offsets are applied by the runtime via imap (dim 1 for
// x/state/out, dim 0 for weight); strides below are the FULL row widths.
//
// conv_state must be zero-initialized at sequence start (conv padding).

namespace kernel {

template <typename T,
          int HIDDEN,      // channels handled by THIS task
          int K,           // conv kernel size (4)
          int SEQ,         // tokens processed this step (1 for decode)
          int X_STRIDE,    // full row width of x (elements)
          int OUT_STRIDE,  // full row width of out (elements)
          int STATE_STRIDE // full row width of conv_state (elements)
          >
__device__ __forceinline__ void
    inkling_sconv_task_impl(void const *x_ptr,
                            void const *weight_ptr,
                            void const *state_ptr, // updated in place
                            void *out_ptr) {
  T const *__restrict__ x = static_cast<T const *>(x_ptr);
  float const *__restrict__ w = static_cast<float const *>(weight_ptr);
  float *__restrict__ state =
      const_cast<float *>(static_cast<float const *>(state_ptr));
  T *__restrict__ out = static_cast<T *>(out_ptr);

  for (int c = threadIdx.x; c < HIDDEN; c += blockDim.x) {
    // sliding window: s[0] oldest ... s[K-2] newest previous token
    float s[K - 1];
#pragma unroll
    for (int i = 0; i < K - 1; i++) {
      s[i] = state[i * STATE_STRIDE + c];
    }
    float taps[K];
#pragma unroll
    for (int i = 0; i < K; i++) {
      taps[i] = w[c * K + i];
    }
#pragma unroll
    for (int t = 0; t < SEQ; t++) {
      float xv = float(x[t * X_STRIDE + c]);
      float acc = xv + taps[K - 1] * xv; // residual + newest tap
#pragma unroll
      for (int i = 0; i < K - 1; i++) {
        acc += taps[i] * s[i];
      }
      out[t * OUT_STRIDE + c] = T(acc);
      // shift window
#pragma unroll
      for (int i = 0; i < K - 2; i++) {
        s[i] = s[i + 1];
      }
      s[K - 2] = xv;
    }
#pragma unroll
    for (int i = 0; i < K - 1; i++) {
      state[i * STATE_STRIDE + c] = s[i];
    }
  }
}

} // namespace kernel
