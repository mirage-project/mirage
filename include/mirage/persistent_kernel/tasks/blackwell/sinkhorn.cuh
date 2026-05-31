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
#include "tasks/common/common_header.cuh"

// mHC Sinkhorn-Knopp normalization, hardcoded to a 4x4 matrix per token.
//
// One thread = one 4x4 matrix. Fully unrolled, register-only, no shared
// memory and no __syncthreads(). Caller launches a 1D grid that strides over
// `valid_tokens` in chunks of blockDim.x.
//
// Sequence (matches the PyTorch reference in profile/utils.py):
//   1. per-row softmax (max-subtract for stability) + eps
//   2. `repeat` col-normalizations interleaved with `repeat - 1`
//      row-normalizations between them; final op is col-norm so columns
//      sum to 1.

namespace kernel {

template <int INPUT_TOKEN_STRIDE, int OUTPUT_TOKEN_STRIDE>
__device__ __forceinline__ void
    sinkhorn_task_impl(void const *__restrict__ comb_res_mix_ptr,
                       void *__restrict__ comb_res_mix_out_ptr,
                       int valid_tokens,
                       int repeat,
                       float eps) {
  float const *__restrict__ in = static_cast<float const *>(comb_res_mix_ptr);
  float *__restrict__ out = static_cast<float *>(comb_res_mix_out_ptr);

  int const tid = blockIdx.x * blockDim.x + threadIdx.x;
  int const stride = gridDim.x * blockDim.x;

  for (int token = tid; token < valid_tokens; token += stride) {
    int const in_base = token * INPUT_TOKEN_STRIDE;
    int const out_base = token * OUTPUT_TOKEN_STRIDE;

    float4 const r0v = reinterpret_cast<float4 const *>(in + in_base)[0];
    float4 const r1v = reinterpret_cast<float4 const *>(in + in_base)[1];
    float4 const r2v = reinterpret_cast<float4 const *>(in + in_base)[2];
    float4 const r3v = reinterpret_cast<float4 const *>(in + in_base)[3];

    float m00 = r0v.x, m01 = r0v.y, m02 = r0v.z, m03 = r0v.w;
    float m10 = r1v.x, m11 = r1v.y, m12 = r1v.z, m13 = r1v.w;
    float m20 = r2v.x, m21 = r2v.y, m22 = r2v.z, m23 = r2v.w;
    float m30 = r3v.x, m31 = r3v.y, m32 = r3v.z, m33 = r3v.w;

    // Step 1: per-row softmax + eps.
    float const rmax0 = fmaxf(fmaxf(m00, m01), fmaxf(m02, m03));
    float const rmax1 = fmaxf(fmaxf(m10, m11), fmaxf(m12, m13));
    float const rmax2 = fmaxf(fmaxf(m20, m21), fmaxf(m22, m23));
    float const rmax3 = fmaxf(fmaxf(m30, m31), fmaxf(m32, m33));

    m00 = __expf(m00 - rmax0);
    m01 = __expf(m01 - rmax0);
    m02 = __expf(m02 - rmax0);
    m03 = __expf(m03 - rmax0);
    m10 = __expf(m10 - rmax1);
    m11 = __expf(m11 - rmax1);
    m12 = __expf(m12 - rmax1);
    m13 = __expf(m13 - rmax1);
    m20 = __expf(m20 - rmax2);
    m21 = __expf(m21 - rmax2);
    m22 = __expf(m22 - rmax2);
    m23 = __expf(m23 - rmax2);
    m30 = __expf(m30 - rmax3);
    m31 = __expf(m31 - rmax3);
    m32 = __expf(m32 - rmax3);
    m33 = __expf(m33 - rmax3);

    float const rs0 = m00 + m01 + m02 + m03;
    float const rs1 = m10 + m11 + m12 + m13;
    float const rs2 = m20 + m21 + m22 + m23;
    float const rs3 = m30 + m31 + m32 + m33;
    float const ri0 = __frcp_rn(rs0);
    float const ri1 = __frcp_rn(rs1);
    float const ri2 = __frcp_rn(rs2);
    float const ri3 = __frcp_rn(rs3);
    m00 = m00 * ri0 + eps;
    m01 = m01 * ri0 + eps;
    m02 = m02 * ri0 + eps;
    m03 = m03 * ri0 + eps;
    m10 = m10 * ri1 + eps;
    m11 = m11 * ri1 + eps;
    m12 = m12 * ri1 + eps;
    m13 = m13 * ri1 + eps;
    m20 = m20 * ri2 + eps;
    m21 = m21 * ri2 + eps;
    m22 = m22 * ri2 + eps;
    m23 = m23 * ri2 + eps;
    m30 = m30 * ri3 + eps;
    m31 = m31 * ri3 + eps;
    m32 = m32 * ri3 + eps;
    m33 = m33 * ri3 + eps;

    // Step 2: alternating col/row normalization, ending on col-norm.
    int const steps = repeat > 0 ? repeat : 1;
#pragma unroll 1
    for (int it = 0; it < steps; ++it) {
      float const cs0 = m00 + m10 + m20 + m30 + eps;
      float const cs1 = m01 + m11 + m21 + m31 + eps;
      float const cs2 = m02 + m12 + m22 + m32 + eps;
      float const cs3 = m03 + m13 + m23 + m33 + eps;
      float const ci0 = __frcp_rn(cs0);
      float const ci1 = __frcp_rn(cs1);
      float const ci2 = __frcp_rn(cs2);
      float const ci3 = __frcp_rn(cs3);
      m00 *= ci0;
      m10 *= ci0;
      m20 *= ci0;
      m30 *= ci0;
      m01 *= ci1;
      m11 *= ci1;
      m21 *= ci1;
      m31 *= ci1;
      m02 *= ci2;
      m12 *= ci2;
      m22 *= ci2;
      m32 *= ci2;
      m03 *= ci3;
      m13 *= ci3;
      m23 *= ci3;
      m33 *= ci3;

      if (it == steps - 1) {
        break;
      }

      float const rs0i = m00 + m01 + m02 + m03 + eps;
      float const rs1i = m10 + m11 + m12 + m13 + eps;
      float const rs2i = m20 + m21 + m22 + m23 + eps;
      float const rs3i = m30 + m31 + m32 + m33 + eps;
      float const ri0i = __frcp_rn(rs0i);
      float const ri1i = __frcp_rn(rs1i);
      float const ri2i = __frcp_rn(rs2i);
      float const ri3i = __frcp_rn(rs3i);
      m00 *= ri0i;
      m01 *= ri0i;
      m02 *= ri0i;
      m03 *= ri0i;
      m10 *= ri1i;
      m11 *= ri1i;
      m12 *= ri1i;
      m13 *= ri1i;
      m20 *= ri2i;
      m21 *= ri2i;
      m22 *= ri2i;
      m23 *= ri2i;
      m30 *= ri3i;
      m31 *= ri3i;
      m32 *= ri3i;
      m33 *= ri3i;
    }

    reinterpret_cast<float4 *>(out + out_base)[0] =
        make_float4(m00, m01, m02, m03);
    reinterpret_cast<float4 *>(out + out_base)[1] =
        make_float4(m10, m11, m12, m13);
    reinterpret_cast<float4 *>(out + out_base)[2] =
        make_float4(m20, m21, m22, m23);
    reinterpret_cast<float4 *>(out + out_base)[3] =
        make_float4(m30, m31, m32, m33);
  }
}

} // namespace kernel
