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

// ================= TopK Sigmoid (Group-Aware) for Hopper =====================
//
// DeepSeek-V3 style routing, same math and output contract as
// topk_sigmoid_task_impl (blackwell/topk_sigmoid_sm100.cuh):
//   1. scores = sigmoid(logits); biased = scores + e_score_correction_bias
//   2. group score = sum of the top-2 biased scores in each group
//   3. keep the TOPK_GROUP best groups, mask the rest
//   4. pick TOPK_EXPERTS experts by biased score from the kept groups
//   5. weight = unbiased score, normalized over the picks and scaled
// Outputs: topk_weights [num_rows, TOPK_EXPERTS] (fp32), routing_indices
// [NUM_EXPERTS, num_rows] (1-based slot, 0 = unused) and active_expert_ids
// [NUM_EXPERTS + 1] (compacted ids, count in the last entry). The logits are
// zeroed in place afterwards so a split-K gate linear can accumulate into
// them again.
//
// Layout: one warp per token row, rows strided over the 8 worker warps, so
// any num_rows works. Lane l owns the VPT = NUM_EXPERTS / 32 contiguous
// experts [l*VPT, l*VPT+VPT). Each expert-selection round is one redux.sync on
// an order-preserving integer key plus a ballot; ties resolve to the lowest
// expert id. active_expert_ids is filled in ascending expert order and slots
// past the count are set to -1.

namespace kernel {

namespace topk_sigmoid_hopper_detail {

// Monotonic map float -> uint32: a < b  <=>  key(a) < key(b) (no NaNs).
// Key 0 is never produced by a finite value or -inf, so it can mean "empty".
__device__ __forceinline__ uint32_t float_to_key(float f) {
  uint32_t const bits = __float_as_uint(f);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <int BYTES>
struct VecOf;
template <>
struct VecOf<16> {
  using type = uint4;
};
template <>
struct VecOf<8> {
  using type = uint2;
};
template <>
struct VecOf<4> {
  using type = uint32_t;
};
template <>
struct VecOf<2> {
  using type = uint16_t;
};

} // namespace topk_sigmoid_hopper_detail

template <typename T,
          int NUM_EXPERTS,
          int NUM_GROUPS,
          int TOPK_GROUP,
          int TOPK_EXPERTS>
__device__ __forceinline__ void
    topk_sigmoid_hopper_task_impl(void *__restrict__ input_ptr,
                                  void const *__restrict__ bias_ptr,
                                  void *__restrict__ output_ptr,
                                  int const num_rows,
                                  void *__restrict__ routing_indices_ptr,
                                  void *__restrict__ active_expert_ids_ptr,
                                  float const routed_scaling_factor) {
  using namespace topk_sigmoid_hopper_detail;
  constexpr int WARP = NUM_THREADS_PER_WARP;
  constexpr int NUM_WORKER_WARPS = WORKER_NUM_THREADS / WARP;
  constexpr int VPT = NUM_EXPERTS / WARP;
  constexpr int EXPERTS_PER_GROUP = NUM_EXPERTS / NUM_GROUPS;
  constexpr int LANES_PER_GROUP = EXPERTS_PER_GROUP / VPT;
  // A power-of-two VPT is loaded with vectors of up to 16 bytes; any other
  // VPT (e.g. 160 experts -> 5 per lane) falls back to element loads.
  constexpr bool VPT_POW2 = (VPT & (VPT - 1)) == 0;
  constexpr int ELTS_PER_LDG =
      !VPT_POW2 ? 1 : ((VPT * sizeof(T) >= 16) ? 16 / sizeof(T) : VPT);
  constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(NUM_EXPERTS % WARP == 0 && VPT >= 1,
                "NUM_EXPERTS must be a multiple of 32");
  static_assert(NUM_EXPERTS % NUM_GROUPS == 0,
                "NUM_GROUPS must divide NUM_EXPERTS");
  static_assert(EXPERTS_PER_GROUP >= 2,
                "group score is the sum of the top-2 experts of a group");
  static_assert(EXPERTS_PER_GROUP % VPT == 0,
                "a lane's experts must not straddle two groups");
  static_assert((LANES_PER_GROUP & (LANES_PER_GROUP - 1)) == 0,
                "the group top-2 reduction needs a power-of-two lane count");
  static_assert(TOPK_GROUP >= 1 && TOPK_GROUP <= NUM_GROUPS,
                "TOPK_GROUP out of range");
  static_assert(TOPK_EXPERTS >= 1 && TOPK_EXPERTS <= WARP,
                "TOPK_EXPERTS must fit one slot per lane");
  static_assert(TOPK_EXPERTS <= TOPK_GROUP * EXPERTS_PER_GROUP,
                "not enough experts in the kept groups");

  T *input = static_cast<T *>(input_ptr);
  float const *bias = static_cast<float const *>(bias_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *routing_indices = static_cast<int *>(routing_indices_ptr);
  int *active_expert_ids = static_cast<int *>(active_expert_ids_ptr);

  // Expert-used flags, set by the row warps and compacted at the end.
  extern __shared__ char smem[];
  int *expert_used = reinterpret_cast<int *>(smem);
  int *warp_totals = expert_used + NUM_EXPERTS;

  int const tid = threadIdx.x;
  int const lane = tid % WARP;
  int const warp = tid / WARP;

  using Vec = typename VecOf<ELTS_PER_LDG * sizeof(T)>::type;
  union Pack {
    Vec v;
    T e[ELTS_PER_LDG];
  };
  auto row_ptr = [&](int row) {
    return reinterpret_cast<Vec *>(input + (size_t)row * NUM_EXPERTS +
                                   lane * VPT);
  };
  // Logits of the row this warp handles next. The first row is requested
  // before the routing buffers are cleared, and every later row while the
  // previous one is being routed, so a warp with several rows does not pay
  // a full memory round trip per row.
  Pack next[LDG_PER_THREAD];
  if (warp < num_rows) {
#pragma unroll
    for (int l = 0; l < LDG_PER_THREAD; l++) {
      next[l].v = row_ptr(warp)[l];
    }
  }
  // Per-lane bias is the same for every row.
  float lane_bias[VPT];
#pragma unroll
  for (int i = 0; i < VPT; i++) {
    lane_bias[i] = bias[lane * VPT + i];
  }

  // ---- Phase 0: clear routing state ----
  for (int i = tid; i < NUM_EXPERTS * num_rows; i += WORKER_NUM_THREADS) {
    routing_indices[i] = 0;
  }
  for (int e = tid; e < NUM_EXPERTS; e += WORKER_NUM_THREADS) {
    expert_used[e] = 0;
  }
  __syncthreads();

  for (int row = warp; row < num_rows; row += NUM_WORKER_WARPS) {
    // ---- Phase 1: sigmoid + bias; zero the logits; prefetch next row ----
    Pack cur[LDG_PER_THREAD];
#pragma unroll
    for (int l = 0; l < LDG_PER_THREAD; l++) {
      cur[l] = next[l];
    }
    if (row + NUM_WORKER_WARPS < num_rows) {
#pragma unroll
      for (int l = 0; l < LDG_PER_THREAD; l++) {
        next[l].v = row_ptr(row + NUM_WORKER_WARPS)[l];
      }
    }
    float score[VPT];  // unbiased sigmoid, used for the weights
    float biased[VPT]; // sigmoid + bias, used for the selection
#pragma unroll
    for (int l = 0; l < LDG_PER_THREAD; l++) {
#pragma unroll
      for (int e = 0; e < ELTS_PER_LDG; e++) {
        float const x = static_cast<float>(cur[l].e[e]);
        float const s = 1.0f / (1.0f + expf(-x));
        score[l * ELTS_PER_LDG + e] = s;
        biased[l * ELTS_PER_LDG + e] = s + lane_bias[l * ELTS_PER_LDG + e];
      }
    }
    {
      Pack z;
#pragma unroll
      for (int e = 0; e < ELTS_PER_LDG; e++) {
        z.e[e] = static_cast<T>(0.0f);
      }
#pragma unroll
      for (int l = 0; l < LDG_PER_THREAD; l++) {
        row_ptr(row)[l] = z.v;
      }
    }

    // ---- Phase 2/3: group scores and top-TOPK_GROUP group selection ----
    if constexpr (TOPK_GROUP < NUM_GROUPS) {
      float top1 = -INFINITY, top2 = -INFINITY;
#pragma unroll
      for (int i = 0; i < VPT; i++) {
        float const v = biased[i];
        if (v > top1) {
          top2 = top1;
          top1 = v;
        } else if (v > top2) {
          top2 = v;
        }
      }
#pragma unroll
      for (int mask = LANES_PER_GROUP / 2; mask > 0; mask /= 2) {
        float const o1 = __shfl_xor_sync(0xffffffff, top1, mask);
        float const o2 = __shfl_xor_sync(0xffffffff, top2, mask);
        if (o1 > top1) {
          top2 = fmaxf(top1, o2);
          top1 = o1;
        } else {
          top2 = fmaxf(top2, o1);
        }
      }
      float const my_group_score = top1 + top2;
      int const my_group = lane / LANES_PER_GROUP;
      // Rank of my group among all groups; ties go to the lower group id.
      int rank = 0;
#pragma unroll
      for (int g = 0; g < NUM_GROUPS; g++) {
        float const s =
            __shfl_sync(0xffffffff, my_group_score, g * LANES_PER_GROUP);
        rank += (s > my_group_score) || (s == my_group_score && g < my_group);
      }
      if (rank >= TOPK_GROUP) {
#pragma unroll
        for (int i = 0; i < VPT; i++) {
          biased[i] = -INFINITY;
        }
      }
    }

    // ---- Phase 4: top-TOPK_EXPERTS expert selection ----
    // Each lane tracks its best remaining expert as an integer key; a round
    // is one warp max-reduction plus a ballot to find the owning lane, which
    // then records the pick itself so nothing else sits on the round's
    // dependency chain.
    uint32_t key[VPT];
    uint32_t best_key = 0;
    int best_i = 0;
#pragma unroll
    for (int i = 0; i < VPT; i++) {
      key[i] = biased[i] == -INFINITY ? 0u : float_to_key(biased[i]);
      if (key[i] > best_key) {
        best_key = key[i];
        best_i = i;
      }
    }

    float my_sum = 0.0f; // sum of the unbiased scores this lane picked
    float *row_out = output + row * TOPK_EXPERTS;
#pragma unroll 1
    for (int k = 0; k < TOPK_EXPERTS; k++) {
      uint32_t const top = __reduce_max_sync(0xffffffff, best_key);
      int const owner = __ffs(__ballot_sync(0xffffffff, best_key == top)) - 1;
      if (lane == owner) {
        float pick_score = 0.0f;
#pragma unroll
        for (int i = 0; i < VPT; i++) {
          if (i == best_i) {
            pick_score = score[i];
            key[i] = 0u;
          }
        }
        int const expert = lane * VPT + best_i;
        row_out[k] = pick_score;
        my_sum += pick_score;
        routing_indices[expert * num_rows + row] = k + 1;
        expert_used[expert] = 1;
        best_key = 0;
        best_i = 0;
#pragma unroll
        for (int i = 0; i < VPT; i++) {
          if (key[i] > best_key) {
            best_key = key[i];
            best_i = i;
          }
        }
      }
    }

    // ---- Phase 5: normalize and scale ----
#pragma unroll
    for (int mask = WARP / 2; mask > 0; mask /= 2) {
      my_sum += __shfl_xor_sync(0xffffffff, my_sum, mask);
    }
    // Orders the owners' raw-weight stores before the rescale below.
    __syncwarp();
    if (lane < TOPK_EXPERTS) {
      row_out[lane] *= routed_scaling_factor / (my_sum + 1e-20f);
    }
  }
  __syncthreads();

  // ---- Phase 6: compact active expert ids in ascending order ----
  int running = 0;
  for (int base = 0; base < NUM_EXPERTS; base += WORKER_NUM_THREADS) {
    int const e = base + tid;
    bool const used = e < NUM_EXPERTS && expert_used[e] != 0;
    unsigned const ballot = __ballot_sync(0xffffffff, used);
    if (lane == 0) {
      warp_totals[warp] = __popc(ballot);
    }
    __syncthreads();
    int before = running;
    int total = 0;
#pragma unroll
    for (int w = 0; w < NUM_WORKER_WARPS; w++) {
      int const c = warp_totals[w];
      before += (w < warp) ? c : 0;
      total += c;
    }
    if (used) {
      active_expert_ids[before + __popc(ballot & ((1u << lane) - 1))] = e;
    }
    running += total;
    __syncthreads();
  }
  for (int i = running + tid; i < NUM_EXPERTS; i += WORKER_NUM_THREADS) {
    active_expert_ids[i] = -1;
  }
  if (tid == 0) {
    active_expert_ids[NUM_EXPERTS] = running;
  }
}

} // namespace kernel
