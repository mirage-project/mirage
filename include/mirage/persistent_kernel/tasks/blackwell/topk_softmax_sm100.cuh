#pragma once
#include <cstdio>
#include <iostream>

// Use Thrust to handle host/device allocations
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

// topk_reduce includes
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

// mirage includes
#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k. 3) This implementation assumes 8 warps are being used.
*/

namespace kernel {

static constexpr int WARP_SIZE = 32;

// ====================== Fused TopK softmax kernel
// =============================== This kernel fuses the softmax, max and argmax
// into a single kernel. Block size is strictly 256 (8 warps): dim3
// block(WARP_SIZE*WARPS_PER_CTA, 1, 1)
template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG>
__device__ __forceinline__ void topk_softmax_task_impl(
    void *__restrict__ input_ptr, // [num_rows, NUM_EXPERTS]
    bool const *__restrict__ finished,
    void *__restrict__ output_ptr, // [num_rows, k]
    int const num_rows,
    int const k,
    void *__restrict__ mpk_routing_indices_ptr, // [NUM_EXPERTS, num_rows] laid
                                                // out as expert-major: expert *
                                                // num_rows
                                                // + row
    void *__restrict__ mpk_active_expert_ids_ptr, // [NUM_EXPERTS + 1] last
                                                  // element stores num active
                                                  // experts
    int const start_expert,
    int const end_expert,
    bool const renormalize,
    bool const round_weights_to_bf16 = false) {
  // Pointers
  T *input = static_cast<T *>(input_ptr);
  float *output = static_cast<float *>(output_ptr);
  int *mpk_routing_indices = static_cast<int *>(mpk_routing_indices_ptr);
  int *mpk_active_expert_ids = static_cast<int *>(mpk_active_expert_ids_ptr);
  // initialize routing indices to 0; active-id marks to -1; count to 0
  for (int expert = start_expert + threadIdx.x; expert < end_expert;
       expert += blockDim.x) {
    if (mpk_routing_indices != nullptr) {
      for (int row = 0; row < num_rows; ++row) {
        mpk_routing_indices[expert * num_rows + row] = 0;
      }
    }
    if (mpk_active_expert_ids != nullptr) {
      mpk_active_expert_ids[expert - start_expert] = -1;
    }
  }
  // Thread 0 always exists, unlike threadIdx.x == NUM_EXPERTS when
  // NUM_EXPERTS == blockDim.x == 256.
  if (threadIdx.x == 0 && mpk_active_expert_ids != nullptr) {
    mpk_active_expert_ids[NUM_EXPERTS] = 0;
  }
  __syncthreads();
  // Compile-time checks
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW =
      ELTS_PER_ROW / VPT; // subgroup size in a warp
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE,
                "THREADS_PER_ROW can be at most warp size");
  static_assert(THREADS_PER_ROW == WARP_SIZE ||
                    THREADS_PER_ROW == WARP_SIZE / 2,
                "This kernel only supports THREADS_PER_ROW of 16 or 32");

  // Qwen3.6 rounds the eight renormalized routing weights to BF16. A one-ULP
  // FP32 softmax difference can change that result at a BF16 midpoint. For
  // its 256 experts, reproduce PyTorch's installed PersistentSoftmax layout:
  // one 32-lane warp per row, eight elements per lane at stride 32, then XOR
  // reductions from lane distance 16 to 1. Reuse the worker's dynamic shared
  // memory instead of increasing its static shared-memory footprint.
  bool const torch_style = round_weights_to_bf16 && num_rows <= 2 * WARPS_PER_CTA &&
                           NUM_EXPERTS == 256 && THREADS_PER_ROW == WARP_SIZE &&
                           k == 8 && start_expert == 0 && end_expert == 256;
  extern __shared__ float qwen36_torch_probabilities[];
  if (torch_style) {
    // B16 uses two waves of eight warps. Preserve the same per-row
    // PyTorch softmax and small-sort order used by B8.
    for (int row_block = 0; row_block < num_rows; row_block += WARPS_PER_CTA) {
    int row = row_block + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (row < num_rows) {
      float elements[8];
      float peak = -INFINITY;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        elements[i] = float(input[row * NUM_EXPERTS + lane + WARP_SIZE * i]);
        peak = peak > elements[i] ? peak : elements[i];
      }
      for (int offset = WARP_SIZE / 2; offset; offset >>= 1)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, offset));
      float sum = 0.f;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        elements[i] = expf(elements[i] - peak);
        sum = __fadd_rn(sum, elements[i]);
      }
      for (int offset = WARP_SIZE / 2; offset; offset >>= 1)
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
      #pragma unroll
      for (int i = 0; i < 8; ++i)
        qwen36_torch_probabilities[row * NUM_EXPERTS + lane + WARP_SIZE * i] =
            __fdiv_rn(elements[i], sum);
    }
    __syncthreads();

    // PyTorch 2.7 gatherTopK first collects all values above the kth value in
    // expert-index order, then the first equal values needed to fill eight
    // slots. Its small-sort path applies a 32-element bitonic network to
    // these eight valid entries. The network's tie order affects the BF16
    // expert accumulation even when the eight weights are identical.
    if (threadIdx.x % WARP_SIZE == 0 && row < num_rows) {
      float const *probs = qwen36_torch_probabilities + row * NUM_EXPERTS;
      float highest[8];
      #pragma unroll
      for (int j = 0; j < 8; ++j) highest[j] = -INFINITY;
      for (int expert = 0; expert < NUM_EXPERTS; ++expert) {
        float v = probs[expert];
        for (int j = 0; j < 8; ++j) {
          if (v > highest[j]) {
            for (int next = 7; next > j; --next)
              highest[next] = highest[next - 1];
            highest[j] = v;
            break;
          }
        }
      }
      float threshold = highest[7];
      float keys[32] = {};
      int experts[32] = {};
      bool valid[32] = {};
      int count = 0;
      for (int expert = 0; expert < NUM_EXPERTS; ++expert) {
        if (probs[expert] > threshold) {
          keys[count] = probs[expert];
          experts[count] = expert;
          valid[count++] = true;
        }
      }
      for (int expert = 0; expert < NUM_EXPERTS && count < 8; ++expert) {
        if (probs[expert] == threshold) {
          keys[count] = probs[expert];
          experts[count] = expert;
          valid[count++] = true;
        }
      }
      for (int size = 2; size < 32; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
          for (int thread = 0; thread < 16; ++thread) {
            int p = 2 * thread - (thread & (stride - 1));
            int q = p + stride;
            bool dir = (thread & (size / 2)) != 0;
            bool swap = ((keys[p] > keys[q] && valid[p]) || !valid[q]);
            if (swap == dir) {
              float tmp_key = keys[p]; keys[p] = keys[q]; keys[q] = tmp_key;
              int tmp_expert = experts[p]; experts[p] = experts[q]; experts[q] = tmp_expert;
              bool tmp_valid = valid[p]; valid[p] = valid[q]; valid[q] = tmp_valid;
            }
          }
        }
      }
      for (int stride = 16; stride > 0; stride /= 2) {
        for (int thread = 0; thread < 16; ++thread) {
          int p = 2 * thread - (thread & (stride - 1));
          int q = p + stride;
          bool swap = ((keys[p] > keys[q] && valid[p]) || !valid[q]);
          if (!swap) {
            float tmp_key = keys[p]; keys[p] = keys[q]; keys[q] = tmp_key;
            int tmp_expert = experts[p]; experts[p] = experts[q]; experts[q] = tmp_expert;
            bool tmp_valid = valid[p]; valid[p] = valid[q]; valid[q] = tmp_valid;
          }
        }
      }
      // torch.sum over eight FP32 values reduces adjacent pairs before
      // combining the two halves. A sequential sum can move a normalized
      // value across a BF16 midpoint even when the top-k probabilities agree.
      float pair0 = __fadd_rn(keys[0], keys[1]);
      float pair1 = __fadd_rn(keys[2], keys[3]);
      float pair2 = __fadd_rn(keys[4], keys[5]);
      float pair3 = __fadd_rn(keys[6], keys[7]);
      float selected_sum = __fadd_rn(__fadd_rn(pair0, pair1),
                                     __fadd_rn(pair2, pair3));
      bool row_active = finished ? !finished[row] : true;
      for (int j = 0; j < 8; ++j) {
        float weight = renormalize ? __fdiv_rn(keys[j], selected_sum) : keys[j];
        output[row * 8 + j] = round_weights_to_bf16 ? float(T(weight)) : weight;
        int expert = experts[j];
        if (row_active && mpk_routing_indices != nullptr)
          mpk_routing_indices[expert * num_rows + row] = j + 1;
        if (row_active && mpk_active_expert_ids != nullptr)
          mpk_active_expert_ids[expert] = expert;
      }
    }
    __syncthreads();
    }
    for (int i = threadIdx.x; i < num_rows * NUM_EXPERTS; i += blockDim.x)
      input[i] = T(0.f);
  } else {
  // Work partitioning
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP =
      ELTS_PER_WARP / ELTS_PER_ROW; // rows each warp processes
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  int const warp_idx = threadIdx.x / WARP_SIZE;
  int const lane_idx = threadIdx.x % WARP_SIZE;
  int const warp_base_row = warp_idx * ROWS_PER_WARP;

  int const thread_row_in_warp = lane_idx / THREADS_PER_ROW;
  for (int row_block = 0; row_block < num_rows; row_block += ROWS_PER_CTA) {
    int const thread_row = row_block + warp_base_row + thread_row_in_warp;
    uint32_t warp_mask = 0xffffffffu;
    if constexpr (THREADS_PER_ROW != WARP_SIZE) {
      constexpr uint32_t subgroup_mask = (1u << THREADS_PER_ROW) - 1u;
      // The final warp can contain a single live sub-group when the final row
      // block has an odd row count. Restrict the shuffle mask to that
      // sub-group instead of hard-coding the lower 16 lanes.
      int const rows_in_block = min(ROWS_PER_CTA, num_rows - row_block);
      if ((rows_in_block % ROWS_PER_WARP) != 0 &&
          thread_row == row_block + rows_in_block - 1) {
        warp_mask = subgroup_mask << (thread_row_in_warp * THREADS_PER_ROW);
      }
    }
    if (thread_row < num_rows) {

      bool const row_is_active = finished ? !finished[thread_row] : true;

      // Compute per-thread read pointers
      T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
      int const thread_group_idx = lane_idx % THREADS_PER_ROW;
      int const first_elt_read_by_thread =
          thread_group_idx * (BYTES_PER_LDG / sizeof(T));
      T *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

      using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;
      T row_chunk_temp[VPT];
      AccessType *row_chunk_vec_ptr =
          reinterpret_cast<AccessType *>(&row_chunk_temp);
      AccessType *vec_thread_read_ptr =
          reinterpret_cast<AccessType *>(thread_read_ptr);

      // Vectorized loads across the row
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
      }

      cutlass::NumericConverter<float, T> converter;

      float row_chunk[VPT];
      for (int ii = 0; ii < VPT; ++ii) {
        int col = first_elt_read_by_thread +
                  (ii / ELTS_PER_LDG) * ELTS_PER_LDG * THREADS_PER_ROW +
                  ii % ELTS_PER_LDG;
        row_chunk[ii] = torch_style
            ? qwen36_torch_probabilities[thread_row * NUM_EXPERTS + col]
            : converter(row_chunk_temp[ii]);
        row_chunk_temp[ii] = static_cast<T>(
            0); // reset input buffer to 0 for split-k gate linear
      }

      // reset input buffer to 0 for split-k gate linear
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        vec_thread_read_ptr[ii * THREADS_PER_ROW] = row_chunk_vec_ptr[ii];
      }

      if (!torch_style) {
        // Existing fused path for models that do not round routing weights.
        float thread_max = row_chunk[0];
        for (int ii = 1; ii < VPT; ++ii)
          thread_max = max(thread_max, row_chunk[ii]);
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
          float other =
              __shfl_xor_sync(warp_mask, thread_max, mask, THREADS_PER_ROW);
          thread_max = max(thread_max, other);
        }
        float row_sum = 0.f;
        for (int ii = 0; ii < VPT; ++ii) {
          row_chunk[ii] = expf(row_chunk[ii] - thread_max);
          row_sum += row_chunk[ii];
        }
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
          row_sum += __shfl_xor_sync(warp_mask, row_sum, mask, THREADS_PER_ROW);
        float const inv_row_sum = 1.f / row_sum;
        for (int ii = 0; ii < VPT; ++ii)
          row_chunk[ii] = row_chunk[ii] * inv_row_sum;
      }

      // Fused Top-K selection within subgroup
      int start_col = first_elt_read_by_thread;
      static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
      float row_sum_for_renormalize = 0.f;

      for (int k_idx = 0; k_idx < k; ++k_idx) {
        float max_val = row_chunk[0];
        int expert = start_col;
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
             ++ldg, col += COLS_PER_GROUP_LDG) {
          for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
            float val = row_chunk[ldg * ELTS_PER_LDG + ii];
            if (val > max_val) {
              max_val = val;
              expert = col + ii;
            }
          }
        }

        // Argmax reduce across subgroup with index tie-breaker (prefer lower
        // index)
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
          float other_max =
              __shfl_xor_sync(warp_mask, max_val, mask, THREADS_PER_ROW);
          int other_expert =
              __shfl_xor_sync(warp_mask, expert, mask, THREADS_PER_ROW);
          if (other_max > max_val ||
              (other_max == max_val && other_expert < expert)) {
            max_val = other_max;
            expert = other_expert;
          }
        }

        // Write out the selected top-k value/index (one thread per subgroup
        // writes)
        if (thread_group_idx == 0) {
          bool const node_uses_expert =
              expert >= start_expert && expert < end_expert;
          bool const should_process_row = row_is_active && node_uses_expert;
          // (batch, topk) row-major — matches mul_sum_add weight layout.
          int const out_idx = thread_row * k + k_idx;
          output[out_idx] = max_val;
          // indices[out_idx] =
          //     should_process_row ? (expert - start_expert) : NUM_EXPERTS;
          row_sum_for_renormalize += max_val;
          // Optionally populate MPK routing structures
          if (should_process_row && mpk_routing_indices != nullptr) {
            int const local_expert = expert - start_expert;
            // Write 1-based rank into routing indices; stride by num_rows per
            // expert
            mpk_routing_indices[local_expert * num_rows + thread_row] =
                k_idx + 1;
            // Sparse mark expert as active; idempotent without atomics
            if (mpk_active_expert_ids != nullptr) {
              mpk_active_expert_ids[local_expert] = local_expert;
            }
          }
        }

        // Blank out the winning value for the next iteration
        if (k_idx + 1 < k) {
          int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
          int const thread_to_clear_in_group =
              (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
          if (thread_group_idx == thread_to_clear_in_group) {
            int const offset_for_expert = expert % ELTS_PER_LDG;
            row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
                -10000.f;
          }
        }
      }

      // Optional renormalization of top-k weights
      if (renormalize && thread_group_idx == 0) {
        float inv = 1.f / row_sum_for_renormalize;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
          int const out_idx = thread_row * k + k_idx;
          output[out_idx] = output[out_idx] * inv;
          if (round_weights_to_bf16)
            output[out_idx] = float(T(output[out_idx]));
        }
      }
    }
  }
  }
  __syncthreads();
  // Compact marks into a dense list and count
  if (mpk_active_expert_ids != nullptr) {
    if (round_weights_to_bf16 && NUM_EXPERTS == 256 && WARPS_PER_CTA == 8 &&
        blockDim.x == 256 && start_expert == 0 && end_expert == NUM_EXPERTS) {
      // Qwen3.6's atomic compaction assigns the same experts to different
      // expert-offset tasks on different launches. Give each active expert
      // a stable rank in expert-id order. Every lane reads its mark before
      // any lane overwrites the shared marks/list buffer.
      __shared__ int warp_counts[WARPS_PER_CTA];
      int lane = threadIdx.x % WARP_SIZE;
      int warp = threadIdx.x / WARP_SIZE;
      int mark = mpk_active_expert_ids[threadIdx.x];
      unsigned int active = __ballot_sync(0xffffffffu, mark >= 0);
      if (lane == 0) warp_counts[warp] = __popc(active);
      __syncthreads();
      int rank = __popc(active & ((1u << lane) - 1u));
      for (int earlier = 0; earlier < warp; ++earlier)
        rank += warp_counts[earlier];
      if (mark >= 0) mpk_active_expert_ids[rank] = threadIdx.x;
      if (threadIdx.x == 0) {
        int count = 0;
        for (int w = 0; w < WARPS_PER_CTA; ++w) count += warp_counts[w];
        mpk_active_expert_ids[NUM_EXPERTS] = count;
      }
      __syncthreads();
      return;
    }
    // The compacted list shares storage with the sparse marks. Read all
    // marks before any lane overwrites an entry another lane still needs.
    constexpr int MARKS_PER_THREAD =
        (NUM_EXPERTS + WARPS_PER_CTA * WARP_SIZE - 1) /
        (WARPS_PER_CTA * WARP_SIZE);
    int marks[MARKS_PER_THREAD];
    int const local_experts = end_expert - start_expert;
    for (int i = 0; i < MARKS_PER_THREAD; ++i) {
      int const local_expert = threadIdx.x + i * blockDim.x;
      marks[i] = local_expert < local_experts
                     ? mpk_active_expert_ids[local_expert]
                     : -1;
    }
    __syncthreads();
    for (int i = 0; i < MARKS_PER_THREAD; ++i) {
      int const local_expert = threadIdx.x + i * blockDim.x;
      if (local_expert >= local_experts) continue;
      int const mark = marks[i];
      if (mark >= 0) {
        int const pos = atomicAdd(mpk_active_expert_ids + NUM_EXPERTS, 1);
        mpk_active_expert_ids[pos] = start_expert + local_expert;
      }
    }
  }
}

namespace detail {
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                "");
  static constexpr int
      VECs_PER_THREAD = (EXPERTS / (ELTS_PER_LDG * WARP_SIZE)) > 0
                            ? (EXPERTS / (ELTS_PER_LDG * WARP_SIZE))
                            : 1;
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / (EXPERTS / VPT);
};
} // namespace detail

} // namespace kernel
