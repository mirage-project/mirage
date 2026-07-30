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

// Gated-DeltaNet causal depthwise conv1d (short FIR) with a persistent
// per-request-slot convolution state pool.  This is the MPK port of vLLM's
// `causal_conv1d_update` for Qwen3.5's GDN layers
// (docs/qwen35/vllm-graph.md 2.1.4, docs/qwen35/v1-architecture.md 3.1-3.3).
//
// Math, per channel d and per new token t of a request's chunk:
//
//   window   = [ s_{t-3}, s_{t-2}, s_{t-1}, x_t ]      (KERNEL_SIZE taps)
//   y[t,d]   = silu( bf16( sum_j W[d,j] * window[j] ) ) (conv1d.bias is None)
//   state'   = last (KERNEL_SIZE-1) entries of [state || chunk]
//
// NUMERIC TARGET - why the accumulator is rounded before the activation.
// v1-architecture.md 1 pins HF `transformers` as the numeric target and says
// that where vLLM and HF differ in low-order bits, HF's behaviour is the one to
// match. This op is exactly such a case:
//
//   vLLM (causal_conv1d.py:943)      SiLU on the fp32 accumulator, one rounding
//   HF  (torch_causal_conv1d_update) `F.conv1d` runs in the weight dtype, so
//                                    the accumulator is rounded to bf16 and
//                                    `F.silu` sees the bf16 value
//
// Measured against the M2-I3 oracle (`test_gdn_conv1d_oracle.py`, real layer-0
// weights): the HF order reproduces `gdn.conv_out` BIT-EXACTLY (0 of 65536
// prefill / 0 of 8192 decode elements differ) while the vLLM order differs on
// ~24% of elements by up to 3 bf16 ULPs. The FIR sum itself stays in fp32.
//
// Layout contract (all bf16 unless noted):
//   input   [num_tokens, INPUT_STRIDE]        row-major, channel is the inner
//                                             axis; the task is handed the row
//                                             of its chunk's first token, at
//                                             its own channel offset.
//   weight  [CONV_DIM, KERNEL_SIZE]           conv1d.weight.view(CONV_DIM, 4),
//                                             sliced to this task's channels.
//   state   [num_slots, KERNEL_SIZE-1, CONV_DIM]
//                                             vLLM's "SD" layout: the 8192-wide
//                                             channel axis is innermost
//                                             (vllm-graph.md 2.1.5).  The task
//                                             is handed its own slot's slice at
//                                             its own channel offset, so
//                                             CONV_DIM stays the ROW stride
//                                             while CHANNELS is what this task
//                                             actually walks.
//   output  [num_tokens, OUTPUT_STRIDE]       separate buffer, not in-place
//                                             (v1-architecture.md 3.2).
//
// Lifecycle (v1-architecture.md 3.3): there is no `prepare_next_batch` hook.
// `zero_state` is the kernel-side predicate `runtime_config.step[req] == 0`,
// i.e. the first prefill chunk of whichever request occupies the slot; a slot
// reused by a later request re-zeros implicitly because its step restarts at 0.
// The updated state is written back unconditionally.
//
// Parallelism: one task == one (request slot, channel block) pair -
// grid = (max_num_batched_requests, CONV_DIM / CHANNELS, 1), with
// `task_metadata.request_id` = the slot and `task_metadata.kv_idx` = the
// channel block. blockIdx is NEVER used to derive identity - it is the worker
// id, not the task id.
//
// The channel split is what vLLM's Triton kernel does too (grid (B, 32),
// BLOCK_N = 256 over 8192 channels, vllm-graph.md 2.1.7 row 6) and it is what
// makes chunked prefill scale: the FIR is causal but has NO dependency between
// output tokens, so a chunk's only serial axis is the register window each
// thread carries. With one task per slot a 256-token chunk pinned an entire
// layer to one SM (measured 1.84 ms/layer); splitting the channels spreads the
// same work over CONV_DIM/CHANNELS tasks. Within a task, channels are split
// across the block's threads and the token loop is a register shift.

namespace kernel {

// `runtime_config.step` is indexed by REQUEST id, while every per-request task
// (and `qo_indptr_buffer`) is indexed by batch SLOT
// (persistent_kernel.cuh:231-236 reads `step[request_ids[i]]` for slot `i`).
// This helper does that one indirection so the generated task code stays free
// of serving-mode preprocessor conditionals.
//
// Returns true when the slot is running the FIRST chunk of whatever request
// occupies it, i.e. when its conv state must be treated as zero
// (v1-architecture.md 3.3).  An empty slot (request id -1) also reports true;
// such a task early-returns on Q_LEN == 0 before it can matter.
// NOTE: fully qualified - `using namespace mirage::runtime` is pulled in by
// persistent_kernel.cuh only AFTER the task headers are included, so task .cuh
// files never see the runtime types unqualified.
__device__ __forceinline__ bool
    gdn_slot_is_first_chunk(mirage::runtime::RuntimeConfig const &config,
                            int slot) {
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) ||                           \
    defined(MODE_ONLINE_NOTOKEN) || defined(MODE_ONLINE_TEST) ||               \
    defined(MODE_ONLINE_PINNED)
  int request_id = config.request_ids[slot];
  if (request_id < 0) {
    return true;
  }
  return config.step[request_id] == 0;
#else
  // Modes without a slot->request table run one request per slot.
  return config.step[slot] == 0;
#endif
}

// CONV_DIM   - full channel count; the ROW stride of the conv-state pool.
// CHANNELS   - channels this task owns (CONV_DIM when the grid has one channel
//              block). All four pointers arrive pre-offset to this task's
//              channel block.
template <typename T,
          int CONV_DIM,
          int CHANNELS,
          int KERNEL_SIZE,
          int INPUT_STRIDE,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void
    gdn_conv1d_sm100_task_impl(void const *input_ptr,
                               void const *weight_ptr,
                               void *state_ptr,
                               void *output_ptr,
                               int q_len,
                               bool zero_state) {
  constexpr int STATE_LEN = KERNEL_SIZE - 1;
  static_assert(KERNEL_SIZE >= 2, "conv width must be at least 2");
  static_assert(CHANNELS > 0 && CHANNELS <= CONV_DIM,
                "CHANNELS must be a slice of CONV_DIM");

  // A slot with no new tokens this iteration (Q_LEN == 0 from qo_indptr) is
  // inactive: return before touching its state so a parked request's state
  // survives untouched.
  if (q_len <= 0) {
    return;
  }

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_state = static_cast<T *>(state_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // `d` is a channel index LOCAL to this task's block; the state's row stride
  // stays CONV_DIM because the pool row spans every channel.
  for (int d = threadIdx.x; d < CHANNELS; d += blockDim.x) {
    // Rolling FIR window: taps [0, STATE_LEN) are the carried state, tap
    // STATE_LEN is the token being produced.
    float window[KERNEL_SIZE];
#pragma unroll
    for (int j = 0; j < STATE_LEN; j++) {
      window[j] =
          zero_state ? 0.0f : static_cast<float>(d_state[j * CONV_DIM + d]);
    }
    float w[KERNEL_SIZE];
#pragma unroll
    for (int j = 0; j < KERNEL_SIZE; j++) {
      w[j] = static_cast<float>(d_weight[d * KERNEL_SIZE + j]);
    }

    // Tokens of a chunk are processed in order: the FIR is causal, so token t
    // needs the (KERNEL_SIZE-1) inputs before it, which for t >= STATE_LEN come
    // from the chunk itself rather than from the carried state.
    for (int t = 0; t < q_len; t++) {
      window[KERNEL_SIZE - 1] =
          static_cast<float>(d_input[t * INPUT_STRIDE + d]);
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < KERNEL_SIZE; j++) {
        acc += w[j] * window[j];
      }
      // The FIR accumulates in fp32, is rounded to T, and only THEN goes
      // through SiLU - see the "numeric target" note in this file's header.
      // The round-trip is a single `cvt.rn.bf16.f32` and is what makes the
      // kernel bit-exact against the HF oracle.
      float y = static_cast<float>(T(acc));
      d_output[t * OUTPUT_STRIDE + d] = T(y / (1.0f + expf(-y)));
#pragma unroll
      for (int j = 0; j < KERNEL_SIZE - 1; j++) {
        window[j] = window[j + 1];
      }
    }

    // window[0 .. STATE_LEN) now holds the last STATE_LEN entries of
    // [state || chunk] - the state the next chunk/step consumes.
#pragma unroll
    for (int j = 0; j < STATE_LEN; j++) {
      d_state[j * CONV_DIM + d] = T(window[j]);
    }
  }
}

} // namespace kernel
