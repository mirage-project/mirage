// Typed SMEM buffer abstraction for v2 task SMEM layout.
//
// Each task type declares a struct of named buffers (one per logical SMEM
// region it uses). Buffer offsets within the task's extern __shared__ region
// are computed at compile time as constexpr accumulations.
//
// Phase 1 of the SMEM redesign: pure code reorganization, no synchronization
// changes. Task body uses `buffers.foo.ptr<T>()` instead of
// `(T*)(smem + FOO_OFFSET)`.
//
// Future phases will add:
//   - mbarrier handshake per buffer (Phase 4: Loader prefetch)
//   - Cross-task SMEM packing via codegen liveness analysis

#pragma once

#include <cstdint>

namespace mirage {
namespace runtime_v2 {

__host__ __device__ constexpr int round_up(int n, int align) {
  return (n + align - 1) & ~(align - 1);
}

// Single typed buffer over a SMEM byte range.
//
// `BYTES` is the unrounded logical size; the actual bytes consumed is
// round_up(BYTES, ALIGN). Default ALIGN=1024 so TMA-swizzle (128B) is safe;
// override with ALIGN=16 (or 4) for buffers that don't receive TMA stores
// (small scratch / scalar reductions / zero pads).
//
// Usage in task body:
//   extern __shared__ char smem[];
//   auto bufs = TaskBuffers(smem);
//   T* p = bufs.foo.template ptr<T>();
template <int BYTES, int ALIGN = 1024>
struct SmemBuffer {
  static constexpr int LOGICAL_BYTES = BYTES;
  static constexpr int PADDED_BYTES = round_up(BYTES, ALIGN);

  char *base_;

  __host__ __device__ explicit SmemBuffer(char *base) : base_(base) {}

  template <typename T>
  __device__ T *ptr() const {
    return reinterpret_cast<T *>(base_);
  }

  __device__ char *raw() const { return base_; }
};

// Helper: chain offsets so each buffer in a struct gets the correct base.
//
// Use the SMEM_BUFFER_DECL macro pattern below to declare a buffer struct,
// or write the offsets manually:
//
//   struct Buffers {
//     SmemBuffer<8192> input;
//     SmemBuffer<8192> weight;
//     SmemBuffer<8192> output;
//     SmemBuffer<32>   reduce;
//
//     static constexpr int INPUT_OFFSET  = 0;
//     static constexpr int WEIGHT_OFFSET = INPUT_OFFSET  + decltype(input)::PADDED_BYTES;
//     static constexpr int OUTPUT_OFFSET = WEIGHT_OFFSET + decltype(weight)::PADDED_BYTES;
//     static constexpr int REDUCE_OFFSET = OUTPUT_OFFSET + decltype(output)::PADDED_BYTES;
//     static constexpr int TOTAL_BYTES   = REDUCE_OFFSET + decltype(reduce)::PADDED_BYTES;
//
//     __device__ explicit Buffers(char *smem)
//         : input(smem + INPUT_OFFSET),
//           weight(smem + WEIGHT_OFFSET),
//           output(smem + OUTPUT_OFFSET),
//           reduce(smem + REDUCE_OFFSET) {}
//   };

}  // namespace runtime_v2
}  // namespace mirage
