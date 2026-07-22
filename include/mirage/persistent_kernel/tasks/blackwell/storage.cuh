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
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

namespace kernel {

// ---------------------------------------------------------------------------
// ASYNC-AGENT SAFETY.
//
// Declare this in STATIC __shared__ — never inside a storage struct that is
// placed in the `extern __shared__` arena:
//
//   __shared__ PipedBarriers<NUM_AB_STAGE, NUM_ACC_STAGE> sm_bars;
//
// Every task body's `extern __shared__` declaration aliases ONE arena at ONE
// base, and a persistent worker CTA runs heterogeneous tasks back-to-back
// separated only by __syncthreads(). __syncthreads() orders THREADS; it drains
// no ASYNCHRONOUS agent. A TMA `expect_tx` completion or a
// `tcgen05.commit ... mbarrier::arrive` keeps writing an mbarrier's state word
// after the issuing task has nominally ended, so an ARENA-RESIDENT barrier
// lets a late arrival land in memory the NEXT task has already reused — a
// fault or silent corruption depending only on what occupies that byte.
//
// nvcc SUMS per-branch static __shared__ (it does not overlay mutually
// exclusive dispatch branches) and places all of it BELOW the dynamic arena
// base, so a barrier block declared this way belongs to one task instantiation
// alone. A late arrival then lands on storage no other task ever touches.
//
// Keeping the TMEM allocation slot here too is deliberate: it sat immediately
// after the barrier array and was the observed victim of a sibling task's
// arrival (read back as an mbarrier state word, 0x001ffffe, and fed to
// tcgen05.mma as a TMEM base).
template <int Num_AB_Stage, int Num_ACC_Stage>
struct PipedBarriers {
  alignas(16) cute::uint64_t ab_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];

  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation
  alignas(16) cute::uint32_t tmem_columns;  // TMEM column allocation size
};

// Linear task storage. The shared memory buffers for A, B, and C matrices.
template <class TypeA, // Tensor A data type
          class TypeB, // Tensor B data type
          class TypeC, // Tensor C data type
          class ASmemLayout,
          class BSmemLayout,
          class CSmemLayout,
          int Num_AB_Stage,
          int Num_ACC_Stage>
struct PipedSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(128) cute::ArrayEngine<TypeC, cute::cosize_v<CSmemLayout>> C;

  alignas(16) cute::uint64_t ab_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];

  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() {
    return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sC() {
    return cute::make_tensor(cute::make_smem_ptr(C.begin()), CSmemLayout{});
  }
};

// Linear task storage with scale factors (SFA/SFB).
template <class TypeA,  // Tensor A data type
          class TypeB,  // Tensor B data type
          class TypeC,  // Tensor C data type
          class TypeSF, // Scale factor data type
          class ASmemLayout,
          class BSmemLayout,
          class CSmemLayout,
          class SFASmemLayout,
          class SFBSmemLayout,
          int Num_AB_Stage,
          int Num_ACC_Stage>
struct PipedSharedStorageWithSF {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(128) cute::ArrayEngine<TypeC, cute::cosize_v<CSmemLayout>> C;
  alignas(128) cute::ArrayEngine<TypeSF, cute::cosize_v<SFASmemLayout>> SFA;
  alignas(128) cute::ArrayEngine<TypeSF, cute::cosize_v<SFBSmemLayout>> SFB;

  alignas(16) cute::uint64_t ab_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];

  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation
  alignas(16) cute::uint32_t tmem_columns;  // TMEM column allocation size

  CUTE_DEVICE constexpr auto tensor_sA() {
    return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sC() {
    return cute::make_tensor(cute::make_smem_ptr(C.begin()), CSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sSFA() {
    return cute::make_tensor(cute::make_smem_ptr(SFA.begin()), SFASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sSFB() {
    return cute::make_tensor(cute::make_smem_ptr(SFB.begin()), SFBSmemLayout{});
  }
};

// Gated Topk storage. The shared memory buffers for A, B, and C matrices.
template <class TypeA,   // Tensor A data type
          class TypeB,   // Tensor B data type
          class TypeRed, // Tensor C data type
          class ASmemLayout,
          class BSmemLayout,
          int Num_AB_Stage,
          int Num_ACC_Stage>
struct GateTopKSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t ab_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];

  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  alignas(16) TypeRed reduce_values_buffer[32]; // Buffer for reduction values

  CUTE_DEVICE constexpr auto tensor_sA() {
    return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

// MoE Linear task storage. The shared memory buffers for A, B, and C matrices.
template <class TypeA, // Tensor A data type
          class TypeB, // Tensor B data type
          class ASmemLayout,
          class BSmemLayout,
          class BSmemCpLayout,
          int Num_Experts,
          int Num_AB_Stage,
          int Num_ACC_Stage>
struct MoESharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t a_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t b_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];

  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];

  alignas(16) cute::uint32_t expert_mask[Num_Experts];

  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() {
    return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_cp_sB() {
    return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemCpLayout{});
  }
};

} // namespace kernel
