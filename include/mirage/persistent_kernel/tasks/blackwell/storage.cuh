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

// Scaled piped task storage. The shared memory buffers for A, B, and C matrices and its scale factor.
template <class TypeA, 
          class TypeB, 
          class TypeC, 
          class TypeSF,
          class ASmemLayout,
          class BSmemLayout,
          class CSmemLayout,
          class SFASmemLayout,
          class SFBSmemLayout,
          int Num_AB_Stage,
          int Num_ACC_Stage>
struct PipedScaledSharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(128) cute::ArrayEngine<TypeC, cute::cosize_v<CSmemLayout>> C;
  alignas(128) cute::ArrayEngine<TypeSF, cute::cosize_v<SFASmemLayout>> SFA;
  alignas(128) cute::ArrayEngine<TypeSF, cute::cosize_v<SFBSmemLayout>> SFB;

  alignas(16) cute::uint64_t ab_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t sf_full_mbar_ptr[Num_AB_Stage];       // SFA/SFB arrived in SMEM
  alignas(16) cute::uint64_t sf_empty_mbar_ptr[Num_AB_Stage];      // SF SMEM buffer free for reuse
  alignas(16) cute::uint64_t sf_tmem_full_mbar_ptr[1];  // SFA/SFB copied to TMEM (single TMEM buffer)
  alignas(16) cute::uint64_t sf_tmem_empty_mbar_ptr[1]; // TMEM SF buffer free for reuse (single TMEM buffer)
  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];

  alignas(16) cute::uint32_t tmem_acc_ptr;
  alignas(16) cute::uint32_t tmem_sfa_ptr;
  alignas(16) cute::uint32_t tmem_sfb_ptr;

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
