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
