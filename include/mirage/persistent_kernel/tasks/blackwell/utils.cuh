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
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

namespace kernel {

CUTLASS_DEVICE
static bool try_wait_barrier(uint64_t &smem_barrier, uint32_t phase) {
  uint32_t smem_int_ptr = cute::cast_smem_ptr_to_uint(&smem_barrier);
  uint32_t waitComplete;

  asm volatile("{\n\t"
               ".reg .pred P1; \n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
               "selp.b32 %0, 1, 0, P1; \n\t"
               "}"
               : "=r"(waitComplete)
               : "r"(smem_int_ptr), "r"(phase));

  return static_cast<bool>(waitComplete);
}

// The shared memory buffers for A, B, C, and D matrices.
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
} // namespace kernel
