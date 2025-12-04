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


// Prefill attention task storage. The shared memory buffers for A, B, and C matrices.
template <class TypeQ, // Tensor A data type
          class TypeKV, // Tensor B data type
          class QSmemLayout,
          class KVSmemLayout,
          class OSmemLayout,
          int QO_TILE_SIZE,
          int NUM_Q_STAGE,
          int NUM_KV_STAGE,
          int NUM_EPI_STAGE>
struct PrefillAttnStorage {
  alignas(128) cute::ArrayEngine<TypeKV, cute::cosize_v<OSmemLayout>> sO;
  alignas(128) cute::ArrayEngine<TypeQ, cute::cosize_v<QSmemLayout>> sQ;
  alignas(128) cute::ArrayEngine<TypeKV, cute::cosize_v<KVSmemLayout>> sK;

  alignas(16) cute::uint64_t load_q_full_mbar_ptr[NUM_Q_STAGE];
  alignas(16) cute::uint64_t load_q_empty_mbar_ptr[NUM_Q_STAGE];
  alignas(16) cute::uint64_t load_kv_full_mbar_ptr[NUM_KV_STAGE];
  alignas(16) cute::uint64_t load_kv_empty_mbar_ptr[NUM_KV_STAGE];
  alignas(16) cute::uint64_t P_full_O_rescaled_mbar_ptr[2];
  alignas(16) cute::uint64_t S_full_mbar_ptr[2];
  alignas(16) cute::uint64_t O_full_mbar_ptr[2];
  alignas(16) cute::uint64_t softmax_corr_full_mbar_ptr[2];
  alignas(16) cute::uint64_t softmax_corr_empty_mbar_ptr[NUM_EPI_STAGE];
  alignas(16) cute::uint64_t corr_epi_full_mbar_ptr[NUM_EPI_STAGE];
  alignas(16) cute::uint64_t corr_epi_empty_mbar_ptr[2];
  alignas(16) cute::uint64_t s0_s1_sequence_mbar_ptr[8];
  alignas(16) cute::uint64_t tmem_dealloc_mbar_ptr[1];
  alignas(16) cute::uint64_t P_full_2_mbar_ptr[2];

  alignas(16) float sScale[NUM_Q_STAGE * QO_TILE_SIZE * 2];
  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sQ() {
    return cute::make_tensor(cute::make_smem_ptr(sQ.begin()), QSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sK() {
    return cute::make_tensor(cute::make_smem_ptr(sK.begin()), KVSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sV() {
    return cute::make_tensor(cute::make_smem_ptr(sK.begin()), KVSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sO() {
    return cute::make_tensor(cute::make_smem_ptr(sO.begin()), OSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sScale() {
    return cute::make_tensor(cute::make_smem_ptr(sScale), cute::make_shape(NUM_Q_STAGE * QO_TILE_SIZE * 2));
  }
  
};

} // namespace kernel
