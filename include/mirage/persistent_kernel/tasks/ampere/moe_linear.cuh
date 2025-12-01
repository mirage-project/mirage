#pragma once
// #include "element_binary.cuh"
// #include "element_unary.cuh"
// #include "reduction.cuh"
// #include "smem_layout.cuh"
// #include "tasks/common/common_header.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <type_traits>

#define MOE_NUM_THREADS 128
#define DEBUG_LOG 0

#define PRINT(x) do { \
    if(thread0()) { \
      printf("%s:", #x); \
      cute::print(x); \
      printf("\n"); \
    } \
} while(0)

#define PRINT_LAYOUT(x) do { \
  if(thread0()) { \
    printf("%s:", #x); \
    cute::print_layout(x); \
    printf("\n"); \
  } \
} while(0)

#define WARM_UP 0

#define DEBUG_EXPERT_IDX 95

constexpr int log2_constexpr(int n, int p = 0) {
  return (n <= 1) ? p : log2_constexpr(n >> 1, p + 1);
}

constexpr int batch_size = 8;
constexpr int experts_size = 128;
constexpr int activate_experts_size = 8;

// Matrix dimensions
const int m = batch_size;
const int k = 2048;
// const int n = 4096;
const int n = 1536;

const int expert_stride = 5;


namespace moe_config {
template <typename T_,
          int BATCH_SIZE_,
          int OUTPUT_SIZE_,
          int REDUCTION_SIZE_,
          int kTileM_,
          int kTileN_,
          int kTileK_,
          int kStage_,
          int kSmemLayoutCBatch_,
          typename ComputeType>
struct GemmConfig;
}

namespace moe_config {

using namespace cute;

template <typename T_,
          int BATCH_SIZE_,
          int OUTPUT_SIZE_,
          int REDUCTION_SIZE_,
          int kTileM_,
          int kTileN_,
          int kTileK_,
          int kStage_,
          int kSmemLayoutCBatch_ = 1,
          typename ComputeType = float>
struct GemmConfig {
  using T = T_;

  static constexpr int BATCH_SIZE = BATCH_SIZE_;
  static constexpr int OUTPUT_SIZE = OUTPUT_SIZE_;
  static constexpr int REDUCTION_SIZE = REDUCTION_SIZE_;
  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_ < OUTPUT_SIZE_ ? kTileN_ : OUTPUT_SIZE_;
  static constexpr int LoopN = (OUTPUT_SIZE_ + kTileN - 1) / kTileN;
  static constexpr int LoopM = (BATCH_SIZE_ + kTileM - 1) / kTileM;
  static constexpr int kTileK = kTileK_;
  static constexpr int kStage = kStage_;
  // TODO: add better way to determine PIPE_DEPTH
  // static constexpr int kStage = OUTPUT_SIZE_ < 128 ? 5 : 3;
  static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;
  static constexpr int BankMaxElemNum = 128 / sizeof(T);

  static constexpr int kShmLoadSwizzleM = 3;
  static constexpr int kShmLoadSwizzleS = 3;
  static constexpr int kShmLoadSwizzleB = 3;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
      make_layout(make_shape(Int<BATCH_SIZE>{}, Int<BankMaxElemNum>{}),
                  make_stride(Int<BankMaxElemNum>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

  using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 1;
  static constexpr int kMmaEURepeatN = 4;
  static constexpr int kMmaEURepeatK = 1;
  static constexpr int N_REG_REPEAT = kTileN / (kMmaEURepeatN * 8); // 8 means 8 in m16n8k16, we just process 8 cols in n dim.
  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN =
      N_REG_REPEAT * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
  // For MMA:
  // TiledMMA
  //   ThrLayoutVMNK:  (_32,_1,_4,_1):(_1,_0,_32,_0)
  //   PermutationMNK: (_16,_64,_16)
  // MMA_Atom
  //   ThrID:      _32:_1
  //   Shape_MNK:  (_16,_8,_16)
  //   LayoutA_TV: ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
  //   LayoutB_TV: ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
  //   LayoutC_TV: ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
  // The shape a stride for TV(thread value) is the offset/address in register, not in shared memory, all of them are col-majority.

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<8>{}, Int<16>{}),
                                           make_stride(Int<16>{}, Int<1>{})),
                              // TODO(Wenqin): maybe we can not use 'Int<8>` below? it should be related to BATCH_SIZE?
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;
  // The G2SCopyA is just in logical space with col-majority shape, it means how each load data in
  // logical coordinate, but not related to the real physical space, CUTE May adjact the load when
  // it really do the load with real physical tensor (col-majority or row-majority).

  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<BankMaxElemNum>{}),
                  make_stride(Int<BankMaxElemNum>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));
  // static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
  //                   size(SmemLayoutC{}),
  //               "C shared memory request is large than A's one pipe");
  static_assert(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) >=
                    size(SmemLayoutC{}),
                "C shared memory request is large than B's one pipe");

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<16>{}, Int<8>{}),
                                           make_stride(Int<8>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  
  using G2SCopyR =
      decltype(make_tiled_copy(S2GCopyAtomC{},
                               make_layout(make_shape(Int<16>{}, Int<8>{}),
                                           make_stride(Int<8>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));

  static constexpr int kThreadNum = size(MMA{});
  static_assert(kThreadNum == 128,
                "This config should use 4 warps (128 threads)");

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize =
      cute::max(shm_size_AB, shm_size_C) * sizeof(T);
};

} // namespace moe_config

namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR,
          bool NOBIAS,
          int PIPE_MAX = 3>
__device__ __forceinline__ void
    moe_linear_kernel(void const *input_ptr,
                void const *weight_ptr,
                void const *residual_ptr,
                void *output_ptr,
                void const *expert_routing_ptr,
                void const *expert_mask_ptr,
              int expert_offset) {
  constexpr int TILE_SIZE = 128;
  constexpr int kSmemLayoutCBatch = 1;

  using Config = moe_config::GemmConfig<T,
                                    BATCH_SIZE,
                                    OUTPUT_SIZE,
                                    REDUCTION_SIZE,
                                    16,
                                    128,
                                    TILE_SIZE,
                                    PIPE_MAX,
                                    kSmemLayoutCBatch,
                                    float>;

  using namespace cute;

  using SmemLayoutA = typename Config::SmemLayoutA;
  using SmemLayoutB = typename Config::SmemLayoutB;
  using SmemLayoutC = typename Config::SmemLayoutC;
  using TiledMMA = typename Config::MMA;

  using S2RCopyAtomA = typename Config::S2RCopyAtomA;
  using S2RCopyAtomB = typename Config::S2RCopyAtomB;
  using G2SCopyA = typename Config::G2SCopyA;
  using G2SCopyB = typename Config::G2SCopyB;
  using R2SCopyAtomC = typename Config::R2SCopyAtomC;
  using S2GCopyAtomC = typename Config::S2GCopyAtomC;
  using S2GCopyC = typename Config::S2GCopyC;
  using G2SCopyR = typename Config::G2SCopyR;
  

  constexpr int kTileM = Config::kTileM; // 16
  constexpr int kTileN = Config::kTileN; // 64
  constexpr int LoopM = Config::LoopM; // 1
  constexpr int LoopN = Config::LoopN; // 1
  constexpr int kTileK = Config::kTileK; // 128
  constexpr int kStage = Config::kStage; // 8

  extern __shared__ char smem[];

  T *shm_data = (T *)((reinterpret_cast<uintptr_t>(smem) + 127) / 128 * 128);

  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T const *__restrict__ d_residual = static_cast<T const *>(residual_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  int const *__restrict__ d_expert_routing = static_cast<int const*>(expert_routing_ptr);
  int const *__restrict__ d_expert_mask = static_cast<int const*>(expert_mask_ptr);

  // Below advance don't need in real MPK.
  int expert_offset_bid = blockIdx.y; // blockIdx.y is the real block id for offset inside an expert weight
  d_weight += OUTPUT_SIZE * REDUCTION_SIZE * expert_offset_bid;
  d_residual += OUTPUT_SIZE * expert_offset_bid;
  d_output += OUTPUT_SIZE * expert_offset_bid;

  // expert_offset = blockIdx.x;


  // auto bM = cute::tile_size<0>(
  //     TiledMMA); // MMA Tile M. We'll use 1 MMAs per MMA Tile M.
  // auto bN = cute::tile_size<1>(
  //     TiledMMA); // MMA Tile N. We'll use 1 MMAs per MMA Tile N.
  // auto bK = cute::tile_size<2>(TiledMMA) *
  //           cute::Int<4>{}; // MMA Tile K. We'll use 4 MMAs per MMA Tile K. For
  //                           // 16b types, wgmma has K16.

  // auto mma_tiler = cute::make_shape(bM, bN, bK); // (MMA_M, MMA_N, MMA_K)
  // auto cd_tiler =
  //     cute::make_shape(bN, bM, bK); // (MmaTile_N, MmaTile_M, MmaTile_K)
  // auto output_tiler =
  //     cute::make_shape(bN, bM, NUM_TOPK); // (MmaTile_N, MmaTile_M, NUM_TOPK)

  // (8,2048):(2048,_1)
  Tensor A = make_tensor(make_gmem_ptr((T *)d_input),
                         make_shape(BATCH_SIZE, REDUCTION_SIZE),
                         make_stride(REDUCTION_SIZE, Int<1>{}));
  // (128,64,2048):(3145728,2048,_1)
  Tensor B = make_tensor(make_gmem_ptr((T *)d_weight),
                         make_shape(NUM_EXPERTS, OUTPUT_SIZE, REDUCTION_SIZE),
                         make_stride(OUTPUT_STRIDE * REDUCTION_SIZE, REDUCTION_SIZE, Int<1>{}));
  // (128,8,64):(12288,1536,_1)
  // TODO(Wenqin): try to use NUM_TOPK as first dim for the output later.
  Tensor D = make_tensor(make_gmem_ptr((T *)d_output),
                         make_shape(NUM_TOPK, BATCH_SIZE, OUTPUT_SIZE),
                         make_stride(BATCH_SIZE * OUTPUT_STRIDE, OUTPUT_STRIDE, Int<1>{}));
  // (128,8,64):(12288,1536,_1)
  Tensor R = make_tensor(make_gmem_ptr((T *)d_residual),
                         make_shape(NUM_EXPERTS, BATCH_SIZE, OUTPUT_SIZE),
                         make_stride(BATCH_SIZE * OUTPUT_STRIDE, OUTPUT_STRIDE, Int<1>{}));

  // (128,8):(8,_1)
  Tensor mRoutingIndices = make_tensor(make_gmem_ptr((int *)d_expert_routing),
                         make_shape(NUM_EXPERTS, BATCH_SIZE),
                         make_stride(BATCH_SIZE, Int<1>{}));

  // (129):(_1)
  Tensor mMask = make_tensor(make_gmem_ptr((int *)d_expert_mask),
                         make_shape(NUM_EXPERTS + 1),
                         make_stride(Int<1>{}));

#if DEBUG_LOG
  PRINT(A);
  PRINT(B);
  PRINT(D);
  PRINT(R);

  PRINT(mRoutingIndices);
  PRINT(mMask);
#endif //DEBUG_LOG


//   cute::Tensor gA = cute::local_tile(
//       A, mma_tiler, mma_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
//   cute::Tensor gB = cute::local_tile(
//       B, mma_tiler, mma_coord, cute::Step<cute::X, cute::_1, cute::_1>{});
//   cute::Tensor gBias = cute::local_tile(
//       R, cd_tiler, mma_coord, cute::Step<cute::_1, cute::_1, cute::X>{});

  // create identity tensors for predicate
  auto cA = make_identity_tensor(shape(A)); // (m,k) -> (m,k)
  auto cB = make_identity_tensor(shape(B)); // (num_experts,n,k) -> (num_experts,n,k)
  auto cC = make_identity_tensor(shape(D)); // (num_experts,m,n) -> (num_experts,m,n)

  int num_activated_experts =
      mMask(NUM_EXPERTS); // last element stores num activated experts

#pragma unroll 1
  for (int activated_expert_offset = expert_offset;
        activated_expert_offset < num_activated_experts;
        activated_expert_offset += EXPERT_STRIDE) {
    int32_t expert_idx = mMask[activated_expert_offset];
    cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
    // if(blockIdx.y == 0) {
    //   printf("expert_idx: %d\n", expert_idx);
    // }
    // if(blockIdx.y == 0 && expert_idx == DEBUG_EXPERT_IDX && threadIdx.x == 0) {
    //   printf("expert_idx: %d, expert_offset: %d\n", expert_idx, expert_offset);
    // }
    // if(blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
    //   printf("expert_idx: %d, expert_offset: %d, first scalar: %f\n", expert_idx, expert_offset, float(B(expert_idx, 0, 0)));
    // }
#pragma unroll
    for (int m_iter = 0; m_iter < LoopM; ++m_iter) {
      // int m_iter = 0;
      // int n_iter = 0;
      // // slice the tensor to small one which is used for current thread block.
      Tensor gA = local_tile(
          A,
          make_tile(Int<kTileM>{}, Int<kTileK>{}),
          make_coord(
              m_iter,
              _)); // (kTileM, kTileK, m, k) (_16,_128,16):(2048,_1,_128)
      auto cta_cA = local_tile( // (_16,_128,16):(_1@0,_1@1,_128@1)
          cA, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(m_iter, _));
#if DEBUG_LOG
      if(activated_expert_offset == expert_offset &&
          m_iter == 0) {
        PRINT(gA);
        PRINT(cta_cA);
      }
#endif //DEBUG_LOG
#pragma unroll
      for (int n_iter = 0; n_iter < LoopN; ++n_iter) {
        Tensor gB = local_tile(
            B(expert_idx, _, _),
            make_tile(Int<kTileN>{}, Int<kTileK>{}),
            make_coord(n_iter, _)); // (kTileN, kTileK, n, k)
                                    // (_64,_128,16):(2048,_1,_128)
        // if(m_iter == 0 &&
        //     n_iter == 0 && blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
        //   printf("for gB, expert_idx: %d, first scalar: %f\n", expert_idx, float(gB(0,0,0)));
        // }
        Tensor gD = local_tile(
            // TODO(Wenqin): the gD at here is not what the real D we would
            // like to store data, it's just a logical tensor for us to get the
            // layout, we will use D to store the final data, maybe try to fix
            // it later.
            D(0, _, _),
            make_tile(Int<kTileM>{}, Int<kTileN>{}),
            make_coord(m_iter, n_iter)); // (kTileM, kTileN, m, n)
                                        // (_16,_64):(1536,_1)
        Tensor gR = local_tile(
            R(expert_idx, _, _),
            make_tile(Int<kTileM>{}, Int<kTileN>{}),
            make_coord(m_iter, n_iter)); // (kTileM, kTileN, m, n)
                                        // (_16,_64):(1536,_1)
#if DEBUG_LOG
        if(activated_expert_offset == expert_offset &&
            m_iter == 0 &&
            n_iter == 0) {
          PRINT(gB);
          PRINT(gD);
          PRINT(gR);
        }
#endif
        // shared memory
        // ((_8,_2),(_64,_2),(_1,_3)):((_64,_512),(_1,_1024),(_0,_2048))
        auto sA =
            make_tensor(make_smem_ptr(Ashm),
                        SmemLayoutA{}); // (kTileM, kTileK, kStage) (_16,_128,_3)
        // ((_8,_8),(_64,_2),(_1,_3)):((_64,_512),(_1,_4096),(_0,_8192))
        auto sB =
            make_tensor(make_smem_ptr(Bshm),
                        SmemLayoutB{}); // (kTileN, kTileK, kStage) (_64,_128,_3)

        auto cta_coord = make_coord(
            n_iter, m_iter, _); // make_coord(m_iter,_) / make_coord(n_iter,_)
        // ArithTuple(42,0,_0) o (_64,_128,16):(_1@1,_1@2,_128@2)
        // NOTE: it should be keep same shape as gB
        auto cta_cB = local_tile(cB(expert_idx, _, _),
                                  make_tile(Int<kTileN>{}, Int<kTileK>{}),
                                  make_coord(n_iter, _)); // same as gB
        // ArithTuple(42,0,0) o (_16,_64):(_1@1,_1@2)
        auto cta_cC = local_tile(cC(expert_idx, _, _),
                                  make_tile(Int<kTileM>{}, Int<kTileN>{}),
                                  make_coord(m_iter, n_iter)); // same as gD

#if DEBUG_LOG
        if(activated_expert_offset == expert_offset &&
            m_iter == 0 &&
            n_iter == 0) {
          PRINT(sA);
          PRINT(sB);
          PRINT(cta_cB);
          PRINT(cta_cC);
        }
#endif
        const int idx = threadIdx.x;

        TiledMMA tiled_mma;
        // TODO(Wenqin): what will happen when idx is greater than 32 for tensor A???
        auto thr_mma = tiled_mma.get_slice(idx);
        // ((_2,_2,_2),_1,_8):((_1,_2,_4),_0,_8)
        auto tCrA = thr_mma.partition_fragment_A(
            gA(_, _, 0)); // (MMA, MMA_M, MMA_K)  ((_2,_2,_2),_1,_8)
        // ((_2,_2),_2,_8):((_1,_2),_32,_4)
        auto tCrB = thr_mma.partition_fragment_B(
            gB(_, _, 0)); // (MMA, MMA_N, MMA_K) ((_2,_2),_2,_8)
        // ((_2,_2),_1,_2):((_1,_2),_0,_4)
        auto tCrD = thr_mma.partition_fragment_C(
            gD); // (MMA, MMA_M, MMA_N) ((_2,_2),_1,_2):((_1,_2),_0,_4)

#if DEBUG_LOG
        if(activated_expert_offset == expert_offset &&
            m_iter == 0 &&
            n_iter == 0) {
          // PRINT(tiled_mma);
          PRINT(thr_mma);
          PRINT(tCrA);
          PRINT(tCrB);
          PRINT(tCrD);
        }
#endif

        auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
        auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);

        using SmemLayoutAtomC = typename Config::SmemLayoutAtomC;
        // (_16,_64):(_64,_1)
        auto sR_init = make_tensor(sA.data(), SmemLayoutAtomC{});

        S2GCopyC s2g_tiled_copy_c;
        auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
        auto tCgR_s2g = s2g_thr_copy_c.partition_D(gR); // gR in global memory
        auto tCsR_s2g =
            s2g_thr_copy_c.partition_S(sR_init); // sR_init in shared memory
        // ((_8,_1),_1,_1):((_1@2,_0),_0,_0)
        auto tCcC = s2g_thr_copy_c.partition_S(cta_cC);

        G2SCopyR g2s_tiled_copy_r;

        //  (_8,_1):(_1,_0)
        auto tCpC = make_tensor<bool>(make_shape(size<0>(tCcC), Int<1>{}),
                                      make_stride(Int<1>{}, Int<0>{}));


#if DEBUG_LOG
        if(activated_expert_offset == expert_offset &&
            m_iter == 0 &&
            n_iter == 0) {
          // PRINT(tiled_mma);
          PRINT(R2SCopyAtomC{});
          PRINT(r2s_tiled_copy_c);
          PRINT(r2s_thr_copy_c);
          PRINT(SmemLayoutAtomC{});
          PRINT(sR_init);
          PRINT(s2g_tiled_copy_c);
          PRINT(s2g_thr_copy_c);
          PRINT(g2s_tiled_copy_r);
          PRINT(tCgR_s2g);
          PRINT(tCsR_s2g);
          PRINT(tCcC);
          PRINT(tCpC);
        }
#endif

        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCpC); ++i) {
          // TODO(Wenqin): the predicate just works for a group for 8(each 
          // thread load 8 elements) * 128 (threads) = 1024 elements? because
          // there is not fine-granularity control?
          tCpC(i, 0) =
              elem_less(tCcC(i, 0, 0), shape(D(0,_, _))); // Remove first dim for experts in D became (BATCH_SIZE, OUTPUT_SIZE)
        }

        if (!NOBIAS) {
          // TODO(Wenqin): The variable named with "s2g", but what we actually do here is "g2s",
          // figure the load shape from GMEM to SMEM and SMEM to registers later.
          cute::copy_if(g2s_tiled_copy_r, tCpC, tCgR_s2g, tCsR_s2g);
          __syncthreads();
          // load residual to accumulator registers
          // SMEM to register
          auto tCrD_r2s_view = r2s_thr_copy_c.retile_D(tCrD); // view of tCrD
          auto tCsR_r2s_view =
              r2s_thr_copy_c.partition_S(sR_init); // view of sR_init
          cute::copy(tCsR_r2s_view, tCrD_r2s_view);
          __syncthreads();
        } else {
          clear(tCrD);
        }

        auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
        auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
        // ((_8,_1),_1,((_2,_2),_2),(_1,_3)):((_1,_0),_0,((16,32),_1024),(_0,_2048))
        // To partition the physical layout for the shared memory to make it match what each thread want.
        auto tAsA = s2r_thr_copy_a.partition_S(
            sA); // ? (CPY, CPY_M, CPY_K, kStage) ((_8,_1),_1,((_2,_2),_2),(_1,_3))
        // ((_8,_1),_1,_8):((_1,_0),_0,_8)
        // A logical for registers, retile_D make it match the shared memory source layout.
        auto tCrA_view = s2r_thr_copy_a.retile_D(
            tCrA); // ? (CPY, CPY_M, CPY_K) ((_8,_1),_1,_8)

        auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
        auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
        // ((_8,_1),_1,((_2,_2),_2),(_1,_3)):((_1,_0),_0,((16,32),_4096),(_0,_8192))
        auto tBsB = s2r_thr_copy_b.partition_S(
            sB); // ? (CPY, CPY_N, CPY_K, kStage) ((_8,_1),_1,((_2,_2),_2),(_1,_3))
        // (((_4,_2),_1),_1,_8):(((_1,_32),_0),_0,_4)
        auto tCrB_view = s2r_thr_copy_b.retile_D(
            tCrB); // ? (CPY, CPY_N, CPY_K) (((_4,_2),_1),_1,_8)

#if DEBUG_LOG
        if(activated_expert_offset == expert_offset &&
            m_iter == 0 &&
            n_iter == 0) {
          PRINT(s2r_tiled_copy_a);
          PRINT(s2r_thr_copy_a);
          PRINT(tAsA);
          PRINT(tCrA_view);
          PRINT(s2r_tiled_copy_b);
          PRINT(s2r_thr_copy_b);
          PRINT(tBsB);
          PRINT(tCrB_view);
        }
#endif

        G2SCopyA g2s_tiled_copy_a;
        auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
        // ((_8,_1),_2,_1,16):((_1,_0),16384,_0,_128)
        // partition gA with tile for (16, 128), so CPY is 8 for each thread, 
        // CPY_M is 2 (2*8=16, we need to copy 16 rows, but 128 threads could juts copy 8 row in one iteration)
        // CPY_K is 1, because we don't need to repeat in K dim
        // k is 16 because 16*128=2048, so there is 16 tiles.
        auto tAgA_copy = g2s_thr_copy_a.partition_S(
            gA); // (CPY, CPY_M, CPY_K, k) ((_8,_1),_2,_1,16)
        // ((_8,_1),_2,_1,(_1,_3)):((_1,_0),_512,_0,(_0,_2048))
        auto tAsA_copy = g2s_thr_copy_a.partition_D(
            sA); // (CPY, CPY_M, CPY_K, kStage) ((_8,_1),_2,_1,(_1,_3))
        // ((_8,_1),_2,_1,16):((_1@1,_0),_8@0,_0,_128@1)
        auto tAcA = g2s_thr_copy_a.partition_S(cta_cA); // (CPY, CPY_M, CPY_K, k) ((_8,_1),_2,_1,16)

        G2SCopyB g2s_tiled_copy_b;
        auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
        // ((_8,_1),_8,_1,16):((_1,_0),16384,_0,_128)
        auto tBgB_copy = g2s_thr_copy_b.partition_S(
            gB); // (CPY, CPY_N, CPY_K, k) ((_8,_1),_8,_1,16)
        // ((_8,_1),_8,_1,(_1,_3)):((_1,_0),_512,_0,(_0,_8192))
        auto tBsB_copy = g2s_thr_copy_b.partition_D(
            sB); // (CPY, CPY_N, CPY_K, kStage) ((_8,_1),_8,_1,(_1,_3))
        // ((_8,_1),_8,_1,16):((_1@2,_0),_8@1,_0,_128@2)
        // each thread load (_8,_1) elements
        // In N dim, it will repeat 8 times
        // In K dim, it will not repeat
        // In k stage, it have 16 stages
        auto tBcB = g2s_thr_copy_b.partition_S(cta_cB); // (CPY, CPY_N, CPY_K, k) ((_8,_1),_8,_1,16)
        // only do predicate on m / n dimension, k dimension use stride-0
        // broadcast
        // (_2,_1):(_1,_0)
        auto tApA = make_tensor<bool>(
            make_shape(size<1>(tAcA),
                      Int<1>{}), // size: M sub-dimension per thread, K dimension
                                  // use 1, placeholder
            make_stride(Int<1>{}, Int<0>{}) // broadcast to K
        );
        
        // TODO(Wenqin): we don't use tBpB anymore, we need to add some
        // static_assert for this, or introduce tBpB again and fix some
        // correctness issue.
        // (_8,_1):(_1,_0)
        auto tBpB = make_tensor<bool>(make_shape(size<1>(tBcB), Int<1>{}),
                                      make_stride(Int<1>{}, Int<0>{}));

        // fill predicate: compare if the coordinate is in shape(A)/shape(B)
        CUTE_UNROLL
        for (int im = 0; im < size<0>(tApA); ++im) {
          // tAcA's coordinate is (CPY, CPY_M, CPY_K, tile_k) -> (m,k)
          tApA(im, 0) =
              elem_less(get<0>(tAcA(0, im, 0, 0)), shape<0>(A)); // m < M ?
        }
        CUTE_UNROLL
        for (int in = 0; in < size<0>(tBpB); ++in) {
          tBpB(in, 0) =
              elem_less(get<0>(tBcB(0, in, 0, 0)), shape<0>(B(0, _, _))); // n < N ?
          // if(m_iter == 0 &&
          //     n_iter == 0 && blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
          //   printf("in: %d, get<0>(tBcB(0, in, 0, 0): %d, shape<0>(B(0, _, _)): %d\n", in, get<0>(tBcB(0, in, 0, 0)), shape<0>(B(0, _, _)));
          // }
        }
        // if(m_iter == 0 &&
        //       n_iter == 0 && blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
        //   PRINT(tBcB);
        //   PRINT(tBcB(0, 0, 0, 0));
        // }

#if DEBUG_LOG
        if(activated_expert_offset == expert_offset &&
            m_iter == 0 &&
            n_iter == 0) {
          PRINT(g2s_tiled_copy_a);
          PRINT(g2s_thr_copy_a);
          PRINT(tAgA_copy);
          PRINT(tAsA_copy);
          PRINT(tAcA);
          PRINT(g2s_tiled_copy_b);
          PRINT(g2s_thr_copy_b);
          PRINT(tBgB_copy);
          PRINT(tBsB_copy);
          PRINT(tBcB);
          PRINT(tApA);
          PRINT(tBpB);
        }
#endif

        int itile_to_read = 0;
        int ismem_read_stage = 0;
        int ismem_write_stage = 0;

        // if(m_iter == 0 &&
        //     n_iter == 0 && blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
        //   printf("before load, for tBsB, expert_idx: %d, first scalar: %f\n", expert_idx, float(tBsB(0,0,0,0)));
        //   for (int in = 0; in < size<0>(tBpB); ++in) {
        //     printf("tBpB(%d, 0): %d\n", in, int(tBpB(in, 0)));
        //   }
        // }

      // warm up
#pragma unroll
        for (int istage = 0; istage < kStage - 1; ++istage) {
          // cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
          //           tAsA_copy(_, _, _, istage));
          // cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
          //           tBsB_copy(_, _, _, istage));
          // TODO(Wenqin): we don't use mRoutingIndices here, because we will use it as predicate later 
          // when we store the content back to GMEM, but is it a better solution if we just load valid
          // data at here? it may reduce the total data we load, but we may need some branches here,
          // that's a trade-off.
          cute::copy_if(g2s_tiled_copy_a,
                        tApA,
                        tAgA_copy(_, _, _, istage),
                        tAsA_copy(_, _, _, istage));
          cute::copy(g2s_tiled_copy_b,
                        tBgB_copy(_, _, _, istage),
                        tBsB_copy(_, _, _, istage));
          cp_async_fence();

          ++itile_to_read;
          ++ismem_write_stage;
        }

        // TODO: cp_async_wait later?
        cp_async_wait<kStage - 2>();
        __syncthreads();

        int ik = 0;
        cute::copy(s2r_tiled_copy_a,
                  tAsA(_, _, ik, ismem_read_stage),
                  tCrA_view(_, _, ik));
        cute::copy(s2r_tiled_copy_b,
                  tBsB(_, _, ik, ismem_read_stage),
                  tCrB_view(_, _, ik));

        constexpr int ntile = REDUCTION_SIZE / kTileK; // 2048 / 128 = 16 tiles for the whole K dim reduction
        constexpr int nk = size<2>(tCrA);              // 8 iteartions inside a tile

        // if(m_iter == 0 &&
        //     n_iter == 0 && blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
        //   printf("for tBsB, expert_idx: %d, first scalar: %f\n", expert_idx, float(tBsB(0,0,0,0)));
        // }

        // if(m_iter == 0 &&
        //     n_iter == 0 && blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
        //   printf("for tBsB_copy, expert_idx: %d, first scalar: %f\n", expert_idx, float(tBsB_copy(0,0,0,0)));
        // }

        // if(m_iter == 0 &&
        //     n_iter == 0 && blockIdx.y == 0 && expert_idx == 95 && threadIdx.x == 0) {
        //   printf("for tBgB_copy, expert_idx: %d, first scalar: %f\n", expert_idx, float(tBgB_copy(0,0,0,0)));
        // }

        // printf("11111\n");
        // if(blockIdx.y == 0 && expert_idx == DEBUG_EXPERT_IDX && threadIdx.x == 0) {
        //   printf("expert: %d's regs for A:\n", expert_idx);
        //   for (int i = 0; i < cute::size<0>(tCrA_view.layout()); i++) {
        //     auto v = tCrA_view(i, 0, ik);      // load element
        //     float fv = float(v);    // convert half/bfloat16 to float
        //     printf("%.4f ", fv);
        //   }
        //   printf("\n");
        // }

        // if(blockIdx.y == 0 && expert_idx == DEBUG_EXPERT_IDX && threadIdx.x == 0) {
        //   printf("expert: %d's regs for B:\n", expert_idx);
        //   for (int i = 0; i < cute::size<0>(tCrB_view.layout()); i++) {
        //     auto v = tCrB_view(i, 0, ik);      // load element
        //     float fv = float(v);    // convert half/bfloat16 to float
        //     printf("%.4f ", fv);
        //   }
        //   printf("\n");
        // }

#pragma unroll 1
        for (int itile = 0; itile < ntile; ++itile) {
#pragma unroll
          for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == 0) {
              if (itile_to_read < ntile) {
                // cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                //           tAsA_copy(_, _, _, ismem_write_stage));
                // cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                //           tBsB_copy(_, _, _, ismem_write_stage));
                cute::copy_if(g2s_tiled_copy_a,
                              tApA,
                              tAgA_copy(_, _, _, itile_to_read),
                              tAsA_copy(_, _, _, ismem_write_stage));
                cute::copy(g2s_tiled_copy_b,
                              tBgB_copy(_, _, _, itile_to_read),
                              tBsB_copy(_, _, _, ismem_write_stage));

                ++itile_to_read;
                ismem_write_stage = (ismem_write_stage + 1) % kStage;
              }

              cp_async_fence();
            }

            if (ik == nk - 1) {
              if(ntile - itile > 2) {
                cp_async_wait<kStage - 2>();
              } else {
                cp_async_wait<0>();
              }
              __syncthreads();

              ismem_read_stage = (ismem_read_stage + 1) % kStage;
            }

            cute::copy(s2r_tiled_copy_a,
                      tAsA(_, _, ik_next, ismem_read_stage),
                      tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b,
                      tBsB(_, _, ik_next, ismem_read_stage),
                      tCrB_view(_, _, ik_next));
            
            // if(expert_idx == 95) {
            //   if(float(tCrB(0, 0, ik)) != 95) {
            //     printf("reg is %f\n", float(tCrB(0, 0, ik)));
            //   }
            // }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
          } // ik
        }   // itile

        // NOTE: We don't need data in SMEM for A and B anymore, so we could just reuse them for C.
        // ((_16,_1),(_64,_1),(_1,_1)):((_64,_0),(_1,_0),(_0,_0))
        auto sC = make_tensor(sA(_, _, ismem_read_stage).data(),
                              SmemLayoutC{}); // ((_16,_1),(_64,_1),(_1,_1)):((_64,_0),(_1,_0),(_0,_0))

        // ((_2,_4),_1,_1):((_1,_2),_0,_0)
        auto tCrC_r2s = r2s_thr_copy_c.retile_S(
            tCrD); // (CPY, CPY_M, CPY_N) ((_2,_4),_1,_1):((_1,_2),_0,_0)
        // ((_2,(_2,_2)),_1,_1,(_1,_1)):((_1,(_512,32)),_0,_0,(_0,_0))
        // That's not a TV layout, but a tensor layout to describe where the register should be stored in,
        // it seems like a row-major tensor.
        auto tCsC_r2s = r2s_thr_copy_c.partition_D(
            sC); // (CPY, _1, _1, pipe)
                // ((_2,(_2,_2)),_1,_1,(_1,_1)):((_1,(_512,32)),_0,_0,(_0,_0))`

        // ((_8,_1),_1,_1,(_1,_1)):((_1,_0),_0,_0,(_0,_0))
        auto tCsC_s2g = s2g_thr_copy_c.partition_S(
            sC); // (CPY, _1, _1, pipe) ((_8,_1),_1,_1,(_1,_1)):((_1,_0),_0,_0,(_0,_0))
        // ((_8,_1),_1,_1):((_1,_0),_0,_0)
        auto tCgC_s2g = s2g_thr_copy_c.partition_D(
            gD); // (CPY, CPY_M, CPY_N) ((_8,_1),_1,_1):((_1,_0),_0,_0)

        // (_8,_1):(_1,_0)
        auto tCpC_ep = make_tensor<bool>(make_shape(size<0>(tCcC), Int<1>{}),
                                        make_stride(Int<1>{}, Int<0>{}));
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCpC_ep); ++i) {
          tCpC_ep(i, 0) = elem_less(tCcC(i, 0, 0), shape(D));
        }

        // ((_2,_4),_1,_1):((_1,_2),_0,_0)
        auto tC_tmp =
            make_tensor_like<T>(tCrC_r2s); // ((_2,_8),_1,_1):((_1,_2),_0,_0)

#if DEBUG_LOG
        if(activated_expert_offset == expert_offset &&
            m_iter == 0 &&
            n_iter == 0) {
          PRINT(sC);
          PRINT(tCrC_r2s);
          PRINT(tCsC_r2s);
          PRINT(tCsC_s2g);
          PRINT(tCgC_s2g);
          PRINT(tCpC_ep);
          PRINT(tC_tmp);
        }
#endif



        cute::copy(tCrC_r2s, tC_tmp);

        cute::copy(r2s_tiled_copy_c, tC_tmp, tCsC_r2s(_, _, _, 0));
        // ((_2,_4),_1,_1) -> ((_2,(_2,_2)),_1,_1)
        __syncthreads();

        // We couldn't use s2g_tiled_copy_c copy here, because its granularity is too big for use to
        // use the predicate, try to find a suitable way later.
        // cute::copy_if(s2g_tiled_copy_c, tCpC_ep, tCsC_s2g(_, _, _, 0), tCgC_s2g);
        constexpr int write_back_batch_size = kTileM < BATCH_SIZE ? kTileM : BATCH_SIZE;
        for(int i = threadIdx.x; i < write_back_batch_size * OUTPUT_SIZE; i += MOE_NUM_THREADS) {
          const int t = i / OUTPUT_SIZE;
          const int o = i % OUTPUT_SIZE;

          if(tRoutingIndex(t) != 0) {
            // Last 0 in sC just for pipeline stag, it should always be 0 in
            // sC, because we just store one stage in pipeline.
            // gD(b, o) = sC(b, o, 0);
            D(tRoutingIndex(t) - 1, t, o) = sC(t, o, 0);
          }
        }
        // TODO(Wenqin): do we really need the below sync?
        __syncthreads();
      } // n_iter
    } // m_iter
  }
}

} // namespace kernel