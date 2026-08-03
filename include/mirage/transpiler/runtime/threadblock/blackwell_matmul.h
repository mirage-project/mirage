// blackwell_matmul_pipeline.h
#pragma once

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <threadblock/input.h>
#include <threadblock/matmul.h>
#include <threadblock/utils.h>

using namespace cute;

#include "element_unary.h"

namespace tb {

template <typename T,
          bool IS_LDMATRIX_AVAIL,
          bool IS_STMATRIX_AVAIL,
          class SmemLayoutA_, // [K, M]
          class SmemLayoutB_, // [N, K]
          class SmemLayoutC_, // [N, M]
          int NUM_THREADS,
          int NUM_EXPS_BEFORE_STORE, // Since matmul may use some advanced
                                     // instructions (like stmatrix) to store
                                     // data, it does not use the standard
                                     // "epilogue" semantic
          bool IS_STORE_ACCUM,
          bool IS_COORPERATIVE,
          bool IS_PIPELINE_A,
          bool IS_PIPELINE_B,
          int PIPELINE_STAGES,
          class ClusterShape_MNK_,
          class TiledMMA_,
          class MmaTiler_MNK_,
          // swapAB: the caller computes C^T = B^T * A^T so a decode-shaped
          // token count (M = 1..8) lands in N, which tcgen05 allows at any
          // multiple of 8, while M takes the weight dim and stays 64/128. The
          // operand roles invert, so the majors and the tile_to_mma_shape Steps
          // invert with them -- leaving these hardcoded produced "Not a
          // canonical UMMA_MN Layout" because the smem atom was selected for
          // the wrong majorness.
          bool SWAP_AB = false,
          // TASK_BODY: this matmul is an MPK megakernel task body, not a
          // standalone kernel. There, blockIdx.x is the WORKER id (one block
          // per SM), not a tile index, and each task handles exactly one tile
          // with pointers the runtime has already offset. Deriving the MMA /
          // cluster coordinate from blockIdx therefore picks the wrong tile and
          // the result is wrong (rel ~1.5) while everything still compiles.
          bool TASK_BODY = false>
struct Blackwell_Matmul {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  using ClusterShape_MNK = ClusterShape_MNK_;
  using TiledMMA = TiledMMA_;
  using MmaTiler_MNK = MmaTiler_MNK_;

  static constexpr UMMA::Major UmmaMajorA =
      SWAP_AB ? UMMA::Major::MN : UMMA::Major::K;
  static constexpr UMMA::Major UmmaMajorB =
      SWAP_AB ? UMMA::Major::K : UMMA::Major::MN;
  // CUTLASS picks the Step by majorness: is_mn_major ? Step<_2,_1,_3> :
  // Step<_1,_2,_3> (see sm100_mma_warpspecialized.hpp).
  using StepA =
      cute::conditional_t<SWAP_AB, Step<_2, _1, _3>, Step<_1, _2, _3>>;
  using StepB =
      cute::conditional_t<SWAP_AB, Step<_1, _2, _3>, Step<_2, _1, _3>>;

  static constexpr int PIPELINE_STAGE_A = IS_PIPELINE_A ? PIPELINE_STAGES : 1;
  static constexpr int PIPELINE_STAGE_B = IS_PIPELINE_B ? PIPELINE_STAGES : 1;

  // The MMA tile spans all CTAs of the MMA atom (2 for a 2x1SM atom), but the
  // smem layout atom describes what a *single* CTA holds. Selecting it from the
  // undivided MmaTiler produced an atom twice the shape it is later tiled into,
  // failing with "tile_to_shape: block shape does not divide the target shape".
  // CUTLASS divides by size(AtomThrID) for exactly this reason -- see
  // SmemShape_M/SmemShape_N in
  // cutlass/gemm/collective/builders/sm100_umma_builder.inl.
  using AtomThrSize = decltype(size(typename TiledMMA::AtomThrID{}));
  using SmemShape_M =
      decltype(shape_div(get<0>(MmaTiler_MNK{}), AtomThrSize{}));
  using SmemShape_N =
      decltype(shape_div(get<1>(MmaTiler_MNK{}), AtomThrSize{}));

  using SmemLayoutAtom_A =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
               UmmaMajorA,
               T,
               SmemShape_M,
               decltype(get<2>(MmaTiler_MNK{}))>());
  using DstMNKLayout_A = decltype(partition_shape_A(
      TiledMMA{},
      make_shape(shape<0>(MmaTiler_MNK{}), shape<2>(MmaTiler_MNK{}))));

  using DstPipeLayout_A = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtom_A{},
      append(DstMNKLayout_A{}, Int<PIPELINE_STAGE_A>{}),
      StepA{}));

  using SmemLayoutAtom_B =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
               UmmaMajorB,
               T,
               SmemShape_N,
               decltype(get<2>(MmaTiler_MNK{}))>());
  using DstMNKLayout_B = decltype(partition_shape_B(
      TiledMMA{},
      make_shape(shape<1>(MmaTiler_MNK{}), shape<2>(MmaTiler_MNK{}))));

  using DstPipeLayout_B = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtom_B{},
      append(DstMNKLayout_B{}, Int<PIPELINE_STAGE_B>{}),
      StepB{}));

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K]
  using SmemLayoutB = SmemLayoutB_;                                // [N, K]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N]

  using N = decltype(get<0>(shape(SmemLayoutB{})));

  // NOTE: on the NON-pipelined path the transpiler's own swizzled layout
  // (SmemLayoutA_/B_) is what the G->S copy writes, while the UMMA reads
  // through DstPipeLayout_A/B above. (A pipelined operand is written by TMA
  // through DstPipeLayout too, so the two agree by construction there -- which
  // is why only the non-pipelined path is limited to a 128B tile pitch.)
  //
  // On that path the two agree only because the transpiler emits the
  // planner's computed swizzle (Swizzle<3,3,3>, the UMMA's effective
  // element-unit swizzle -- one bit of chunk granularity off and every
  // matmul returns right values at wrong positions).
  //
  // Do not reconstruct the read layout by composing DstPipeLayout with a
  // linearizing layout: that drops the smem_ptr_flag[16b] bit-granularity
  // semantics and silently yields different addresses.

  static constexpr int global_N = size<1>(ClusterShape_MNK{}) * N{};
  using GmemStrideTypeC = Stride<Int<global_N>, Int<1>>;


  // Use the sm100_get_tmem_load_op function to automatically select the optimal
  // tmem load operation
  // CUTLASS 4.2.1 signature is
  //   sm100_get_tmem_load_op<GmemStrideTypeD, ElementAccumulator, ElementD,
  //                          TmemShape_MN, bool IsBlockScaleSupported>
  // ElementAccumulator must be float: tcgen05 accumulates in fp32 and this
  // argument drives num_col_bits in the op selection. (The 4.2.1 signature
  // takes a bool as its last argument, not the fusion op older releases
  // took.)
  using TMemLoadOp =
      decltype(cutlass::epilogue::collective::detail::sm100_get_tmem_load_op<
               GmemStrideTypeC,
               float,
               T,
               // Per-warp TMEM subpartition shape. CUTLASS derives this as
               // shape_div(EpilogueTileShape_MN, TmemWarpShape_MN) with
               // TmemWarpShape = (4,1) -- see sm100_builder.inl. Both extents
               // must follow the tile; fixing either (e.g. hardcoding
               // columns to one N or rows to M=128/4) breaks the R2S copy at
               // other shapes ("AtomTVLayout does not exist").
               Shape<decltype(shape_div(SmemShape_M{}, Int<4>{})), SmemShape_N>,
               false>());

  using SMemStoreOp =
      decltype(cutlass::epilogue::collective::detail::sm100_get_smem_store_op<
               GmemStrideTypeC,
               T,
               T,
               TMemLoadOp>());

  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  // tmem_base_ptr is taken BY VALUE so the caller can pass `*base + offset`:
  // a body with several matmuls must give each accumulator its own TMEM
  // columns. Sharing one base made the second matmul overwrite the first's
  // accumulator -- two matmuls in one graph came back at rel ~2.4.
  static __device__ __forceinline__ auto
      get_mma_tC(int blockIdx_x, int blockIdx_y, uint32_t tmem_base_ptr) {
    // See TASK_BODY above: a task body is one CTA's worth of work.
    auto cta_mma = get_cta_mma<TiledMMA, ClusterShape_MNK>(
        TASK_BODY ? 0 : blockIdx.x, TASK_BODY ? 0 : blockIdx.y);

    Tensor dummy_sC = make_tensor(make_smem_ptr((T *)nullptr), SmemLayoutC{});
    auto tCsC = cta_mma.partition_C(dummy_sC);

    Tensor tCtAcc = cta_mma.make_fragment_C(tCsC);
    tCtAcc.data() = tmem_base_ptr;

    return tCtAcc;
  }

  static __device__ __forceinline__ auto get_mma_rC(int blockIdx_x,
                                                    int blockIdx_y) {
    // Make a fake tensor

    Tensor dummy_sC = make_tensor(make_smem_ptr((T *)nullptr), SmemLayoutC{});

    TiledMMA tiled_mma;
    auto mma_coord_vmnk =
        get_mma_coord_vmnk<TiledMMA, ClusterShape_MNK>(blockIdx_x, blockIdx_y);
    auto mma_v = get<0>(mma_coord_vmnk);
    auto cta_mma = tiled_mma.get_slice(mma_v);

    auto tCsC = cta_mma.partition_C(dummy_sC);

    clear(tCsC);
    return tCsC;
  }

  // write from tensor memory to smem
  template <class TmemAccTensor>
  static __device__ __forceinline__ void write_tC_to_sC(
      T *__restrict__ s_ptr, TmemAccTensor const &tCtAcc, int thread_idx) {
    // only one warp group is used for Tmem load
    if (thread_idx >= mirage::config::NUM_THREADS_PER_GROUP) {
      return;
    }

    TiledCopy tiled_t2r_copy = make_tmem_copy(TMemLoadOp{}, tCtAcc);
    ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

    auto mma_v = _0{}; // if write to smem, no need to use peer id

    TiledMMA tiled_mma;
    auto cta_mma = tiled_mma.get_slice(mma_v);
    auto sC = make_tensor(make_smem_ptr(s_ptr), SmemLayoutC{});

    auto tCsC = cta_mma.partition_C(
        sC); // (MmaC, NumMma_M, NumMma_N) MmaC is half of cta_mma.Mma
    auto tDsC = thr_t2r_copy.partition_D(tCsC); // (CpyD, NumCpy_M, NumCpy_N)
    auto tDtAcc =
        thr_t2r_copy.partition_S(tCtAcc); // (CpyS, NumCpy_M, NumCpy_N)
    // TMEM holds fp32 accumulators (the TiledMMA's ElementAccumulator), and the
    // TMEM_LOAD op requires the destination register fragment to have that same
    // element type. Loading straight into a T (bf16) fragment made CUTLASS
    // reject the copy with "Expected src to have the specific TMEM layout
    // required by CopyOp" / "dst layout doesn't vectorize into registers".
    // Load as fp32, then narrow to T for the register->smem store.
    auto tDrAcc_f32 =
        make_tensor<float>(shape(tDsC)); // (CpyD, NumCpy_M, NumCpy_N)

    // Load TMEM -> RMEM
    copy(tiled_t2r_copy, tDtAcc, tDrAcc_f32);
    // tcgen05.ld is ASYNC: without waiting for it, a caller that immediately
    // issues the next MMA into the same accumulator (a chained matmul's next
    // iteration) can clobber TMEM mid-load. CUTLASS epilogues fence after
    // every TMEM load for exactly this reason.
    cutlass::arch::fence_view_async_tmem_load();

    auto tDrAcc = make_tensor<T>(shape(tDsC));
    CUTE_UNROLL
    for (int i = 0; i < size(tDrAcc_f32); ++i) {
      float x = tDrAcc_f32(i);
      // The elementwise op the scheduler fused into this matmul's epilogue
      // (e.g. the exp of Q@K^T -> exp -> @V). This parameter was accepted and
      // silently IGNORED here -- the Ampere/Hopper matmul applies it in its
      // store epilogue (matmul.h), Blackwell's TMEM epilogue did not, so the
      // fused op simply vanished from the kernel.
      if constexpr (NUM_EXPS_BEFORE_STORE > 0) {
        CUTE_UNROLL
        for (int e = 0; e < NUM_EXPS_BEFORE_STORE; ++e) {
          x = perform_element_unary_op<float, ElementUnaryOpType::EXP>(x);
        }
      }
      tDrAcc(i) = static_cast<T>(x);
    }

    TiledCopy tiled_r2s_copy =
        make_tiled_copy_D(Copy_Atom<SMemStoreOp, T>{}, tiled_t2r_copy);
    ThrCopy thread_r2s = tiled_r2s_copy.get_slice(thread_idx);
    auto r2s_rC = thread_r2s.retile_S(tDrAcc);
    auto r2s_sC = thread_r2s.partition_D(tCsC);

    copy(tiled_r2s_copy, r2s_rC, r2s_sC);
  }

  // a_ptr, b_ptr are from smem, mma_tC is from tmem.
  //
  // Pipelined form. The pipeline objects are not used by the MMA itself -- only
  // `read_stage` is, to index the pipelined smem layout -- so this simply
  // forwards to the core implementation below.
  template <class TmemAccTensor,
            class BlackwellAsyncPipeline_A,
            class BlackwellAsyncPipeline_B>
  static __device__ __forceinline__ void run(TmemAccTensor &mma_tC,
                                             T *__restrict__ a_ptr,
                                             T *__restrict__ b_ptr,
                                             int k_iter,
                                             TiledMMA tiled_mma,
                                             int read_stage,
                                             BlackwellAsyncPipeline_A &,
                                             BlackwellAsyncPipeline_B &) {
    run(mma_tC, a_ptr, b_ptr, k_iter, tiled_mma, read_stage);
  }

  // Non-pipelined form, used when the operands are loaded with plain
  // synchronous copies instead of a TMA pipeline. This is the shape an MPK task
  // body has: a single tile, no pipelining. With IS_PIPELINE_A/B false the
  // DstPipeLayouts carry a single stage, so read_stage is always 0.
  template <class TmemAccTensor>
  static __device__ __forceinline__ void run(TmemAccTensor &mma_tC,
                                             T *__restrict__ a_ptr,
                                             T *__restrict__ b_ptr,
                                             int k_iter,
                                             TiledMMA tiled_mma,
                                             int read_stage) {
    run(mma_tC, a_ptr, b_ptr, k_iter, tiled_mma, read_stage, read_stage);
  }

  // Per-operand read stages: a matmul may MIX a pipelined operand (TMA-fed,
  // stage advances with the loop) with a non-pipelined one (loaded once,
  // stage extent 1, so its index must stay 0). Q @ K_i^T in attention is
  // exactly this -- Q is loaded once, K is a pipelined tile. Indexing the
  // non-pipelined operand's single-stage layout with the shared read_stage
  // read past its extent.
  template <class TmemAccTensor>
  static __device__ __forceinline__ void run(TmemAccTensor &mma_tC,
                                             T *__restrict__ a_ptr,
                                             T *__restrict__ b_ptr,
                                             int k_iter,
                                             TiledMMA tiled_mma,
                                             int read_stage_a,
                                             int read_stage_b) {
    if (warp_id() == 0) {
      if (k_iter == 0) {
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
      }

      auto mma_coord_vmnk = get_mma_coord_vmnk<TiledMMA, ClusterShape_MNK>(
          TASK_BODY ? 0 : blockIdx.x, TASK_BODY ? 0 : blockIdx.y);

      auto mma_v = get<0>(mma_coord_vmnk);
      auto cta_mma = tiled_mma.get_slice(mma_v);

      Tensor tCsA = make_tensor(make_smem_ptr(a_ptr), DstPipeLayout_A{});
      Tensor tCsB = make_tensor(make_smem_ptr(b_ptr), DstPipeLayout_B{});

      Tensor tCrA = cta_mma.make_fragment_A(tCsA);
      Tensor tCrB = cta_mma.make_fragment_B(tCsB);

      // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma,
             tCrA(_, _, k_block, read_stage_a),
             tCrB(_, _, k_block, read_stage_b),
             mma_tC);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
    }
  }
};

} // namespace tb
