#include <iostream>
#include <cstdio>

// Use Thrust to handle host/device allocations
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Cutlass includes
#include <cutlass/half.h>                       // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                      // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>           // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>   // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp>  // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>   // TMEM allocator for SM100
// using namespace cute;

namespace kernel {

  // The shared memory buffers for A, B, C, and D matrices.
  template <class TypeA,           // Tensor A data type
            class TypeB,           // Tensor B data type
            class ASmemLayout,     // (MmaA, NumMma_M, NumMma_K, ...)
            class BSmemLayout>     // (MmaB, NumMma_N, NumMma_K, ...)
  struct SharedStorage
  {
    alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
    alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

    alignas(16) cute::uint64_t mma_barrier;  // Barrier to track MMA computation on SMEM
    alignas(16) cute::uint64_t tma_barrier;  // Barrier to track TMA data transfers to SMEM

    alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

    CUTE_DEVICE constexpr auto tensor_sA() { return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{}); }
    CUTE_DEVICE constexpr auto tensor_sB() { return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{}); }
  };

  template <class SharedStorage,
          class ATensor, class BTensor, class CTensor, class DTensor,
          class MmaTiler_MNK, class TiledMMA, class ClusterShape_MNK,
          class TmaAtomA, class TmaAtomB>
  __device__ __forceinline__ void linear_kernel_sm100(
            ATensor mA,                      // (Gemm_M, Gemm_K)
            BTensor mB,                      // (Gemm_N, Gemm_K)
            CTensor mC,                      // (Gemm_M, Gemm_N)
            DTensor mD,                      // (Gemm_M, Gemm_N)
            MmaTiler_MNK mma_tiler,          // <MmaTile_M, MmaTile_N, MmaTile_K>
            TiledMMA tiled_mma,              // <    Mma_M,     Mma_N,     Mma_K>
            ClusterShape_MNK cluster_shape,  // (ClusterM, ClusterN, ClusterK)
            TmaAtomA const* tma_atom_A,
            TmaAtomB const* tma_atom_B
  ) {
    // Step 1: The Prologue.

    // The CTA layout within the Cluster: (V,M,N,K) -> CTA idx
    cute::Layout cluster_layout_vmnk = cute::tiled_divide(cute::make_layout(cluster_shape),
                                              cute::make_tile(typename TiledMMA::AtomThrID{}));

    // Construct the MMA grid coordinate from the CTA grid coordinate
    auto mma_coord_vmnk = cute::make_coord(blockIdx.x % cute::size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                                    cute::_, //    MMA-M coordinate
                                    cute::_, //    MMA-N coordinate
                                    cute::_);//    MMA-K coordinate

    // Partition the GMEM tensors with the mma_tiler and mma_coord to get the slices processed
    //   by this mma tile.
    // CuTe provides local_tile partitioning function. local_tile accepts 4 parameters:
    //   * Tensor to partition
    //   * Tiler to use for partitioning
    //   * Coordinate to use for slicing the partitioned tensor
    //   * Projection to ignore unwanted modes of the Tiler and Coordinate
    auto mma_coord = cute::select<1,2,3>(mma_coord_vmnk);
    cute::Tensor gA = cute::local_tile(mA, mma_tiler, mma_coord, cute::Step<cute::_1,cute::X,cute::_1>{});  // (MmaTile_M, MmaTile_K, Tiles_K)
    cute::Tensor gB = cute::local_tile(mB, mma_tiler, mma_coord, cute::Step<cute::X,cute::_1,cute::_1>{});  // (MmaTile_N, MmaTile_K, Tiles_K)
    cute::Tensor gC = cute::local_tile(mC, mma_tiler, mma_coord, cute::Step<cute::_1,cute::_1,cute::X>{});  // (MmaTile_M, MmaTile_N)
    cute::Tensor gD = cute::local_tile(mD, mma_tiler, mma_coord, cute::Step<cute::_1,cute::_1,cute::X>{});  // (MmaTile_M, MmaTile_N)

    // The SMEM tensors

    // Allocate SMEM
    extern __shared__ char shared_memory[];
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

    // Represent the SMEM buffers for A and B
    cute::Tensor tCsA = shared_storage.tensor_sA();         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
    cute::Tensor tCsB = shared_storage.tensor_sB();         // (MmaB, NumMma_M, NumMma_K, Tiles_K)

    //
    // Mma partitioning for A and B
    //

    auto mma_v = cute::get<0>(mma_coord_vmnk);
    cute::ThrMMA cta_mma = tiled_mma.get_slice(mma_v);   // Use Peer CTA coordinate
    cute::Tensor tCgA = cta_mma.partition_A(gA);         // (MmaA, NumMma_M, NumMma_K, Tiles_K)
    cute::Tensor tCgB = cta_mma.partition_B(gB);         // (MmaB, NumMma_N, NumMma_K, Tiles_K)
    cute::Tensor tCgC = cta_mma.partition_C(gC);         // (MmaC, NumMma_M, NumMma_N)
    cute::Tensor tCgD = cta_mma.partition_C(gD);         // (MmaC, NumMma_M, NumMma_N)

    // MMA Fragment Allocation
    // We allocate "fragments" which are SMEM descriptors that serve as inputs to cute::gemm operations.
    // For tcgen05.mma operations:
    // - Matrices A and B are sourced from SMEM
    // - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
    // - The first mode of each descriptor represents the SMEM for a single MMA operation
    cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);      // (MmaA, NumMma_M, NumMma_K, Tiles_K)
    cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);      // (MmaB, NumMma_M, NumMma_K, Tiles_K)

    // TMEM Allocation
    // On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
    // ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate layout for the accumulator.
    cute::Tensor tCtAcc = cta_mma.make_fragment_C(tCgC(cute::_, cute::_, cute::_, 0, 0));    // (MmaC, NumMma_M, NumMma_N)

    uint32_t elect_one_thr  = cute::elect_one_sync();
    uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};

    if (elect_one_warp) {
      tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
    }
    __syncthreads(); // Wait for all threads until warp0 allocates TMEM
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    // TMA Setup
    //
    //   These are TMA partitionings, which have a dedicated custom partitioner.
    //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
    //      Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
    //   For A tensor: The group_modes<0,3> transforms the (MmaA, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
    //      into ((MmaA, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile MK.
    //   For B tensor: The group_modes<0,3> transforms the (MmaB, NumMma_M, NumMma_K, Tiles_K)-shaped tensor
    //      into ((MmaB, NumMma_M, NumMma_K), Tiles_K). The partitioning only pays attention to mode-0, the MMA Tile NK.
    //   Simply put, the TMA will be responsible for everything in mode-0 with a single call to cute::copy.
    //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.

    auto [tAgA, tAsA] = cute::tma_partition(*tma_atom_A,
                                      cute::Int<0>{}, cute::Layout<cute::_1>{},
                                      cute::group_modes<0,3>(tCsA), cute::group_modes<0,3>(tCgA));

    auto [tBgB, tBsB] = cute::tma_partition(*tma_atom_B,
                                      cute::Int<0>{}, cute::Layout<cute::_1>{},
                                      cute::group_modes<0,3>(tCsB), cute::group_modes<0,3>(tCgB));

    // Calculate total bytes that TMA will transfer each tile to track completion
    int tma_transaction_bytes = sizeof(cute::make_tensor_like(tAsA))
                              + sizeof(cute::make_tensor_like(tBsB));


    // Barrier Initialization
    // Barriers in SMEM initialized by a single thread.
    if (elect_one_warp && elect_one_thr) {
      cute::initialize_barrier(shared_storage.mma_barrier, /* num_ctas */ 1);
      cute::initialize_barrier(shared_storage.tma_barrier, /* num_threads */ 1);
    }
    int mma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
    int tma_barrier_phase_bit = 0;  // Each barrier has an associated phase_bit.
    __syncthreads();                // Make sure all threads observe barrier initialization.

    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile){
      for (int n_tile =0; n_tile < cute::size<3>(tCgB); ++n_tile){
        // Step 2: The Mainloop.

        // Set mma accumulate option to zero so that the first MMA instruction will clear the TMEM accumulator.
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
        for (int k_tile = 0; k_tile < cute::size<4>(tCgA); ++k_tile)
        {
          // Step 2a: Load A and B tiles

          // TMA Load Operations:
          // - Execute asynchronous TMA loads with single thread
          // - Set transaction bytes and execute with barrier
          if (elect_one_warp && elect_one_thr) {
            cute::set_barrier_transaction_bytes(shared_storage.tma_barrier, tma_transaction_bytes);
            cute::copy(tma_atom_A->with(shared_storage.tma_barrier), tAgA(cute::_, m_tile, k_tile), tAsA); // Load MmaTile_M x MmaTile_K A tile
            cute::copy(tma_atom_B->with(shared_storage.tma_barrier), tBgB(cute::_, n_tile, k_tile), tBsB); // Load MmaTile_N x MmaTile_K B tile
            // cute::print("tAgA[0]:\t"); cute::print(tAgA(0)); cute::print("\n");
          }

          // Step 2b: Execute the MMAs for this tile

          // Wait for TMA loads to SMEM to complete
          cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
          tma_barrier_phase_bit ^= 1;

          // tcgen05.mma instructions require single-thread execution:
          // - Only one warp performs the MMA-related loop operations
          // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
          // - No explicit elect_one_sync region is needed from the user
          if (elect_one_warp) {
            // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
            for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
              cute::gemm(tiled_mma, tCrA(cute::_,cute::_,k_block), tCrB(cute::_,cute::_,k_block), tCtAcc);
              tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            }
            // Ensure MMAs are completed, only then we can reuse the A and B SMEM.
            cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
          }
          // Wait MMAs to complete to avoid overwriting the A and B SMEM.
          cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
          mma_barrier_phase_bit ^= 1;
        }

        // Step 3: The Epilogue.

        // Create the tiled copy operation for the accumulator (TMEM -> RMEM)
        cute::TiledCopy tiled_t2r_copy = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
        cute::ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);

        cute::Tensor tDgC = thr_t2r_copy.partition_D(tCgC(cute::_, cute::_, cute::_, m_tile, n_tile));                   // (CpyD, NumCpy_M, NumCpy_N)
        cute::Tensor tDrC = cute::make_fragment_like(tDgC);                   // (CpyD, NumCpy_M, NumCpy_N)
        // Load C tensor GMEM -> RMEM
        cute::copy(tDgC, tDrC);

        cute::Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);               // (CpyS, NumCpy_M, NumCpy_N)
        cute::Tensor tDgD   = thr_t2r_copy.partition_D(tCgD(cute::_, cute::_, cute::_, m_tile, n_tile));                 // (CpyD, NumCpy_M, NumCpy_N)
        using AccType = typename decltype(tCtAcc)::value_type;
        cute::Tensor tDrAcc = cute::make_tensor<AccType>(cute::shape(tDgD));              // (CpyD, NumCpy_M, NumCpy_N)
        // Load TMEM -> RMEM
        cute::copy(tiled_t2r_copy, tDtAcc, tDrAcc);

        // AXPBY RMEM -> RMEM: tDrC = alpha * tDrAcc + beta * tDrC
        cute::axpby(1, tDrAcc, 1, tDrC);
        // Store RMEM -> GMEM
        cute::copy(tDrC, tDgD);

        __syncthreads();
      } // for n_tile
    }

    

    // Release the right to allocate before deallocations so that the next CTA can rasterize
    // Then deallocate TMEM
    if (elect_one_warp) {
      tmem_allocator.release_allocation_lock();
      tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

  } // end linear_kernel_sm100



} // namespace kernel

