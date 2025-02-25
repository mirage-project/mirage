// cluster.h - Implementation of threadblock block level cluster operators
//
#pragma once

#include <cstdio>
#include <cuda/ptx>
#include <cuda/barrier>
#include <cooperative_groups.h>

namespace tb {

using namespace cooperative_groups;

// utils
static __device__ __forceinline__ void cluster_sync() {
  // arrive & wait, use align/aquire?
  // asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
  // asm volatile("barrier.cluster.arrive.aligned;\n" : :);
}

static __device__ __forceinline__ uint32_t _ptr_to_int32(void const* const ptr){
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}


// mbarrier init
static __device__ __forceinline__ void
    mbarrier_init(uint64_t *__addr,
                  const uint32_t &__count) {
  asm("mbarrier.init.shared.b64 [%0], %1;"
      :
      : "r"(_ptr_to_int32(__addr)), "r"(__count)
      : "memory");
}

// threads participant in computation
static __device__ __forceinline__ uint64_t
    mbarrier_arrive_expect(uint64_t *__addr,
                           const uint32_t &__tx_count) {
  uint64_t __state;
   asm("mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 %0, [%1], %2; // 8. "
        : "=l"(__state)
        : "r"(_ptr_to_int32(__addr)), "r"(__tx_count)
        : "memory");
  return __state;
}

// // threads not participant in computation
// static __device__ __forceinline__ void
//     mbarrier_arrive(uint64_t *__addr,
//                     const uint32_t &__tx_count) {
//   uint64_t __state;
//   // scope cluster
//   asm("mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  [%1];           // 3a. "
//         : "=l"(__state)
//         : "r"(_ptr_to_int32(__addr))
//         : "memory");
// }

// put a value to remote buffer in same cluster
static __device__ __forceinline__ void cluster_put_float(
    half_t *__addr, half_t const &__value, uint64_t *__remote_bar)
    {
  asm("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], "
      "%1, [%2];    // 1. "
      :
      : "r"(_ptr_to_int32(__addr)),
        "r"(/*as_b32*/ *reinterpret_cast<const int32_t *>(
            &__value)),
        "r"(_ptr_to_int32(__remote_bar))
      : "memory");
}

static __device__ __forceinline__ bool
    mbarrier_try_wait(uint64_t *__addr,
                      const uint64_t &__state) {
  uint32_t __waitComplete;
  asm("{\n\t .reg .pred P_OUT; \n\t"
        "mbarrier.try_wait.acquire.cluster.shared::cta.b64         P_OUT, [%1], %2;                        // 6a. \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__waitComplete)
        : "r"(_ptr_to_int32(__addr)), "l"(__state)
        : "memory");
  return static_cast<bool>(__waitComplete);
}

// // Set the destination block-ID in cluster for a given SMEM Address
// static __device__ __forceinline__ uint32_t set_block_rank(uint64_t smemAddr,
//                                                           uint32_t rank) {
//   uint32_t result;
//   asm volatile("mapa.shared::cluster.u64  %0, %1, %2;\n"
//                : "=r"(result)
//                : "r"(smemAddr), "r"(rank));
//   return result;
// }

// reduction
template <typename T, int CLUSTER_SIZE, int NUM_THREADS, int BUF_SIZE>
class ClusterReduction {

public:

  static constexpr int THREAD_COUNT = BUF_SIZE / NUM_THREADS;
  

  using barrier_t = cuda::barrier<cuda::thread_scope_block>;

  static __device__ __forceinline__ void run(uint64_t* bar_i,
                                        T * __restrict__ dst,
                                        T *const __restrict__ src,
                                        int block_rank,
                                        int thread_idx) {
    // using cuda::ptx::sem_release;
    // using cuda::ptx::sem_acquire;
    // using cuda::ptx::space_cluster;
    // using cuda::ptx::space_shared;
    // using cuda::ptx::scope_cluster;    


  static __device__ __forceinline__ void run(uint64_t* bar,
                                        T * __restrict__ dst,
                                        T *const __restrict__ src,
                                        int block_rank,
                                        int thread_idx) {
                                      
    auto cluster = this_cluster();   
                                  
    // init barrier, all threads participant
    mbarrier_init(bar, blockDim.x * blockDim.y * blockDim.z);

    // cluster sync
    cluster.sync();


    uint64_t arrival_token;

    if (block_rank != 0) {
      unsigned int block0rank = 0;
      // uint64_t *remote_bar = set_block_rank(bar, block0rank);
      uint64_t *remote_bar = cluster.map_shared_rank(bar, block0rank);
      T *remote_receive_buffer =  cluster.map_shared_rank(dst, block0rank);
      // T *remote_receive_buffer = set_block_rank(dst, block0rank);


      arrival_token = mbarrier_arrive_expect(bar, THREAD_COUNT);

      // step 1 all blocks write to block0
      for (int i = thread_idx; i < BUF_SIZE; i += NUM_THREADS) {
        cluster_put_float(remote_receive_buffer + i, src[i], remote_bar);
      }


      while (!mbarrier_try_wait(bar, arrival_token)) {
      }

    }

    

    cluster.sync();

    if (block_rank == 0) {
      for (int b = 1; b < CLUSTER_SIZE; b++) {
        unsigned int target_blockrank = b;
        // uint64_t *remote_bar = set_block_rank(bar, target_blockrank);
        // T *remote_receive_buffer = set_block_rank(dst, target_blockrank);
        uint64_t *remote_bar = cluster.map_shared_rank(bar, target_blockrank);
        T *remote_receive_buffer =  cluster.map_shared_rank(dst, target_blockrank);

        arrival_token = mbarrier_arrive_expect(bar, THREAD_COUNT);

        // step 2 block0 write to all blocks inside the cluster
        for (int i = thread_idx; i < BUF_SIZE; i += NUM_THREADS) {
          cluster_put_float(remote_receive_buffer + i, src[i], remote_bar);
        }
        while (!mbarrier_try_wait(bar, arrival_token)) {
        }
      }
    }

    cluster.sync();
  }
};

} // namespace tb