// cluster.h - Implementation of threadblock block level cluster operators
//
#pragma once

namespace tb {

// utils
static __device__ __forceinline__ void cluster_sync() {
  // arrive & wait, use align/aquire?
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" : :);
  asm volatile("barrier.cluster.arrive.aligned;\n" : :);
}

// mbarrier init
static __device__ __forceinline__ void
    mbarrier_init(_CUDA_VSTD::uint64_t *__addr,
                  const _CUDA_VSTD::uint32_t &__count) {
  asm("mbarrier.init.shared.b64 [%0], %1;"
      :
      : "r"(__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
}

// threads participant in computation
static __device__ __forceinline__ void
    mbarrier_arrive_expect(_CUDA_VSTD::uint64_t *__addr,
                           const _CUDA_VSTD::uint32_t &__tx_count) {
  _CUDA_VSTD::uint64_t __state;
  asm("mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 %0, [%1], %2; "
      "// 8. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)), "r"(__tx_count)
      : "memory");
  return __state;
}

// threads not participant in computation
static __device__ __forceinline__ void
    mbarrier_arrive(_CUDA_VSTD::uint64_t *__addr,
                    const _CUDA_VSTD::uint32_t &__tx_count) {
  _CUDA_VSTD::uint64_t __state;
  // scope cluster
  asm("mbarrier.arrive.release.cluster.shared::cta.b64                   %0,  "
      "[%1];           // 3a. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr))
      : "memory");
  return __state;
}

// put a value to remote buffer in same cluster
static __device__ __forceinline__ void cluster_put_float(
    _Type *__addr, _Type const &__value, _CUDA_VSTD::uint64_t *__remote_bar) {
  asm("st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], "
      "%1, [%2];    // 1. "
      :
      : "r"(__as_ptr_remote_dsmem(__addr)),
        "r"(/*as_b32*/ *reinterpret_cast<const _CUDA_VSTD::int32_t *>(
            &__value)),
        "r"(__as_ptr_remote_dsmem(__remote_bar))
      : "memory");
}

static __device__ __forceinline__ void
    mbarrier_try_wait(_CUDA_VSTD::uint64_t *__addr,
                      const _CUDA_VSTD::uint64_t &__state) {
  _CUDA_VSTD::uint32_t __waitComplete;
  asm("{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.acquire.cluster.shared::cta.b64         P_OUT, [%1], "
      "%2;                        // 6a. \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__waitComplete)
      : "r"(__as_ptr_smem(__addr)), "l"(__state)
      : "memory");
  return __waitComplete
}

// reduction
template <typename T, int CLUSTER_SIZE, int NUM_THREADS, int BUF_SIZE>
class ClusterReduction {

  static constexpr int THREAD_COUNT = BUF_SIZE / NUM_THREADS;
  static __device__ __forceinline__ run(uint64_t bar,
                                        int thread_idx,
                                        int block_rank,
                                        T *const__restrict__ src,
                                        T *__restrict__ dst) {

    // init barrier, all threads participant
    mbarrier_init(&bar, blockDim.x * blockDim.y * blockDim.z);
    // cluster sync
    cluster_sync();

    if (block_rank != 0) {
      unsigned int block0rank = 0;
      uint64_t *remote_bar = cluster.map_shared_rank(bar, block0rank);
      T *remote_receive_buffer = cluster.map_shared_rank(dst, block0rank);

      arrival_token = mbarrier_arrive_expect(bar, THREAD_COUNT);

      // step 1 all blocks write to block0
      for (int i = thread_idx; i < BUF_SIZE; i += NUM_THREADS) {
        cluster_put_float(remote_receive_buffer[i], src[i], bar);
      }

      while (!mbarrier_try_wait(bar, arrival_token)) {
      }
    }

    cluster_sync();

    if (block_rank == 0) {
      for (int b = 1; b < CLUSTER_SIZE; b++) {
        unsigned int target_blockrank = b;
        uint64_t *remote_bar = cluster.map_shared_rank(bar, target_blockrank);
        T *remote_receive_buffer =
            cluster.map_shared_rank(dst, target_blockrank);

        arrival_token = mbarrier_arrive_expect(bar, THREAD_COUNT);

        // step 2 block0 write to all blocks inside the cluster
        for (int i = thread_idx; i < BUF_SIZE; i += NUM_THREADS) {
          cluster_put_float(remote_receive_buffer[i], src[i], bar);
        }
        while (!mbarrier_try_wait(bar, arrival_token)) {
        }
      }
    }

    cluster_sync();
  }
}

} // namespace tb