// blackwell_pipeline.h
#pragma once
#include "pipeline.h"
#include <cutlass/pipeline/sm100_pipeline.hpp>

namespace tb {

// AtomThrShape_MNK must match the MMA atom's CTA granularity: PipelineTmaUmmaAsync
// emits .cta_group::2 UMMA barrier ops for Shape<_2,_1,_1>. The transpiler now
// generates 1-SM MMA (the MPK runtime is single-CTA with no multicast), so
// defaulting to the 2-CTA shape made ptxas reject the kernel for mixing
// .cta_group::1 and .cta_group::2.
template <int _Stage,
          class ClusterShape_MNK_,
          class AtomThrShape_MNK_ = Shape<_1, _1, _1>>
struct BlackwellAsyncPipeline {
  static constexpr int Stage = _Stage;
  using ClusterShape = ClusterShape_MNK_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;
  using MainloopPipeline = typename cutlass::
      PipelineTmaUmmaAsync<Stage, ClusterShape, AtomThrShape_MNK>;
  using PipelineState = typename cutlass::PipelineState<Stage>;
  // using SharedStorage = tb::SharedStorage<MainloopPipeline, Stage>;
  using PipelineParams = typename MainloopPipeline::Params;
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;

public:
  PipelineState smem_pipe_read;
  PipelineParams pipeline_params;
  PipelineStorage<MainloopPipeline> pipeline_storage;
  MainloopPipeline pipeline;
  PipelineState smem_pipe_write;

  __device__ __forceinline__
      BlackwellAsyncPipeline(void *__restrict__ shared_memory_offset,
                             bool producer,
                             bool consumer,
                             uint32_t transactionBytes,
                             uint32_t num_consumers,
                             bool is_leader_cta)
      : smem_pipe_read(),
        smem_pipe_write(cutlass::make_producer_start_state<MainloopPipeline>()),
        pipeline_params{
            transactionBytes,
            producer
                ? MainloopPipeline::ThreadCategory::Producer
                : (consumer ? MainloopPipeline::ThreadCategory::Consumer
                            : MainloopPipeline::ThreadCategory::NonParticipant),
            (threadIdx.x % cutlass::NumThreadsPerWarpGroup) == 0 &&
                is_leader_cta,
            // The consumer count must match the threads that actually call
            // consumer_wait/consumer_release for THIS pipeline, and
            // PipelineTmaUmmaAsync only accepts 32 or a multiple of 128
            // (sm100_pipeline.hpp: it spreads the empty-arrive duty across the
            // participating threads, and for any other count no thread signals
            // at all -- the producer then blocks forever).
            //
            // A matmul-consumed pipeline is driven by one elected warp, so it
            // passes 32; an elementwise-consumed one runs on every consumer
            // thread and passes 128 * num_consumer_wgs. The caller decides,
            // because only the transpiler knows which op consumes the stensor.
            num_consumers},
        pipeline_storage(shared_memory_offset),
        pipeline(*(pipeline_storage.mainloop),
                 pipeline_params,
                 ClusterShape_MNK_{},
                 cute::true_type{}, // InitBarriers
                 cute::true_type{}) // InitMasks
  {
    // Fail loudly rather than deadlock: PipelineTmaUmmaAsync silently disables
    // every empty-arrive signaller for counts outside {32, k*128}.
    assert((num_consumers == cutlass::NumThreadsPerWarp ||
            num_consumers % cutlass::NumThreadsPerWarpGroup == 0) &&
           "BlackwellAsyncPipeline: num_consumers must be 32 or a multiple of "
           "128, else the producer blocks forever");
    cutlass::pipeline_init_arrive_relaxed(size(ClusterShape{}));
    cutlass::pipeline_init_wait(size(ClusterShape{}));
  }

  // debug
  // __device__ __forceinline__ std::pair<BarrierType *, int>
  // producer_acquire(uint32_t k_iter) {
  //   pipeline.producer_acquire(smem_pipe_write, k_iter);
  //   BarrierType *tma_barrier =
  //   pipeline.producer_get_barrier(smem_pipe_write); int write_stage =
  //   smem_pipe_write.index(); return {tma_barrier, write_stage};
  // }

  __device__ __forceinline__ std::pair<BarrierType *, int> producer_acquire() {
    pipeline.producer_acquire(smem_pipe_write);
    BarrierType *tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
    int write_stage = smem_pipe_write.index();
    return {tma_barrier, write_stage};
  }

  __device__ __forceinline__ void producer_advance() {
    ++smem_pipe_write;
  }

  __device__ __forceinline__ int consumer_wait() {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
    return smem_pipe_read.index();
  }

  __device__ __forceinline__ void producer_commit(PipelineState state) {
    pipeline.producer_commit(state);
  }

  __device__ __forceinline__ void consumer_release() {
    pipeline.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
  }
};

} // namespace tb