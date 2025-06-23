// blackwell_pipeline.h
#pragma once
#include "pipeline.h"
#include <cutlass/pipeline/sm100_pipeline.hpp>

namespace tb {

template <int _Stage,
          class ClusterShape_MNK_,
          class AtomThrShape_MNK_ = Shape<_2, _1, _1>>
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
                             uint32_t num_consumer_wgs,
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
            cutlass::NumThreadsPerWarpGroup * num_consumer_wgs},
        pipeline_storage(shared_memory_offset),
        pipeline(*(pipeline_storage.mainloop),
                 pipeline_params,
                 ClusterShape_MNK_{},
                 cute::true_type{}, // InitBarriers
                 cute::true_type{}) // InitMasks
  {
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