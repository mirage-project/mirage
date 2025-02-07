// utils.h - Implementation of threadblock level input operators
#include "cutlass/pipeline/sm90_pipeline.hpp"

#pragma once

namespace tb {

template <typename MainloopPipeline>
struct PipelineStorage : cute::aligned_struct<16, _1> {
  using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;

  MainloopPipelineStorage *mainloop;

  __device__ __forceinline__
      PipelineStorage(void *__restrict__ shared_memory_offset) {
    mainloop =
        reinterpret_cast<MainloopPipelineStorage *>(shared_memory_offset);
  }
};

template <int _Stage>
struct HopperAsyncPipeline {
  static constexpr int Stage = _Stage;
  using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stage>;
  using PipelineState = typename cutlass::PipelineState<Stage>;
  // using SharedStorage = tb::SharedStorage<MainloopPipeline, Stage>;
  using PipelineParams = typename MainloopPipeline::Params;
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;

public:
  PipelineState smem_pipe_read;
  PipelineParams pipeline_params;
  MainloopPipeline pipeline;
  // std::optional<MainloopPipeline> pipeline;
  PipelineState smem_pipe_write;

  PipelineStorage<MainloopPipeline> pipeline_storage;

  __device__ __forceinline__
      HopperAsyncPipeline(void *__restrict__ shared_memory_offset,
                          bool producer,
                          bool consumer,
                          uint32_t transactionBytes,
                          uint32_t num_consumer_wgs)
      : smem_pipe_read(),
        smem_pipe_write(cutlass::make_producer_start_state<MainloopPipeline>()),
        pipeline_params{
            transactionBytes,
            producer
                ? MainloopPipeline::ThreadCategory::Producer
                : (consumer ? MainloopPipeline::ThreadCategory::Consumer
                            : MainloopPipeline::ThreadCategory::NonParticipant),
            (threadIdx.x % cutlass::NumThreadsPerWarpGroup) == 0,
            cutlass::NumThreadsPerWarpGroup * num_consumer_wgs},
        pipeline_storage(shared_memory_offset),
        pipeline(*(pipeline_storage.mainloop),
                 pipeline_params,
                 Shape<_1, _1, _1>{}) {}

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

  // debug only
  __device__ __forceinline__ void producer_commit() {
    pipeline.producer_commit(smem_pipe_write, (8192));
  }

  __device__ __forceinline__ void consumer_release() {
    pipeline.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
  }
};

} // namespace tb