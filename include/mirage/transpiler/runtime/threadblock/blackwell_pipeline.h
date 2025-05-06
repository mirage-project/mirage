// blackwell_pipeline.h
#pragma once
#include <cutlass/pipeline/sm100_pipeline.hpp>

namespace tb {

template<int _Stage>
struct BlackwellAsyncPipeline {

  static constexpr int Stage = _Stage;
  using Mainloop      = cutlass::PipelineTmaAsync<Stage>;
  using Barrier       = typename Mainloop::ProducerBarrierType;
  using PState        = cutlass::PipelineState<Stage>;
  using Params        = typename Mainloop::Params;

  PState smem_read;
  PState smem_write{ cutlass::make_producer_start_state<Mainloop>() };
  Params p;
  cute::aligned_struct<16,_1>               pipe_smem;
  typename Mainloop::SharedStorage &storage =
      *reinterpret_cast<typename Mainloop::SharedStorage*>(&pipe_smem);
  Mainloop pipe;

  __device__
  BlackwellAsyncPipeline(void* base, bool prod, bool cons,
              uint32_t bytes, uint32_t n_consumer_wgs)
  : p{ bytes,
       prod ? Mainloop::ThreadCategory::Producer
            : cons ? Mainloop::ThreadCategory::Consumer
                   : Mainloop::ThreadCategory::NonParticipant,
       (threadIdx.x % cutlass::NumThreadsPerWarpGroup)==0,
       uint32_t(cutlass::NumThreadsPerWarpGroup*n_consumer_wgs) },
    storage(*reinterpret_cast<typename Mainloop::SharedStorage*>(base)),
    pipe(storage, p, cute::Shape<_1,_1,_1>{}) {}

  __device__ __forceinline__ auto producer_acquire() {
      pipe.producer_acquire(smem_write);
      return std::make_pair(pipe.producer_get_barrier(smem_write),
                            smem_write.index());
  }
  __device__ __forceinline__ void producer_advance() { ++smem_write; }

  __device__ __forceinline__ int consumer_wait()  {
      auto tok = pipe.consumer_try_wait(smem_read);
      pipe.consumer_wait(smem_read, tok);
      return smem_read.index();
  }
  __device__ __forceinline__ void consumer_release() { pipe.consumer_release(smem_read); ++smem_read;}
};

} // namespace tb
