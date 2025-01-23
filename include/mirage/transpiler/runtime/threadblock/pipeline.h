// utils.h - Implementation of threadblock level input operators


#pragma once


namespace tb {
        
        template<int Stage, int transactionBytes>
        struct HopperAsyncPipeline{
            using MainloopPipeline = typename cutlass::PipelineTmaAsync<Stage>;
            using PipelineState = typename cutlass::PipelineState<Stage>;
            using SharedStorage = tb::SharedStorage<MainloopPipeline, Stage>;
            using PipelineParams = typename MainloopPipeline::Params;
            using BarrierType = typename MainloopPipeline::ProducerBarrierType;

            PipelineParams pipeline_params;
            PipelineState smem_pipe_read;
            PipelineParams pipeline_params;
            MainloopPipeline pipeline;

            auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read); pipeline.consumer_wait(smem_pipe_read, barrier_token);};

             static __device__ __forceinline__ void init(){
                pipeline_params.role = ((warp_group_role == tb::WarpGroupRole::Producer) && (producer_warp_role == tb::ProducerWarpRole::MainloopEpilogue))  ? MainloopPipeline::ThreadCategory::Producer : MainloopPipeline::ThreadCategory::Consumer;
                pipeline_params.is_leader = warp_group_thread_idx == 0;
                pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;
                pipeline_params.transaction_bytes = transactionBytes;
                pipeline = MainloopPipeline(shared_storage.pipelines[0].mainloop, pipeline_params, Shape<_1, _1, _1>{});
             }

             static __device__ __forceinline__ void producer_acquire(){
                pipeline.producer_acquire(smem_pipe_write);
                BarrierType *tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        `       int write_stage = smem_pipe_write.index();
                return {tma_barrier, write_stage};
             }

             static __device__ __forceinline__ void producer_advance(){
                smem_pipe_write++;
             }

              static __device__ __forceinline__ int consumer_wait(){
                consumer_wait(pipeline, smem_pipe_read);
                return smem_pipe_read.index();
              }

              static __device__ __forceinline__ void consumer_release(){
                    smem_pipe_read++;
              }
        }

}