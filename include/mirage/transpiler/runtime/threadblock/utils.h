// utils.h - Implementation of threadblock level input operators


#pragma once


namespace tb {

static constexpr int NumThreads

// number of regs in consumers * num_threads  + number of regs in producers * num_threads <= 65536
//exp. 32 * 128 + 160 * 3 * 128 == 65536
constexpr std::array<int, 4> registerFileDistribution_P1C3 = {32, 160, 160, 160};

constexpr std::array<int, 4> registerFileDistribution_P2C2 = {24, 24, 232, 232};

constexpr std::array<int, 3> registerFileDistribution_P1C2 = {32, 240, 240};

static __device__ __forceinline__ lane_id(uint32_t barrier_id){
    return threadIdx.x & 0x1f;
}

static __device__ __forceinline__ warp_id_sync(){
    return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarp, 0);
}


template <int N, int NUM_THREADS>
struct WarpGroup{
    
    static constexpr int NUM_WARPS = N;
    static constexpr int NUM_THREADS_PER_WARP = NUM_THREADS;
    static constexpr int NUM_THREADS_PER_GROUP = NUM_THREADS * NUM_WARPS;

    // sync inside a warp group
    static __device__ __forceinline__ sync(uint32_t barrier_id){
#if ARCH_GRACE_HOPPER
        asm volatile("bar.sync %0, %1;\n" :: "r"(barrier_id), "n"(NUM_THREADS_PER_GROUP));
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
    }
}

}


