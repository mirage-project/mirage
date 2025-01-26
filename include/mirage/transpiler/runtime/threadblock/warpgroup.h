// utils.h - Implementation of warpgroup level helper functions

#include "../../../config.h"
#pragma once

namespace tb {



// number of regs in consumers * num_threads  + number of regs in producers * num_threads <= 65536
//exp. 32 * 128 + 160 * 3 * 128 == 65536
// constexpr std::array<int, 4> registerFileDistribution_P1C3 = {32, 160, 160, 160};
// constexpr std::array<int, 3> registerFileDistribution_P1C2 = {32, 240, 240};

// constexpr std::map<int, const void*> registerFileDistributionMap = {
//         {2, &registerFileDistribution_P1C2},
//         {3, &registerFileDistribution_P1C3}
// };

// constexpr std::array<int, 4> registerFileDistribution_P2C2 = {24, 24, 232, 232};



static __device__ __forceinline__ int lane_id(){
    return threadIdx.x & 0x1f;
}

static __device__ __forceinline__ int warp_id_in_wg(){
    return __shfl_sync(0xffffffff, (threadIdx.x / mirage::config::NUM_THREADS_PER_WARP) % mirage::config::NUM_WARPS_PER_GROUP, 0);
}

static __device__ __forceinline__ int warp_id(){
    return __shfl_sync(0xffffffff, threadIdx.x / mirage::config::NUM_THREADS_PER_WARP, 0);
}

static __device__ __forceinline__ int warpgroup_id(){
    return __shfl_sync(0xffffffff, threadIdx.x / mirage::config::NUM_THREADS_PER_WARP_GROUP, 0);
}

// decrease register files in a wg
template<uint32_t RegCount>
static __device__ __forceinline__ void decrease_regs(){
#ifdef MIRAGE_GRACE_HOPPER
    asm volatile( "setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
}


// increase register files in a wg
template<uint32_t RegCount>
static __device__ __forceinline__ void increase_regs(){
#ifdef MIRAGE_GRACE_HOPPER
    asm volatile( "setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
#elif defined(__CUDA_ARCH__)
    asm volatile ("brkpt;\n" ::);
#endif
}

// sync inside a warp group
static __device__ __forceinline__ void warpgroup_sync(uint32_t barrier_id){
    
#ifdef MIRAGE_GRACE_HOPPER
    asm volatile("bar.sync %0, %1;\n" :: "r"(barrier_id), "n"(mirage::config::NUM_THREADS_PER_GROUP));
#elif defined(__CUDA_ARCH__)
asm volatile ("brkpt;\n" ::);
#endif
}



// template <int N, int NUM_THREADS>
// struct WarpGroup{
//     static constexpr int NUM_WARPS = N;
//     static constexpr int NUM_THREADS_PER_WARP = NUM_THREADS;
//     static constexpr int NUM_THREADS_PER_GROUP = NUM_THREADS * NUM_WARPS;

//     static __device__ __forceinline__ warpgroup_id(){
//         return __shfl_sync(0xffffffff, threadIdx.x / NUM_THREADS_PER_GROUP, 0);
//     }

//     // sync inside a warp group
//     static __device__ __forceinline__ sync(uint32_t barrier_id){
// #if MIRAGE_GRACE_HOPPER
//         asm volatile("bar.sync %0, %1;\n" :: "r"(barrier_id), "n"(NUM_THREADS_PER_GROUP));
// #elif defined(__CUDA_ARCH__)
//     asm volatile ("brkpt;\n" ::);
// #endif
//     }
// }

}


