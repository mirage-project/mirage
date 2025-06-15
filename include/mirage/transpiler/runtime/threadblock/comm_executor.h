// comm_executor.h - Implementation of thread block level p2p communications
#pragma once

#include "cute/config.hpp"
#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"
#include "profiler.h"
#include "mpi.h"

#include <nvshmem.h>
#include <nvshmemx.h>
#include <cstdio>

namespace tb {

template <typename T, typename TileLayout, bool sync>
class CommExecutor {
public:
  CUTE_DEVICE
  static void send(T *__restrict__ recv_ptr,
                  T *__restrict__ send_ptr,
                  uint64_t * signal,
                  int peer,
                  int npes,
                  uint64_t *profiler_buffer=NULL) {
    //PROFILER_CLOSURE_PARAMS_DECL
    //PROFILER_INIT(profiler_buffer, 0, 2, (threadIdx.x % 128 == 0));
    // Call in with a thread block
    // Sending a tile to a peer and set the corresponding signal
    // Tile may not be consecutive in physical memory
    // printf("ntered send, sending to PE %d\n", peer);
    auto tile_layout = TileLayout{};
    auto _shape = cute::shape(tile_layout);

    // Find contiguous block decomposition
    const int contiguous_chunk_size = find_contiguous_chunk();
    // printf("contiguous_chunk_size: %d\n", contiguous_chunk_size);
    const int num_chunks = cute::size(tile_layout) / contiguous_chunk_size;

    auto tile_layout_standalone = cute::make_layout(_shape);
    // cute::print_layout(tile_layout);
    // cute::print_layout(tile_layout_standalone);

    const int mode = 1; // 0: warp, 1: block

    // Assuming 32 threads per warp
    const int executor_num = mode == 0 ? blockDim.x / 32 : 1;
    const int chunks_per_executor = mode == 0 ? (num_chunks + executor_num - 1) / executor_num : num_chunks;
    const int executor_id = mode == 0 ? threadIdx.x / 32 : 0;
    // printf("executor_num: %d, chunks_per_executor: %d, executor_id: %d\n", executor_num, chunks_per_executor, executor_id);
    //using comm_executor = typename CommWarpExecutor<T, sync> ? mode == 0 : typename CommBlockExecutor<T, sync>;

    int end_chunk_idx = min((executor_id+1) * chunks_per_executor, num_chunks);
    for (int chunk_idx = executor_id * chunks_per_executor; chunk_idx < end_chunk_idx; ++chunk_idx) {
      //PROFILER_EVENT_START(2, static_cast<uint32_t>(0));
      int idx = chunk_idx * contiguous_chunk_size;
      auto chunk_coord = cute::idx2crd(idx, tile_layout_standalone);

      int offset = cute::crd2idx(chunk_coord, tile_layout);
      //PROFILER_EVENT_END(2, static_cast<uint32_t>(0));

      //PROFILER_EVENT_START(2, static_cast<uint32_t>(1));
      if (mode == 0) {
        if (signal != NULL && chunk_idx == end_chunk_idx - 1) {
          nvshmem_fence();
          nvshmemx_putmem_signal_nbi_warp(recv_ptr + offset, send_ptr + offset, contiguous_chunk_size * sizeof(T), signal, 1, NVSHMEM_SIGNAL_ADD, peer); 
        } else {
          nvshmemx_putmem_warp(recv_ptr + offset, send_ptr + offset, contiguous_chunk_size * sizeof(T), peer); 
        }
      } else {
        if (signal != NULL && chunk_idx == end_chunk_idx - 1) {
          nvshmem_fence();
          nvshmemx_putmem_signal_nbi_block(recv_ptr + offset, send_ptr + offset, contiguous_chunk_size * sizeof(T), signal, 1, NVSHMEM_SIGNAL_ADD, peer); 
        } else {
          nvshmemx_putmem_nbi_block(recv_ptr + offset, send_ptr + offset, contiguous_chunk_size * sizeof(T), peer); 
        }
      }
      //PROFILER_EVENT_END(2, static_cast<uint32_t>(1));
    }
    if (false && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && signal != NULL) {
      nvshmem_fence();
      nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_ADD, peer);
    }

    //PROFILER_EVENT_START(2, static_cast<uint32_t>(2));
    if constexpr (sync) {
      if (threadIdx.x == 0) {
        nvshmem_signal_wait_until(signal, NVSHMEM_CMP_EQ, npes * 8);
        *signal = 0;
      }
      __syncthreads();
    }
    //PROFILER_EVENT_END(2, static_cast<uint32_t>(2));
  }

private:
  template <int I = 0, int Product = 1>
  static constexpr int find_contiguous_chunk_impl() {
    constexpr TileLayout layout{};

    if constexpr (I == cute::rank(layout)) {
      return Product;
    } else {
      constexpr auto stride = cute::get<I>(cute::stride(layout));
      constexpr auto shape = cute::get<I>(cute::shape(layout));
      if constexpr (stride == Product) {
        return find_contiguous_chunk_impl<I+1, Product * shape>();
      } else {
        return Product;
      }
    }
  }

  CUTE_DEVICE
  static constexpr int find_contiguous_chunk() {
    return find_contiguous_chunk_impl();
  }
};

// Block, Warp, Thread
template <typename T, bool sync>
class CommBlockExecutor {
public:
  CUTE_DEVICE
  static void send(T *__restrict__ recv_ptr,
                  T *__restrict__ send_ptr,
                  int64_t num_elems,
                  int peer,
                  uint64_t * signal) {
    // put -> push based, get -> pull based
    // Launch (_nbi if not sync)
    nvshmemx_int16_put_block(recv_ptr, send_ptr, num_elems, peer); 
    
    if (signal != NULL) {
      // Ordering
      nvshmem_fence();

      // signaling (_nbi if not sync)
      nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, peer);
    }
  }
};

template <typename T, bool sync>
class CommWarpExecutor {
public:
  CUTE_DEVICE
  static void send(T *__restrict__ recv_ptr,
                  T *__restrict__ send_ptr,
                  int64_t num_elems,
                  int peer,
                  uint64_t * signal) {
    // put -> push based, get -> pull based
    // Launch (_nbi if not sync)
    nvshmemx_float_put_warp(recv_ptr, send_ptr, num_elems, peer); 
    
    if (signal != NULL) {
      // Ordering
      nvshmem_fence();

      // signaling (_nbi if not sync)
      nvshmemx_signal_op(signal, 1, NVSHMEM_SIGNAL_SET, peer);
    }
  }
};

} // namespace tb
