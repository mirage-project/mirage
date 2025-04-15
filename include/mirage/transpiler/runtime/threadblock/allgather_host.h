#pragma once

#include "comm_executor.h"
#include <nvshmem.h>
// #include "/usr/include/nvshmem_12/nvshmem.h"
#include <nvshmemx.h>
#include <cassert>
#include <cstdio>

namespace tb {

//TODO: Make tiles instead of whole tensor (Jianan)
template <typename T, typename TileLayout>
void allgather_host(T* dst_tensor, T* src_tensor, 
                        uint64_t* signals, size_t tensor_size, 
                        int mype, int npes, int div_dim = 0, bool use_pull = false,
                        cudaStream_t stream = nullptr) {
  
  CommExecutor<T, TileLayout, false> comm_executor;

  // TODO: Assume only a whole tile is sent
  signals = signals + mype;
  dst_tensor = dst_tensor + mype * tensor_size;

  if (use_pull) {
    printf("Pull mode not supported yet\n");
    assert(false);
  } else {
    for (int dst_pe = 0; dst_pe < npes; dst_pe++) {
        // if (dst_pe == mype) continue;
        if (stream == nullptr) {
          printf("Stream in allgather_host is nullptr!\n");
          assert(false);
        }
        nvshmemx_putmem_signal_nbi_on_stream(dst_tensor, src_tensor, tensor_size * sizeof(T), signals, 1, NVSHMEM_SIGNAL_SET, dst_pe, stream);
    }
  }
}

__device__ inline void allgather_signal_wait_until_ne(uint64_t* signal, uint64_t signal_idx, uint64_t cmp_val = 0) {
    nvshmem_signal_wait_until(signal + signal_idx, NVSHMEM_CMP_NE, cmp_val);
}

}