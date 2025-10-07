/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

static constexpr int RESERVED_SMEM_SIZE =
    16 * 1024; // for metadata and other uses
static constexpr int PAGE_SIZE = 8 * 1024;
static constexpr int MAX_PAGES =
    (MAX_SMEM_SIZE - RESERVED_SMEM_SIZE) / PAGE_SIZE;

enum class PageState {
  FREE,
  OCCUPIED,
};

struct PageInfo {
  PageState state;
  // data
};

class SharedMemoryManager {
private:
  char *base_ptr;
  PageInfo page_info[MAX_PAGES];

public:
  __device__ SharedMemoryManager(char *smem_base, int total_smem_size) {
    base_ptr = smem_base;
    if (threadIdx.x == 0) {
      for (int i = 0; i < MAX_PAGES; i++) {
        page_info[i].state = PageState::FREE;
      }
    }
    __syncthreads();
  }

  // Allocate contiguous pages
  __device__ int allocate_pages(int num_pages, TensorLifetime lifetime) {
    if (threadIdx.x == 0) {
      // Find contiguous free pages
      for (int start = 0; start <= MAX_PAGES - num_pages; start++) {
        bool available = true;
        for (int i = 0; i < num_pages; i++) {
          if (page_info[start + i].state != PageState::FREE) {
            available = false;
            break;
          }
        }
        if (available) {
          // Mark pages as occupied
          for (int i = 0; i < num_pages; i++) {
            page_info[start + i].state = PageState::OCCUPIED;
          }
          // __threadfence_block();
          return start;
        }
      }
    }
    __syncthreads();
    return -1; // Allocation failed
  }

  // Get pointer with start page id
  __device__ void *get_page_ptr(int page_id) {
    return base_ptr + (page_id * PAGE_SIZE);
  }

  // release pages if possible
  __device__ void
      try_release_pages(int page_id, int num_pages, TensorLifetime lifetime) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_pages; i++) {
        if (lifetime == TensorLifetime::INPUT ||
            lifetime == TensorLifetime::COMPUTE) {
          // Can free immediately
          page_info[page_id + i].state = PageState::FREE;
        }
      }
      // __threadfence_block();
    }
    __syncthreads();
  }

  // Explicit release for OUTPUT tensors
  __device__ void release_all_pages(int page_id, int num_pages) {
    if (threadIdx.x == 0) {
      for (int i = 0; i < num_pages; i++) {
        page_info[page_id + i].state = PageState::FREE;
      }
    }
    __syncthreads();
  }

  // Wait for pages to be ready for compute
  __device__ void wait_for_ready(int page_id, int num_pages) {
    while (true) {
      __syncthreads();
      bool all_ready = true;

      if (threadIdx.x == 0) {
        for (int i = 0; i < num_pages; i++) {
          if (page_info[page_id + i].state == PageState::FREE) {
            all_ready = false;
            break;
          }
        }
      }

      __syncthreads();
      if (all_ready) {
        break;
      }
    }
  }

  // Check if allocation would succeed (for prefetch tasks)
  __device__ bool can_allocate(int num_pages) {
    bool result = false;
    if (threadIdx.x == 0) {
      for (int start = 0; start <= MAX_PAGES - num_pages; start++) {
        bool available = true;
        for (int i = 0; i < num_pages; i++) {
          if (page_info[start + i].state != PageState::FREE) {
            available = false;
            break;
          }
        }
        if (available) {
          result = true;
          break;
        }
      }
    }
    __syncthreads();
    return result;
  }
};
