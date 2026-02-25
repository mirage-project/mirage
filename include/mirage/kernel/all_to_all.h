/* Copyright 2023-2024 CMU
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

#include "mirage/kernel/graph.h"
#include "mirage/kernel/device_memory_manager.h"

namespace mirage {
namespace kernel {

class KNAllToAll : public KNOperator {
public:
  // Use the top-level enum (defined in all_to_all_type.h) so that graph.h
  // can declare Graph::all_to_all() without a circular include dependency.
  using AllToAllType = mirage::kernel::AllToAllType;

  KNAllToAll(Graph *_graph,
             DTensor const &input,
             AllToAllType _a2a_type,
             int _num_experts,
             int _experts_per_rank,
             int _topk,
             DTensor const &routing_indices,
             DTensor const &routing_weights);

  ~KNAllToAll();

  bool fingerprint(void) override;
  operator json() const override;

public:
  AllToAllType a2a_type;
  int num_experts;
  int experts_per_rank;
  int topk;
  int world_size;
  int dp_size;
  int node_size;
  int rank;

  // send buffer: staging area for tokens going to non-NVLink ranks
  // shape: [world_size * batch_size * hidden_dim] elements
  void *send_buffer;

  // recv_ptrs[r] = pointer to rank r's recv buffer (peer GPU memory via NVLink)
  // nullptr for remote ranks that require send_buffer staging
  void **recv_ptrs;

  // per-rank token counts and cumulative offsets in send_buffer
  int *send_counts;    // [world_size]
  int *send_offsets;   // [world_size]

  // per-rank token counts and cumulative offsets in recv buffer
  int *recv_counts;    // [world_size]
  int *recv_offsets;   // [world_size]

  // sync_flags[r] = 1 once rank r finishes sending; combine kernel spins on these
  volatile int *sync_flags;   // [world_size]

  // cooperative_groups grid sync counter
  int *grid_counter;          // [1]
};

} // namespace kernel
} // namespace mirage
