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

#include "mirage/kernel/all_to_all.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/device_memory_manager.h"
#include <cassert>
#include <cstring>
#include <iostream>

namespace mirage {
namespace kernel {

DTensor Graph::all_to_all(DTensor const &input,
                          KNAllToAll::AllToAllType type,
                          int num_experts,
                          int experts_per_rank,
                          int topk,
                          DTensor const &routing_indices,
                          DTensor const &routing_weights) {
  KNOperator *op = create_all_to_all_op(
      input, type, num_experts, experts_per_rank, topk,
      routing_indices, routing_weights);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

DTensor *Graph::all_to_all(DTensor const *input,
                           KNAllToAll::AllToAllType type,
                           int num_experts,
                           int experts_per_rank,
                           int topk,
                           DTensor const *routing_indices,
                           DTensor const *routing_weights) {
  KNOperator *op = create_all_to_all_op(
      *input, type, num_experts, experts_per_rank, topk,
      *routing_indices, *routing_weights);
  assert(op != nullptr);
  operators.push_back(op);
  return &op->output_tensors[0];
}

KNOperator *Graph::create_all_to_all_op(DTensor const &input,
                                        KNAllToAll::AllToAllType type,
                                        int num_experts,
                                        int experts_per_rank,
                                        int topk,
                                        DTensor const &routing_indices,
                                        DTensor const &routing_weights) {
  return new KNAllToAll(this, input, type, num_experts, experts_per_rank,
                        topk, routing_indices, routing_weights);
}

KNAllToAll::KNAllToAll(Graph *_graph,
                       DTensor const &input,
                       AllToAllType _a2a_type,
                       int _num_experts,
                       int _experts_per_rank,
                       int _topk,
                       DTensor const &routing_indices,
                       DTensor const &routing_weights)
    : KNOperator(_graph,
                 _a2a_type == AllToAllType::DISPATCH
                     ? mirage::type::KN_ALL_TO_ALL_DISPATCH_OP
                     : mirage::type::KN_ALL_TO_ALL_COMBINE_OP,
                 std::vector<DTensor>{input, routing_indices, routing_weights}),
      a2a_type(_a2a_type),
      num_experts(_num_experts),
      experts_per_rank(_experts_per_rank),
      topk(_topk),
      send_buffer(nullptr),
      recv_ptrs(nullptr),
      send_counts(nullptr),
      send_offsets(nullptr),
      recv_counts(nullptr),
      recv_offsets(nullptr),
      sync_flags(nullptr),
      grid_counter(nullptr) {

  // Validate: input is [batch_size, hidden_dim]
  assert(input.num_dims == 2);
  // routing_indices and routing_weights are [batch_size, topk]
  assert(routing_indices.num_dims == 2);
  assert(routing_weights.num_dims == 2);
  assert(routing_indices.dim[0] == input.dim[0]);
  assert(routing_weights.dim[0] == input.dim[0]);
  assert(routing_indices.dim[1] == _topk);
  assert(routing_weights.dim[1] == _topk);
  assert(_num_experts % _experts_per_rank == 0);

  world_size = _num_experts / _experts_per_rank;
  rank = 0;      // TODO: read from graph or environment
  node_size = 8; // TODO: detect actual node topology
  dp_size = 1;   // TODO: read from graph

  int batch_size = input.dim[0];
  int hidden_dim = input.dim[1];
  size_t elem_size = mirage::type::get_datatype_size(input.data_type);

  // Create output tensor with same shape as input: [batch_size, hidden_dim]
  DTensor output;
  output.data_type = input.data_type;
  output.layout = input.layout;
  output.num_dims = 2;
  output.dim[0] = batch_size;
  output.dim[1] = hidden_dim;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  kgraph->allocate(output);
  output_tensors.push_back(output);

  // Allocate send buffer: worst case every token goes to every rank
  // shape: [world_size, batch_size, hidden_dim]
  size_t send_buf_bytes = (size_t)world_size * batch_size * hidden_dim * elem_size;
  cudaMalloc(&send_buffer, send_buf_bytes);

  // Allocate recv_ptrs array on device (entries filled at runtime via IPC)
  cudaMalloc(&recv_ptrs, world_size * sizeof(void *));
  cudaMemset(recv_ptrs, 0, world_size * sizeof(void *));

  // Allocate per-rank count and offset arrays
  cudaMalloc(&send_counts,  world_size * sizeof(int));
  cudaMalloc(&send_offsets, world_size * sizeof(int));
  cudaMalloc(&recv_counts,  world_size * sizeof(int));
  cudaMalloc(&recv_offsets, world_size * sizeof(int));

  // Allocate sync flags (combine kernel spins until dispatch sets these to 1)
  cudaMalloc((void **)&sync_flags, world_size * sizeof(int));
  cudaMemset((void *)sync_flags, 0, world_size * sizeof(int));

  // Allocate cooperative groups grid sync counter
  cudaMalloc(&grid_counter, sizeof(int));
  cudaMemset(grid_counter, 0, sizeof(int));
}

KNAllToAll::~KNAllToAll() {
  kgraph->free(output_tensors[0]);

  if (send_buffer)  cudaFree(send_buffer);
  if (recv_ptrs)    cudaFree(recv_ptrs);
  if (send_counts)  cudaFree(send_counts);
  if (send_offsets) cudaFree(send_offsets);
  if (recv_counts)  cudaFree(recv_counts);
  if (recv_offsets) cudaFree(recv_offsets);
  if (sync_flags)   cudaFree((void *)sync_flags);
  if (grid_counter) cudaFree(grid_counter);
}

KNAllToAll::operator json() const {
  return json{
      {"op_type", op_type},
      {"a2a_type", a2a_type == AllToAllType::DISPATCH ? "dispatch" : "combine"},
      {"num_experts", num_experts},
      {"experts_per_rank", experts_per_rank},
      {"topk", topk},
      {"world_size", world_size},
      {"rank", rank},
      {"node_size", node_size},
      {"input_tensors", input_tensors},
      {"output_tensors", output_tensors}};
}

bool KNAllToAll::fingerprint(void) {
  assert(false && "KNAllToAll::fingerprint not implemented");
  return false;
}

} // namespace kernel
} // namespace mirage
