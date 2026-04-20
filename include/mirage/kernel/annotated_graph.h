/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include "mirage/config.h"
#include "mirage/kernel/customized.h"
#include "mirage/kernel/graph.h"
#include "mirage/persistent_kernel/runtime_header.h"

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mirage {
namespace kernel {

// A per-side view of a layer's task grid with respect to a single producer or
// consumer edge (or the unified LCM result at a fork/join).
//
// Layout:
//   event_dim[0..MAX_TENSOR_DIMS): number of events along each TENSOR
//     dimension. Product across these dims = total events at this boundary.
//   last3[0..2]: block of tasks per event along each GRID axis (x, y, z).
//     Derived as grid_dim[g] / event_dim[axis_map[g]] when axis_map[g] >= 0,
//     else grid_dim[g] (unmapped / replicated axis).
//   grid_dim: the owning layer's bgraph grid dimensions.
//   axis_map: input_map (consumer-side) or output_map (producer-side) from
//     the bridging tensor; grid-axis -> tensor-dim (or -1 if replicated).
struct TaskView {
  std::array<int, mirage::config::MAX_TENSOR_DIMS> event_dim;
  std::array<int, 3> last3;
  dim3 grid_dim;
  int3 axis_map;
};

// One data-flow edge producer_layer -> consumer_layer on a specific bridging
// DTensor (identified by the producer's out_slot).
struct EdgeInfo {
  int prod_layer;
  int cons_layer;
  int out_slot;
  int in_slot;
  size_t tensor_guid;
  int3 output_map;
  int3 input_map;
  std::array<int, mirage::config::MAX_TENSOR_DIMS> event_dim;
  TaskView producer_side_view;
  TaskView consumer_side_view;
  int fork_group_id = -1;
  int join_group_id = -1;
  bool is_residual_stripped = false;
};

// Group of edges sharing a single fork-producer layer; the LCM-adjusted
// last3 is unified across all branches (in producer grid-axis space).
struct ForkGroupInfo {
  int producer_layer;
  std::vector<int> outgoing_edges;
  std::array<int, 3> lcm_last3;
};

// Group of edges sharing a single join-consumer layer; LCM-adjusted last3
// unified across incoming branches (in consumer grid-axis space).
struct JoinGroupInfo {
  int consumer_layer;
  std::vector<int> incoming_edges;
  std::array<int, 3> lcm_last3;
};

struct LayerInfo {
  mirage::kernel::KNCustomizedOp const *op = nullptr;
  std::vector<int> in_edges;
  std::vector<int> out_edges;
  bool is_fork_producer = false;
  bool is_join_consumer = false;
  int fork_parent_group = -1;
  int fork_branch_index = -1;
  mirage::runtime::TaskType task_type;
  int variant_id = 0;
  int num_inputs = 0;
  int num_outputs = 0;
};

struct AnnotatedGraph {
  std::vector<LayerInfo> layers;
  std::vector<int> ordered_layers;
  std::vector<EdgeInfo> edges;
  std::vector<ForkGroupInfo> fork_groups;
  std::vector<JoinGroupInfo> join_groups;
  std::vector<EdgeInfo> stripped_residual_edges;
};

using TaskConfigMap = std::unordered_map<
    mirage::kernel::KNOperator const *,
    std::tuple<int, int, mirage::runtime::TaskType, int>>;

// Build the AnnotatedGraph from a KNGraph. Throws std::runtime_error on:
//   - cycles in the DAG (should be impossible under most-recent-writer)
//   - disallowed role combinations (case 2: join-consumer + fork-consumer;
//     case 3: join-producer + fork-producer)
//   - fork/join LCM that exceeds the producer's / consumer's grid_dim.
AnnotatedGraph build_annotated_graph(mirage::kernel::Graph const &kn_graph,
                                     TaskConfigMap const &task_configs);

// Optional: dumps the AnnotatedGraph to JSON for debugging when
// MIRAGE_DUMP_ANNOTATED_GRAPH is set. Returns empty string when disabled.
std::string maybe_dump_annotated_graph(AnnotatedGraph const &ag);

} // namespace kernel
} // namespace mirage
