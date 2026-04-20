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
// Why per-side (not per-layer): a layer's consumer-side view (which drives
// its tasks' `trigger_event` assignment, i.e. the event that LAUNCHES each
// task post-prelaunch) is independent from its producer-side view (which
// drives `dependent_event`, i.e. the event that each task FIRES on
// completion). For layers playing both case-1 (join-consumer + fork-producer)
// or case-4 (join-producer + fork-consumer) roles, the two views are
// analyzed by separate LCM passes and never conflict because they write to
// different slots on the FullTaskDesc.
//
// Why event_dim indexes tensor dims but last3 indexes grid axes: the event
// grid lives in the bridging tensor's dim space (so it's invariant to how
// each side's grid maps to it); the task block per event is a sub-cuboid
// of the OWNING layer's bid grid (so grid-axis coords are natural).
//   event_dim[0..MAX_TENSOR_DIMS): count of events along each TENSOR dim.
//     Product = total events at this boundary.
//   last3[0..2]: block of tasks per event along each GRID axis (x, y, z).
//     Derived as grid_dim[g] / event_dim[axis_map[g]] when axis_map[g] >= 0,
//     else grid_dim[g] (axis is replicated and every task on it observes
//     the full tensor).
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
// DTensor.
//
// Edge identity is (prod_layer, out_slot, cons_layer, in_slot), NOT the
// tensor guid. qwen3 reuses physical tensor buffers aggressively (e.g.
// x = attn_out; x = mlp_out), so a single guid can have multiple distinct
// producers across a program; using (prod, out_slot) pins each read to the
// writer that was current at read-registration time, which is the behavior
// we want and trivially avoids false cycles.
//
// After fork/join LCM, `event_dim` may be reduced on the tensor dims that
// corresponded to grid axes where the LCM absorbed a factor, and the two
// task_views are recomputed accordingly. Residual edges are kept in the
// edge list (is_residual_stripped = true) but excluded from per-layer
// in_edges/out_edges so they don't influence role classification or event
// emission.
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

// Group of edges sharing a single fork-producer layer.
//
// The LCM is taken on grid-axis space (not tensor-dim space) because the
// producer's grid is shared across all branches (only one producer layer),
// while the bridging tensor and the consumer grids/input_maps can differ
// per branch. Grid-axis LCM is therefore the only frame that makes all
// branches commensurable. After LCM, every branch's producer-side last3
// equals `lcm_last3`, which lets us lay out each fork event as a single
// contiguous producer sub-cuboid feeding an interleaved slab of consumer
// tasks drawn from all branches.
struct ForkGroupInfo {
  int producer_layer;
  std::vector<int> outgoing_edges;
  std::array<int, 3> lcm_last3;
};

// Group of edges sharing a single join-consumer layer.
//
// Symmetric to ForkGroupInfo: the consumer's grid is shared across incoming
// branches, so we LCM on consumer grid axes. Each join event corresponds to
// one sub-cuboid of the consumer's grid and collects completion signals
// from N producer sub-cuboids (one per incoming edge).
struct JoinGroupInfo {
  int consumer_layer;
  std::vector<int> incoming_edges;
  std::array<int, 3> lcm_last3;
};

struct LayerInfo {
  mirage::kernel::KNCustomizedOp const *op = nullptr;
  // Non-stripped edges only. in/out_edges.size() reflects the DAG topology
  // used by classification, not the raw incoming/outgoing tensor count.
  std::vector<int> in_edges;
  std::vector<int> out_edges;
  // is_fork_producer: >=2 DISTINCT downstream consumer layers. Multiple
  // out-edges to the same consumer (e.g. a producer with 2 output tensors
  // both feeding the same next op, as in qwen3) are NOT a fork: they stay
  // on one normal chain edge-group from the dependency perspective.
  bool is_fork_producer = false;
  // is_join_consumer: >=2 DISTINCT upstream producer layers. Same "distinct"
  // rule for the symmetric reason.
  bool is_join_consumer = false;
  // Set on the immediate fork-consumer layers (those directly triggered by
  // a fork event). branch_index 0 is the "bundle head" that actually runs
  // the interleaved emission; other branches are skipped by the outer loop.
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

// Build the AnnotatedGraph from a KNGraph.
//
// Pipeline (see annotated_graph.cc for step-by-step):
//   (a) DAG construction with most-recent-writer: walk ops in insertion
//       order, bind each input to the latest prior writer of its guid.
//   (b) Cycle check via Kahn's algorithm.
//   (c) Residual stripping: drop any edge u->v that coexists with a path
//       u->...->v of length >=2 (transitive dependency makes the direct
//       edge redundant; this converts e.g. transformer residuals into
//       plain chains).
//   (d) Role classification (distinct consumers/producers counted).
//   (e) Case 2/3 rejection (see docs/mpk/parallel_path_design.md).
//   (f) Topological ordering (determines task emission order).
//   (g) Per-edge GCD event_dim + per-side TaskViews.
//   (h) Fork LCM unifies producer-side last3 across all fork branches.
//   (i) Join LCM unifies consumer-side last3 across all join incoming edges.
//   (j) Tag immediate fork-consumer layers with their parent group and
//       branch index for interleaved emission.
//
// Throws std::runtime_error on:
//   - cycles (should be impossible under most-recent-writer for well-formed
//     programs; acts as a defensive guard)
//   - disallowed role combinations (case 2: join-consumer + fork-consumer
//     would need two trigger_events on one task; case 3: join-producer +
//     fork-producer would need two dependent_events on one task)
//   - a fork/join LCM that doesn't divide the shared grid_dim (shouldn't
//     happen if per-edge event_dims were correct GCDs of valid partitions,
//     since the LCM of divisors of N always divides N).
AnnotatedGraph build_annotated_graph(mirage::kernel::Graph const &kn_graph,
                                     TaskConfigMap const &task_configs);

// Optional: dumps the AnnotatedGraph to JSON for debugging when
// MIRAGE_DUMP_ANNOTATED_GRAPH is set. Returns empty string when disabled.
std::string maybe_dump_annotated_graph(AnnotatedGraph const &ag);

} // namespace kernel
} // namespace mirage
