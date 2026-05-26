/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include "mirage/kernel/annotated_graph.h"
#include "mirage/threadblock/operator.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace mirage {
namespace kernel {

namespace {

namespace tb = mirage::threadblock;

int axis_lookup(int3 const &m, int g) {
  return (g == 0) ? m.x : (g == 1) ? m.y : m.z;
}

int grid_lookup(dim3 const &d, int g) {
  return (g == 0) ? (int)d.x : (g == 1) ? (int)d.y : (int)d.z;
}

// Compute last3 for a task_view: last3[g] = grid[g] / event_dim[axis_map[g]]
// when axis_map[g] >= 0 (axis partitions a tensor dim that carries events);
// else last3[g] = grid[g] (the axis is replicated across the tensor, so
// every block on this axis observes the whole tensor and falls into the
// same event slot — the whole axis is inside one "task block").
std::array<int, 3> derive_last3(
    std::array<int, mirage::config::MAX_TENSOR_DIMS> const &event_dim,
    dim3 const &grid,
    int3 const &axis_map) {
  std::array<int, 3> last3{};
  for (int g = 0; g < 3; g++) {
    int d = axis_lookup(axis_map, g);
    int gsize = grid_lookup(grid, g);
    if (d >= 0 && d < mirage::config::MAX_TENSOR_DIMS) {
      int ev = event_dim[d];
      last3[g] = (ev > 0) ? (gsize / ev) : gsize;
    } else {
      last3[g] = gsize;
    }
  }
  return last3;
}

// Build per-tensor-dim partition vector from a layer's grid_dim and a map.
std::array<int, mirage::config::MAX_TENSOR_DIMS>
    build_partition(dim3 const &grid, int3 const &m) {
  std::array<int, mirage::config::MAX_TENSOR_DIMS> part{};
  for (int d = 0; d < (int)mirage::config::MAX_TENSOR_DIMS; d++) {
    part[d] = 1;
  }
  if (m.x >= 0 && m.x < (int)mirage::config::MAX_TENSOR_DIMS) {
    part[m.x] = (int)grid.x;
  }
  if (m.y >= 0 && m.y < (int)mirage::config::MAX_TENSOR_DIMS) {
    part[m.y] = (int)grid.y;
  }
  if (m.z >= 0 && m.z < (int)mirage::config::MAX_TENSOR_DIMS) {
    part[m.z] = (int)grid.z;
  }
  return part;
}

// Parse bgraph.operators into (inputs, outputs) by position. Outputs are also
// TB_INPUT_OPs (see runtime.cc:266-274); their input_map field is the
// output_map.
void split_bgraph_ops(tb::Graph const &bgraph,
                      int num_inputs,
                      std::vector<tb::TBInputOp *> &inputs,
                      std::vector<tb::TBInputOp *> &outputs) {
  for (auto const &op : bgraph.operators) {
    if (op->op_type != mirage::type::TB_INPUT_OP) {
      continue;
    }
    auto *ip = static_cast<tb::TBInputOp *>(op);
    if ((int)inputs.size() < num_inputs) {
      inputs.push_back(ip);
    } else {
      outputs.push_back(ip);
    }
  }
}

} // namespace

AnnotatedGraph build_annotated_graph(mirage::kernel::Graph const &kn_graph,
                                     TaskConfigMap const &task_configs) {
  AnnotatedGraph ag;

  // ---------------------------------------------------------------------
  // Step (a): DAG construction with most-recent-writer rule.
  //
  // Why not identify edges by guid alone: qwen3 (and most real MPK models)
  // reuses DTensor buffers across layers — the same guid can be written by
  // multiple customized ops over the course of the graph. Pure guid matching
  // would create spurious cycles (later writer reads the same guid that
  // an earlier reader in a "down" edge also reads).
  //
  // Rule: when layer L reads tensor g, bind the edge to whichever producer
  // wrote g MOST RECENTLY before L. After L is processed, L's outputs update
  // last_writer. Subsequent reads see L as the new writer. This is exactly
  // SSA-style def-use: we're implicitly renaming the reused buffer.
  //
  // Tensor-init writes need a program-order WAW edge to the next writer of the
  // same guid. MPK uses this for MoE zero-fill followed by expert accumulation;
  // without the edge, parallel-path scheduling can run the two writers
  // concurrently even though the old linear schedule made zero-fill visible
  // first. We intentionally do not add WAW edges for arbitrary scratch reuse:
  // RAW edges already protect live ranges, and global WAW edges over-constrain
  // common recycled scratch buffers.
  // ---------------------------------------------------------------------
  // Multi-writer last-writer map, keyed by the underlying storage's GUID
  // (resolve_base_guid). Each entry records the (layer, out_slot) producer
  // plus the byte window [view_offset, view_offset + size_bytes) that the
  // write touched, and whether that producer wrote a view or the full
  // storage tensor.
  //
  // Why a list (not a single writer): write-views allow multiple producers to
  // write disjoint slices of one storage tensor. A subsequent reader that
  // touches the full tensor must depend on ALL those writers; a reader that
  // touches one slice depends only on writers whose windows overlap.
  // C20 (2026-05-17): track each writer's 2D bounding box in the parent
  // storage's element coordinates instead of a single contiguous byte
  // window. A 2D narrow along the inner dim (e.g. q_a slot [0:1536) of
  // qkv_a_out [128, 2176]) writes a STRIDED pattern in memory, not a
  // contiguous byte range — `bytes_size()` over-states the footprint
  // and the old single-axis window check fired on disjoint column
  // slices, producing spurious WAW in_edges (kv_gather seeing q_a
  // rmsnorm as a producer, then failing case-2 fork+join validation).
  // For 2D views/roots we decompose view_offset (bytes) into
  // (row_first, col_first) using the parent row stride that the view
  // already inherits in stride[0]. Higher-D views currently fall back
  // to the conservative col window [0, parent_row_stride).
  struct WriterEntry {
    int layer;
    int out_slot;
    int64_t row_first;  // first row written (elements along outer dim)
    int64_t row_last;   // one past last row
    int64_t col_first;  // first col written (elements along inner dim)
    int64_t col_last;   // one past last col
    bool is_virtual_writer;
  };
  std::unordered_map<size_t, std::vector<WriterEntry>> last_writers;

  // Map KNCustomizedOp* -> layer index so downstream passes can locate by op.
  std::unordered_map<KNCustomizedOp const *, int> op_to_layer;

  // Compute the writer/reader's 2D bbox in the parent's element coordinates.
  // For non-views: row range [0, dim[0]), col range [0, stride[0]). For 2D
  // narrow views (inner-dim narrow): row range [0, dim[0]), col range
  // [view_offset/dtype_size, +dim[1]). For 2D narrow views (outer-dim narrow):
  // row range [view_offset / (stride[0] * dtype_size), +dim[0]), col range
  // [0, stride[0]). Combined narrows decompose via divmod. Higher-rank views
  // collapse all inner dims into one row stride and skip the row check.
  auto compute_bbox = [](DTensor const &dt) {
    struct BBox {
      int64_t row_first;
      int64_t row_last;
      int64_t col_first;
      int64_t col_last;
    };
    size_t dtype_size = mirage::type::get_datatype_size(dt.data_type);
    int64_t row_stride =
        dt.num_dims >= 2 ? static_cast<int64_t>(dt.stride[0]) : 1;
    int64_t dim0 = dt.num_dims >= 1 ? static_cast<int64_t>(dt.dim[0]) : 1;
    int64_t dim_inner = dt.num_dims >= 2
                            ? static_cast<int64_t>(dt.dim[dt.num_dims - 1])
                            : row_stride;
    if (dt.is_virtual() && dtype_size > 0 && row_stride > 0) {
      int64_t view_off_elems =
          dt.view_offset / static_cast<int64_t>(dtype_size);
      int64_t row_first = view_off_elems / row_stride;
      int64_t col_first = view_off_elems % row_stride;
      return BBox{row_first,
                  row_first + dim0,
                  col_first,
                  col_first + dim_inner};
    }
    return BBox{0, dim0, 0, row_stride};
  };

  for (auto const &op : kn_graph.operators) {
    if (op->op_type == mirage::type::KN_INPUT_OP) {
      continue;
    }
    if (op->op_type != mirage::type::KN_CUSTOMIZED_OP) {
      continue;
    }

    auto const *cur_op = dynamic_cast<KNCustomizedOp const *>(op);
    auto it = task_configs.find(op);
    if (it == task_configs.end()) {
      throw std::runtime_error(
          "build_annotated_graph: missing task_config for customized op");
    }
    int num_inputs = std::get<0>(it->second);
    int num_outputs = std::get<1>(it->second);
    mirage::runtime::TaskType task_type = std::get<2>(it->second);
    int variant_id = std::get<3>(it->second);

    std::vector<tb::TBInputOp *> input_ops, output_ops;
    split_bgraph_ops(cur_op->bgraph, num_inputs, input_ops, output_ops);
    if ((int)input_ops.size() != num_inputs ||
        (int)output_ops.size() != num_outputs) {
      throw std::runtime_error(
          "build_annotated_graph: bgraph inputs/outputs count mismatch");
    }

    int layer_idx = (int)ag.layers.size();
    LayerInfo li;
    li.op = cur_op;
    li.task_type = task_type;
    li.variant_id = variant_id;
    li.num_inputs = num_inputs;
    li.num_outputs = num_outputs;
    ag.layers.push_back(li);
    op_to_layer[cur_op] = layer_idx;

    // Read inputs: for each producer-tensor whose window overlaps this read,
    // add an edge. Window analysis is the only place where view semantics
    // change graph construction; event building reads `is_barrier_edge` and
    // emits one coarse event instead of GCD-based per-tile events.
    for (int in_slot = 0; in_slot < num_inputs; in_slot++) {
      auto *ip = input_ops[in_slot];
      DTensor const &cdt = ip->dtensor;
      size_t base = cdt.resolve_base_guid();
      auto wit = last_writers.find(base);
      if (wit == last_writers.end()) {
        // Graph input — no edge in the DAG.
        continue;
      }
      auto rbox = compute_bbox(cdt);
      bool c_is_virtual = cdt.is_virtual();

      // C20 (2026-05-17): shadow-aware edge selection. Walk writers in
      // REVERSE layer order, tracking the still-uncovered column range
      // of the reader. Each writer only contributes an edge for the
      // sub-range it most-recently wrote — later writers shadow earlier
      // ones over the columns they overwrote. Without this, a full
      // qkv_a_proj producer (cols [0, 2176)) stays as a stale producer
      // of every narrow-view consumer downstream even after rmsnorm
      // and other view-writes have fully overwritten the slot they
      // read; downstream MLA kv_gather then sees two distinct
      // producers per slot and trips case-2/case-3 fork/join checks.
      // Row range is assumed full for the readers we currently
      // support (no row-slice views yet); only the column dimension
      // gets fragment-cover tracking.
      std::vector<std::pair<int64_t, int64_t>> uncovered_cols;
      uncovered_cols.push_back({rbox.col_first, rbox.col_last});
      auto const &writers = wit->second;
      for (auto rit = writers.rbegin(); rit != writers.rend(); ++rit) {
        if (uncovered_cols.empty()) {
          break;
        }
        WriterEntry const &we = *rit;
        // Row overlap is a fast reject.
        if (!(we.row_first < rbox.row_last &&
              rbox.row_first < we.row_last)) {
          continue;
        }
        bool wrote_anything = false;
        std::vector<std::pair<int64_t, int64_t>> new_uncovered;
        for (auto const &frag : uncovered_cols) {
          int64_t a = std::max(we.col_first, frag.first);
          int64_t b = std::min(we.col_last, frag.second);
          if (a < b) {
            wrote_anything = true;
            if (frag.first < a) {
              new_uncovered.push_back({frag.first, a});
            }
            if (b < frag.second) {
              new_uncovered.push_back({b, frag.second});
            }
          } else {
            new_uncovered.push_back(frag);
          }
        }
        if (!wrote_anything) {
          continue;
        }
        uncovered_cols = std::move(new_uncovered);

        EdgeInfo e;
        e.prod_layer = we.layer;
        e.cons_layer = layer_idx;
        e.out_slot = we.out_slot;
        e.in_slot = in_slot;
        e.tensor_guid = cdt.guid;
        e.input_map = ip->input_map;
        e.is_barrier_edge = c_is_virtual || we.is_virtual_writer;

        auto const *prod_op = ag.layers[we.layer].op;
        std::vector<tb::TBInputOp *> prod_inputs, prod_outputs;
        split_bgraph_ops(prod_op->bgraph,
                         ag.layers[we.layer].num_inputs,
                         prod_inputs,
                         prod_outputs);
        if (we.out_slot < 0 || we.out_slot >= (int)prod_outputs.size()) {
          throw std::runtime_error(
              "build_annotated_graph: invalid out_slot for producer");
        }
        e.output_map = prod_outputs[we.out_slot]->input_map;

        int edge_idx = (int)ag.edges.size();
        ag.edges.push_back(e);
        ag.layers[layer_idx].in_edges.push_back(edge_idx);
        ag.layers[we.layer].out_edges.push_back(edge_idx);
      }
    }

    // Write outputs: update last_writers after inputs are bound.
    // - Non-virtual writes overwrite the full storage tensor: clear all prior
    //   writers and start fresh with a single full-window entry.
    // - Virtual (write-view) writes append a partial entry so multiple
    //   producers writing disjoint slices coexist.
    for (int out_slot = 0; out_slot < num_outputs; out_slot++) {
      DTensor const &odt = output_ops[out_slot]->dtensor;
      size_t base = odt.resolve_base_guid();
      bool o_is_virtual = odt.is_virtual();
      auto wbox = compute_bbox(odt);

      // Preserve the TASK_TENSOR_INIT WAW behaviour: if the previous writer
      // was a zero-fill (init) for the SAME tensor (matching guid + window),
      // add an explicit WAW edge so the next writer waits for init.
      if (!o_is_virtual) {
        auto wit = last_writers.find(base);
        if (wit != last_writers.end()) {
          for (WriterEntry const &we : wit->second) {
            if (we.is_virtual_writer) {
              continue;
            }
            if (ag.layers[we.layer].task_type !=
                mirage::runtime::TASK_TENSOR_INIT) {
              continue;
            }
            bool duplicate_edge = false;
            for (int eidx : ag.layers[layer_idx].in_edges) {
              EdgeInfo const &existing = ag.edges[eidx];
              if (existing.prod_layer == we.layer &&
                  existing.out_slot == we.out_slot &&
                  existing.tensor_guid == static_cast<size_t>(odt.guid)) {
                duplicate_edge = true;
                break;
              }
            }
            if (!duplicate_edge) {
              EdgeInfo e;
              e.prod_layer = we.layer;
              e.cons_layer = layer_idx;
              e.out_slot = we.out_slot;
              e.in_slot = -1;
              e.tensor_guid = odt.guid;
              e.input_map = output_ops[out_slot]->input_map;
              auto const *prod_op = ag.layers[we.layer].op;
              std::vector<tb::TBInputOp *> prod_inputs, prod_outputs;
              split_bgraph_ops(prod_op->bgraph,
                               ag.layers[we.layer].num_inputs,
                               prod_inputs,
                               prod_outputs);
              if (we.out_slot < 0 ||
                  we.out_slot >= (int)prod_outputs.size()) {
                throw std::runtime_error(
                    "build_annotated_graph: invalid out_slot for WAW producer");
              }
              e.output_map = prod_outputs[we.out_slot]->input_map;
              // WAW edges keep the non-virtual semantics they had before;
              // is_barrier_edge stays false unless we explicitly need it.
              int edge_idx = (int)ag.edges.size();
              ag.edges.push_back(e);
              ag.layers[layer_idx].in_edges.push_back(edge_idx);
              ag.layers[we.layer].out_edges.push_back(edge_idx);
            }
          }
        }
      }

      WriterEntry we{layer_idx,
                     out_slot,
                     wbox.row_first,
                     wbox.row_last,
                     wbox.col_first,
                     wbox.col_last,
                     o_is_virtual};
      if (!o_is_virtual) {
        // A full-storage write supersedes any prior writers (view or full).
        last_writers[base].clear();
      }
      last_writers[base].push_back(we);
    }
  }

  int const V = (int)ag.layers.size();

  // ---------------------------------------------------------------------
  // Step (b): cycle detection via Kahn's algorithm.
  // ---------------------------------------------------------------------
  {
    std::vector<int> in_deg(V, 0);
    for (auto const &e : ag.edges) {
      in_deg[e.cons_layer]++;
    }
    std::queue<int> q;
    for (int i = 0; i < V; i++) {
      if (in_deg[i] == 0) {
        q.push(i);
      }
    }
    int processed = 0;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      processed++;
      for (int eidx : ag.layers[u].out_edges) {
        int v = ag.edges[eidx].cons_layer;
        if (--in_deg[v] == 0) {
          q.push(v);
        }
      }
    }
    if (processed != V) {
      std::ostringstream msg;
      msg << "build_annotated_graph: cycle detected; offenders:";
      for (int i = 0; i < V; i++) {
        if (in_deg[i] > 0) {
          msg << " layer " << i;
        }
      }
      throw std::runtime_error(msg.str());
    }
  }

  // ---------------------------------------------------------------------
  // Step (c): residual stripping.
  //
  // A "residual" edge u->v is the direct shortcut in a transformer-like
  // pattern: u forks to a computed path u->w->...->v and also directly to
  // v (where v adds the residual). We strip the direct edge because the
  // longer path's chain of events transitively forces u to complete before
  // v starts, so the direct edge contributes no scheduling information —
  // keeping it would just force v to be classified as a join-consumer and
  // potentially propagate a case-2 violation.
  //
  // Single-shot semantics matter: if we stripped iteratively we might drop
  // the longer path before the direct edge, leaving the direct edge intact
  // (and the pattern undetected). So we compute reachability ONCE on the
  // original edge set, then mark all residuals to strip, then remove them
  // atomically.
  //
  // Cost: O(V * (V+E)) for the BFS; for qwen3 (~300 layers, ~400 edges)
  // this is sub-millisecond and doesn't merit a smarter transitive-reduction
  // algorithm. Revisit if V grows past ~50k.
  // ---------------------------------------------------------------------
  std::vector<std::unordered_set<int>> reachable(V);
  for (int s = 0; s < V; s++) {
    // BFS from s
    std::queue<int> q;
    q.push(s);
    std::vector<char> seen(V, 0);
    seen[s] = 1;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int eidx : ag.layers[u].out_edges) {
        int v = ag.edges[eidx].cons_layer;
        if (!seen[v]) {
          seen[v] = 1;
          reachable[s].insert(v);
          q.push(v);
        }
      }
    }
  }

  {
    std::vector<char> strip_flag(ag.edges.size(), 0);
    for (size_t eidx = 0; eidx < ag.edges.size(); eidx++) {
      auto const &e = ag.edges[eidx];
      // Never strip a barrier edge — a longer non-barrier path provides only
      // fine-grained per-tile synchronization on its constituent edges, which
      // does not transitively imply "all of u finished before v starts" that
      // the barrier guarantees.
      if (e.is_barrier_edge) {
        continue;
      }
      int u = e.prod_layer, v = e.cons_layer;
      // Does any intermediate w (successor of u other than v) reach v?
      for (int oe : ag.layers[u].out_edges) {
        if ((size_t)oe == eidx) {
          continue;
        }
        int w = ag.edges[oe].cons_layer;
        if (w == v) {
          continue;
        }
        if (reachable[w].count(v) > 0) {
          strip_flag[eidx] = 1;
          break;
        }
      }
    }
    // Remove stripped edges from per-layer in_edges / out_edges. Keep the
    // edge records in ag.edges (indices remain valid); mark stripped.
    for (size_t eidx = 0; eidx < ag.edges.size(); eidx++) {
      if (strip_flag[eidx]) {
        ag.edges[eidx].is_residual_stripped = true;
        ag.stripped_residual_edges.push_back(ag.edges[eidx]);
      }
    }
    for (int i = 0; i < V; i++) {
      auto &in_e = ag.layers[i].in_edges;
      in_e.erase(std::remove_if(
                     in_e.begin(),
                     in_e.end(),
                     [&](int e) { return ag.edges[e].is_residual_stripped; }),
                 in_e.end());
      auto &out_e = ag.layers[i].out_edges;
      out_e.erase(std::remove_if(
                      out_e.begin(),
                      out_e.end(),
                      [&](int e) { return ag.edges[e].is_residual_stripped; }),
                  out_e.end());
    }
  }

  // ---------------------------------------------------------------------
  // Step (d): per-layer role classification (post-strip).
  //
  // Why "distinct" layers (not raw edge count): a layer can produce >1
  // output tensor, all feeding the same consumer (e.g. a fused attention
  // layer with 2 outputs that both feed the next block). Counting out_edges
  // raw would flag this as a fork with trivially redundant branches,
  // creating a false case-3 violation when the real dependency model is a
  // plain chain. This showed up when dry-running qwen3 and was the cause
  // of the first wave of spurious compile errors.
  // ---------------------------------------------------------------------
  // Barrier edges (view-induced) participate in classification just like
  // fine-grained edges. After step (g) gives a barrier edge event_dim=1
  // (and hence last3 = full grid_dim), the fork/join LCM uniformly
  // degrades a mixed bundle to a single event, which is exactly the
  // barrier semantic. This routes sibling write-views or sibling
  // read-views of one parent through the standard fork/join paths
  // instead of an ad-hoc barrier branch.
  for (int i = 0; i < V; i++) {
    std::unordered_set<int> distinct_cons, distinct_prod;
    for (int eidx : ag.layers[i].out_edges) {
      distinct_cons.insert(ag.edges[eidx].cons_layer);
    }
    for (int eidx : ag.layers[i].in_edges) {
      distinct_prod.insert(ag.edges[eidx].prod_layer);
    }
    ag.layers[i].is_fork_producer = distinct_cons.size() >= 2;
    ag.layers[i].is_join_consumer = distinct_prod.size() >= 2;
  }

  // ---------------------------------------------------------------------
  // Step (e): case 2 / 3 validation.
  //
  // FullTaskDesc has exactly one `trigger_event` slot (the event a task
  // fires on completion) and one `dependent_event` slot (the event a task
  // waits for, post-prelaunch). The disallowed combinations are the ones
  // that would need two of either slot on the same task:
  //
  //   Case 2 (join-consumer + fork-consumer): X's tasks would need to be
  //     triggered by TWO events (the upstream fork event and the join
  //     event at X itself). Not representable.
  //   Case 3 (join-producer + fork-producer): L's tasks would need to
  //     FIRE two events (the fork event at L and the downstream join
  //     event). Not representable.
  //
  // Note: cases 2 and 3 always co-occur — a case-2 violation at X implies
  // one of X's producers is a case-3 violator (its edge to X makes it a
  // join-producer, and its multiple consumers make it a fork-producer).
  // We detect whichever comes first in layer order and reject.
  //
  // is_fork_consumer[L] := any in-edge comes from a fork-producer.
  // is_join_producer[L] := any out-edge goes to a join-consumer.
  // ---------------------------------------------------------------------
  std::vector<char> is_fork_consumer(V, 0), is_join_producer(V, 0);
  for (int i = 0; i < V; i++) {
    for (int eidx : ag.layers[i].in_edges) {
      if (ag.layers[ag.edges[eidx].prod_layer].is_fork_producer) {
        is_fork_consumer[i] = 1;
        break;
      }
    }
    for (int eidx : ag.layers[i].out_edges) {
      if (ag.layers[ag.edges[eidx].cons_layer].is_join_consumer) {
        is_join_producer[i] = 1;
        break;
      }
    }
  }
  for (int i = 0; i < V; i++) {
    if (ag.layers[i].is_join_consumer && is_fork_consumer[i]) {
      std::ostringstream msg;
      msg << "build_annotated_graph: layer " << i
          << " is both a join-consumer and a fork-consumer (case 2); "
             "a task cannot have two trigger_events.";
      msg << " task_type=" << static_cast<int>(ag.layers[i].task_type)
          << " in_edges=";
      for (int eidx : ag.layers[i].in_edges) {
        auto const &e = ag.edges[eidx];
        msg << " [" << e.prod_layer << ":" << e.out_slot << "->" << e.cons_layer
            << ":" << e.in_slot << " guid=" << e.tensor_guid << "]";
      }
      msg << " out_edges=";
      for (int eidx : ag.layers[i].out_edges) {
        auto const &e = ag.edges[eidx];
        msg << " [" << e.prod_layer << ":" << e.out_slot << "->" << e.cons_layer
            << ":" << e.in_slot << " guid=" << e.tensor_guid << "]";
      }
      throw std::runtime_error(msg.str());
    }
    if (is_join_producer[i] && ag.layers[i].is_fork_producer) {
      std::ostringstream msg;
      msg << "build_annotated_graph: layer " << i
          << " is both a join-producer and a fork-producer (case 3); "
             "a task cannot have two dependent_events.";
      msg << " task_type=" << static_cast<int>(ag.layers[i].task_type)
          << " in_edges=";
      for (int eidx : ag.layers[i].in_edges) {
        auto const &e = ag.edges[eidx];
        msg << " [" << e.prod_layer << ":" << e.out_slot << "->" << e.cons_layer
            << ":" << e.in_slot << " guid=" << e.tensor_guid << "]";
      }
      msg << " out_edges=";
      for (int eidx : ag.layers[i].out_edges) {
        auto const &e = ag.edges[eidx];
        msg << " [" << e.prod_layer << ":" << e.out_slot << "->" << e.cons_layer
            << ":" << e.in_slot << " guid=" << e.tensor_guid << "]";
      }
      // Diagnostic: dump each consumer's in_edges so we can see which
      // consumer is the join-consumer that made this layer a
      // join-producer.
      for (int eidx : ag.layers[i].out_edges) {
        int cons = ag.edges[eidx].cons_layer;
        if (cons < 0 || cons >= (int)ag.layers.size()) {
          continue;
        }
        auto const &CL = ag.layers[cons];
        msg << "\n  consumer L" << cons
            << " task_type=" << static_cast<int>(CL.task_type)
            << (CL.is_join_consumer ? " [JOIN-CONSUMER]" : "")
            << " in=" << CL.in_edges.size() << ":";
        for (int ie : CL.in_edges) {
          auto const &ce = ag.edges[ie];
          msg << " [" << ce.prod_layer << ":" << ce.out_slot << "->" << cons
              << ":" << ce.in_slot << " guid=" << ce.tensor_guid << "]";
        }
      }
      // Extra diagnostic: for the first non-join-consumer fork-out
      // sibling of i (typically L1 = fused task), dump its forward
      // reachable set + out_edges so we can see why the strip pass
      // didn't remove i->{join-consumer} (it should have, if the
      // sibling reaches the join-consumer via a longer path).
      for (int eidx : ag.layers[i].out_edges) {
        int cons = ag.edges[eidx].cons_layer;
        if (cons < 0 || cons >= (int)ag.layers.size()) {
          continue;
        }
        if (ag.layers[cons].is_join_consumer) {
          continue;
        }
        msg << "\n  sibling L" << cons
            << " task_type=" << static_cast<int>(ag.layers[cons].task_type)
            << " out_edges:";
        for (int oe : ag.layers[cons].out_edges) {
          auto const &oce = ag.edges[oe];
          msg << " [L" << cons << ":" << oce.out_slot << "->L" << oce.cons_layer
              << ":" << oce.in_slot << " guid=" << oce.tensor_guid << "]";
        }
        break;
      }
      throw std::runtime_error(msg.str());
    }
  }

  // ---------------------------------------------------------------------
  // Step (f): topological ordering (Kahn's) with a (depth, index) tie-break.
  //
  // Why depth, not just layer index: for a symmetric fork like
  // A -> B -> C -> D and A -> E -> F -> G, an index-only tie-break yields
  // [A, B, C, D, E, F, G] (chain alpha emitted fully before chain beta).
  // That means C and F's chain events end up on opposite ends of the
  // prelaunch queue, even though they are ready to run at the same depth.
  // Using depth-first-ascending, index-second gives [A, B, E, C, F, D, G]:
  // within each depth stratum we still respect original insertion order,
  // but across strata we advance both chains in lockstep. B and E are
  // (required) interleaved within a single fork event by the bundle-head
  // emission (runtime.cc); the downstream layers C/F/D/G are interleaved
  // here via the topo order. For pure chain graphs (e.g. qwen3 after
  // residual stripping) depth == layer index, so the output matches the
  // previous insertion-order behaviour.
  //
  // Depth is defined as 0 for layers with no incoming edge, and
  // depth[v] = max_{(u, v) in E} (depth[u] + 1) otherwise. We compute it
  // by a first Kahn pass (any valid order suffices) and then run a second
  // Kahn pass using the depth array as the primary tie-break key.
  // ---------------------------------------------------------------------
  std::vector<int> depth(V, 0);
  {
    // First pass: any valid topo order to compute depth.
    std::vector<int> in_deg_tmp(V, 0);
    for (int i = 0; i < V; i++) {
      in_deg_tmp[i] = (int)ag.layers[i].in_edges.size();
    }
    std::queue<int> q;
    for (int i = 0; i < V; i++) {
      if (in_deg_tmp[i] == 0) {
        q.push(i);
      }
    }
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int eidx : ag.layers[u].out_edges) {
        int v = ag.edges[eidx].cons_layer;
        depth[v] = std::max(depth[v], depth[u] + 1);
        if (--in_deg_tmp[v] == 0) {
          q.push(v);
        }
      }
    }
  }
  {
    // Second pass: min-heap keyed by (depth, index). Smaller depth first;
    // within the same depth, smaller original-insertion index first.
    std::vector<int> in_deg(V, 0);
    for (int i = 0; i < V; i++) {
      in_deg[i] = (int)ag.layers[i].in_edges.size();
    }
    auto cmp = [&depth](int a, int b) {
      if (depth[a] != depth[b]) {
        return depth[a] > depth[b];
      }
      return a > b;
    };
    std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);
    for (int i = 0; i < V; i++) {
      if (in_deg[i] == 0) {
        pq.push(i);
      }
    }
    while (!pq.empty()) {
      int u = pq.top();
      pq.pop();
      ag.ordered_layers.push_back(u);
      for (int eidx : ag.layers[u].out_edges) {
        int v = ag.edges[eidx].cons_layer;
        if (--in_deg[v] == 0) {
          pq.push(v);
        }
      }
    }
    if ((int)ag.ordered_layers.size() != V) {
      throw std::runtime_error(
          "build_annotated_graph: topo order incomplete after strip");
    }
  }

  // ---------------------------------------------------------------------
  // Step (g): per-edge GCD event_dim + producer/consumer task_views.
  // ---------------------------------------------------------------------
  for (auto &e : ag.edges) {
    if (e.is_residual_stripped) {
      continue;
    }
    auto const *prod_op = ag.layers[e.prod_layer].op;
    auto const *cons_op = ag.layers[e.cons_layer].op;
    if (e.is_barrier_edge) {
      // View-induced barrier: event_dim = (1, 1, ..., 1) yields last3 =
      // full grid_dim on both sides, i.e. ONE event per edge spanning all
      // producer tasks and launching all consumer tasks. Downstream fork
      // and join LCM passes automatically degrade any mixed bundle that
      // contains this edge to a single event, which is exactly the
      // coarse-barrier semantic we want for views.
      for (int d = 0; d < (int)mirage::config::MAX_TENSOR_DIMS; d++) {
        e.event_dim[d] = 1;
      }
    } else {
      auto prod_part = build_partition(prod_op->bgraph.grid_dim, e.output_map);
      auto cons_part = build_partition(cons_op->bgraph.grid_dim, e.input_map);
      for (int d = 0; d < (int)mirage::config::MAX_TENSOR_DIMS; d++) {
        e.event_dim[d] = std::gcd(prod_part[d], cons_part[d]);
      }
    }
    e.producer_side_view.event_dim = e.event_dim;
    e.producer_side_view.grid_dim = prod_op->bgraph.grid_dim;
    e.producer_side_view.axis_map = e.output_map;
    e.producer_side_view.last3 =
        derive_last3(e.event_dim, prod_op->bgraph.grid_dim, e.output_map);

    e.consumer_side_view.event_dim = e.event_dim;
    e.consumer_side_view.grid_dim = cons_op->bgraph.grid_dim;
    e.consumer_side_view.axis_map = e.input_map;
    e.consumer_side_view.last3 =
        derive_last3(e.event_dim, cons_op->bgraph.grid_dim, e.input_map);
  }

  // ---------------------------------------------------------------------
  // Step (h): fork LCM pass.
  //
  // For each fork producer F, unify the producer-side last3 across all
  // outgoing branches. The "factor" we gain on last3 (= lcm / branch_last3)
  // is absorbed by REDUCING each branch's event_dim on the tensor dim that
  // maps to that grid axis; so the number of events shrinks and each event
  // now triggers a larger block of producer/consumer tasks.
  //
  // Why grid-axis LCM: F's grid is shared across branches, but each branch
  // has its own bridging tensor (possibly different output slots of F) and
  // its own consumer grid/input_map. Grid-axis space is the common frame;
  // tensor-dim space is per-edge. LCM on grid axes + propagation through
  // output_map/input_map gives a consistent per-branch event_dim.
  //
  // Safety: lcm_last3[g] must divide F.grid_dim[g]. This is guaranteed
  // for well-formed inputs because each branch_last3[g] divides grid_dim[g]
  // (it's grid_dim / event_dim where event_dim is a GCD of divisors of
  // grid_dim), and the LCM of divisors of N is a divisor of N.
  // ---------------------------------------------------------------------
  for (int i = 0; i < V; i++) {
    if (!ag.layers[i].is_fork_producer) {
      continue;
    }
    ForkGroupInfo fg;
    fg.producer_layer = i;
    std::unordered_set<int> seen_consumers;
    for (int eidx : ag.layers[i].out_edges) {
      int cons_layer = ag.edges[eidx].cons_layer;
      if (seen_consumers.insert(cons_layer).second) {
        fg.outgoing_edges.push_back(eidx);
      }
    }

    // N-way LCM per grid axis.
    std::array<int, 3> lcm_last3{};
    for (int g = 0; g < 3; g++) {
      int acc = 1;
      for (int eidx : fg.outgoing_edges) {
        acc = (int)std::lcm<long long>(
            acc, ag.edges[eidx].producer_side_view.last3[g]);
      }
      lcm_last3[g] = acc;
    }
    // Safety: lcm_last3 must divide producer grid_dim.
    dim3 const &pg = ag.layers[i].op->bgraph.grid_dim;
    if (lcm_last3[0] > (int)pg.x || (int)pg.x % lcm_last3[0] != 0 ||
        lcm_last3[1] > (int)pg.y || (int)pg.y % lcm_last3[1] != 0 ||
        lcm_last3[2] > (int)pg.z || (int)pg.z % lcm_last3[2] != 0) {
      std::ostringstream msg;
      msg << "build_annotated_graph: fork LCM last3 (" << lcm_last3[0] << ","
          << lcm_last3[1] << "," << lcm_last3[2]
          << ") does not divide producer grid_dim (" << pg.x << "," << pg.y
          << "," << pg.z << ") at layer " << i;
      throw std::runtime_error(msg.str());
    }
    fg.lcm_last3 = lcm_last3;

    int fg_id = (int)ag.fork_groups.size();
    for (size_t b = 0; b < fg.outgoing_edges.size(); b++) {
      int eidx = fg.outgoing_edges[b];
      auto &e = ag.edges[eidx];
      e.fork_group_id = fg_id;
      // Reduce event_dim on tensor dims corresponding to producer's grid axes
      // where the scale factor lives.
      for (int g = 0; g < 3; g++) {
        int branch_last = e.producer_side_view.last3[g];
        int scale = lcm_last3[g] / std::max(branch_last, 1);
        if (scale <= 1) {
          continue;
        }
        int d = axis_lookup(e.output_map, g);
        if (d < 0 || d >= (int)mirage::config::MAX_TENSOR_DIMS) {
          // Replicated axis cannot absorb a factor; this means branch.last3[g]
          // already equals grid[g] and another branch demands more along g,
          // which would require > grid[g]. Our safety check above rules this
          // out (LCM divides grid).
          continue;
        }
        int &ev = e.event_dim[d];
        if (ev % scale != 0) {
          throw std::runtime_error(
              "build_annotated_graph: fork LCM scale doesn't divide event_dim");
        }
        ev /= scale;
      }
      // Recompute producer-side view with lcm_last3 (unified across branches).
      e.producer_side_view.event_dim = e.event_dim;
      e.producer_side_view.last3 = lcm_last3;
      // Recompute consumer-side view with new event_dim via consumer's
      // input_map.
      auto const *cons_op = ag.layers[e.cons_layer].op;
      e.consumer_side_view.event_dim = e.event_dim;
      e.consumer_side_view.grid_dim = cons_op->bgraph.grid_dim;
      e.consumer_side_view.axis_map = e.input_map;
      e.consumer_side_view.last3 =
          derive_last3(e.event_dim, cons_op->bgraph.grid_dim, e.input_map);
    }
    ag.fork_groups.push_back(fg);
  }

  // ---------------------------------------------------------------------
  // Step (i): join LCM pass. Symmetric — unify consumer-side last3 on the
  // join-consumer's grid axes across incoming branches.
  // ---------------------------------------------------------------------
  for (int i = 0; i < V; i++) {
    if (!ag.layers[i].is_join_consumer) {
      continue;
    }
    JoinGroupInfo jg;
    jg.consumer_layer = i;
    std::unordered_set<int> seen_producers;
    for (int eidx : ag.layers[i].in_edges) {
      int prod_layer = ag.edges[eidx].prod_layer;
      if (seen_producers.insert(prod_layer).second) {
        jg.incoming_edges.push_back(eidx);
      }
    }

    std::array<int, 3> lcm_last3{};
    for (int g = 0; g < 3; g++) {
      int acc = 1;
      for (int eidx : jg.incoming_edges) {
        acc = (int)std::lcm<long long>(
            acc, ag.edges[eidx].consumer_side_view.last3[g]);
      }
      lcm_last3[g] = acc;
    }
    dim3 const &cg = ag.layers[i].op->bgraph.grid_dim;
    if (lcm_last3[0] > (int)cg.x || (int)cg.x % lcm_last3[0] != 0 ||
        lcm_last3[1] > (int)cg.y || (int)cg.y % lcm_last3[1] != 0 ||
        lcm_last3[2] > (int)cg.z || (int)cg.z % lcm_last3[2] != 0) {
      std::ostringstream msg;
      msg << "build_annotated_graph: join LCM last3 doesn't divide consumer "
             "grid_dim at layer "
          << i;
      throw std::runtime_error(msg.str());
    }
    jg.lcm_last3 = lcm_last3;

    int jg_id = (int)ag.join_groups.size();
    for (size_t b = 0; b < jg.incoming_edges.size(); b++) {
      int eidx = jg.incoming_edges[b];
      auto &e = ag.edges[eidx];
      e.join_group_id = jg_id;
      for (int g = 0; g < 3; g++) {
        int branch_last = e.consumer_side_view.last3[g];
        int scale = lcm_last3[g] / std::max(branch_last, 1);
        if (scale <= 1) {
          continue;
        }
        int d = axis_lookup(e.input_map, g);
        if (d < 0 || d >= (int)mirage::config::MAX_TENSOR_DIMS) {
          continue;
        }
        int &ev = e.event_dim[d];
        if (ev % scale != 0) {
          throw std::runtime_error(
              "build_annotated_graph: join LCM scale doesn't divide event_dim");
        }
        ev /= scale;
      }
      e.consumer_side_view.event_dim = e.event_dim;
      e.consumer_side_view.last3 = lcm_last3;
      auto const *prod_op = ag.layers[e.prod_layer].op;
      e.producer_side_view.event_dim = e.event_dim;
      e.producer_side_view.grid_dim = prod_op->bgraph.grid_dim;
      e.producer_side_view.axis_map = e.output_map;
      e.producer_side_view.last3 =
          derive_last3(e.event_dim, prod_op->bgraph.grid_dim, e.output_map);
    }
    ag.join_groups.push_back(jg);
  }

  // ---------------------------------------------------------------------
  // Step (j): finalize — tag immediate fork-consumer layers with their
  // parent fork group + branch index.
  // ---------------------------------------------------------------------
  for (int fg_id = 0; fg_id < (int)ag.fork_groups.size(); fg_id++) {
    auto const &fg = ag.fork_groups[fg_id];
    for (size_t b = 0; b < fg.outgoing_edges.size(); b++) {
      int eidx = fg.outgoing_edges[b];
      int cons = ag.edges[eidx].cons_layer;
      ag.layers[cons].fork_parent_group = fg_id;
      ag.layers[cons].fork_branch_index = (int)b;
    }
  }

  return ag;
}

std::string maybe_dump_annotated_graph(AnnotatedGraph const &ag) {
  char const *env = std::getenv("MIRAGE_DUMP_ANNOTATED_GRAPH");
  if (env == nullptr || std::string(env) == "0") {
    return "";
  }
  std::ostringstream os;
  os << "AnnotatedGraph: " << ag.layers.size() << " layers, " << ag.edges.size()
     << " edges (" << ag.stripped_residual_edges.size()
     << " residuals stripped), " << ag.fork_groups.size() << " fork groups, "
     << ag.join_groups.size() << " join groups\n";
  for (int i = 0; i < (int)ag.layers.size(); i++) {
    auto const &L = ag.layers[i];
    os << "  layer " << i << " in=" << L.in_edges.size()
       << " out=" << L.out_edges.size() << (L.is_fork_producer ? " [FORK]" : "")
       << (L.is_join_consumer ? " [JOIN]" : "")
       << (L.fork_parent_group >= 0 ? " [fork-consumer]" : "") << "\n";
  }
  os << "  edges:\n";
  for (size_t i = 0; i < ag.edges.size(); i++) {
    auto const &e = ag.edges[i];
    os << "    [" << i << "] " << e.prod_layer << ":" << e.out_slot << " -> "
       << e.cons_layer << ":" << e.in_slot << " guid=" << e.tensor_guid
       << (e.is_barrier_edge ? " [BARRIER]" : "")
       << (e.is_residual_stripped ? " [STRIPPED]" : "") << "\n";
  }
  os << "  ordered_layers: [";
  for (size_t i = 0; i < ag.ordered_layers.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << ag.ordered_layers[i];
  }
  os << "]\n";
  return os.str();
}

} // namespace kernel
} // namespace mirage
