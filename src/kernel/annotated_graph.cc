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
std::array<int, mirage::config::MAX_TENSOR_DIMS> build_partition(
    dim3 const &grid, int3 const &m) {
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
  // ---------------------------------------------------------------------
  std::unordered_map<size_t, std::pair<int, int>> last_writer;

  // Map KNCustomizedOp* -> layer index so downstream passes can locate by op.
  std::unordered_map<KNCustomizedOp const *, int> op_to_layer;

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

    // Read inputs: bind each to its current last_writer (if any).
    for (int in_slot = 0; in_slot < num_inputs; in_slot++) {
      auto *ip = input_ops[in_slot];
      size_t guid = ip->dtensor.guid;
      auto wit = last_writer.find(guid);
      if (wit == last_writer.end()) {
        // Graph input — no edge in the DAG.
        continue;
      }
      int prod_layer = wit->second.first;
      int out_slot = wit->second.second;

      EdgeInfo e;
      e.prod_layer = prod_layer;
      e.cons_layer = layer_idx;
      e.out_slot = out_slot;
      e.in_slot = in_slot;
      e.tensor_guid = guid;
      e.input_map = ip->input_map;

      // Recover output_map from producer's output_op at out_slot.
      auto const *prod_op = ag.layers[prod_layer].op;
      std::vector<tb::TBInputOp *> prod_inputs, prod_outputs;
      split_bgraph_ops(prod_op->bgraph,
                       ag.layers[prod_layer].num_inputs,
                       prod_inputs,
                       prod_outputs);
      if (out_slot < 0 || out_slot >= (int)prod_outputs.size()) {
        throw std::runtime_error(
            "build_annotated_graph: invalid out_slot for producer");
      }
      e.output_map = prod_outputs[out_slot]->input_map;

      int edge_idx = (int)ag.edges.size();
      ag.edges.push_back(e);
      ag.layers[layer_idx].in_edges.push_back(edge_idx);
      ag.layers[prod_layer].out_edges.push_back(edge_idx);
    }

    // Write outputs: update last_writer after inputs are bound.
    for (int out_slot = 0; out_slot < num_outputs; out_slot++) {
      size_t guid = output_ops[out_slot]->dtensor.guid;
      last_writer[guid] = {layer_idx, out_slot};
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
      in_e.erase(std::remove_if(in_e.begin(),
                                in_e.end(),
                                [&](int e) {
                                  return ag.edges[e].is_residual_stripped;
                                }),
                 in_e.end());
      auto &out_e = ag.layers[i].out_edges;
      out_e.erase(std::remove_if(out_e.begin(),
                                 out_e.end(),
                                 [&](int e) {
                                   return ag.edges[e].is_residual_stripped;
                                 }),
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
      throw std::runtime_error(msg.str());
    }
    if (is_join_producer[i] && ag.layers[i].is_fork_producer) {
      std::ostringstream msg;
      msg << "build_annotated_graph: layer " << i
          << " is both a join-producer and a fork-producer (case 3); "
             "a task cannot have two dependent_events.";
      throw std::runtime_error(msg.str());
    }
  }

  // ---------------------------------------------------------------------
  // Step (f): topological ordering (Kahn's). Tiebreaker: prefer the smallest
  // layer index, which preserves original insertion order when unconstrained.
  // ---------------------------------------------------------------------
  {
    std::vector<int> in_deg(V, 0);
    for (int i = 0; i < V; i++) {
      in_deg[i] = (int)ag.layers[i].in_edges.size();
    }
    // min-heap priority queue by layer index
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
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
    auto prod_part = build_partition(prod_op->bgraph.grid_dim, e.output_map);
    auto cons_part = build_partition(cons_op->bgraph.grid_dim, e.input_map);
    for (int d = 0; d < (int)mirage::config::MAX_TENSOR_DIMS; d++) {
      e.event_dim[d] = std::gcd(prod_part[d], cons_part[d]);
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
    fg.outgoing_edges = ag.layers[i].out_edges;

    // N-way LCM per grid axis.
    std::array<int, 3> lcm_last3{};
    for (int g = 0; g < 3; g++) {
      int acc = 1;
      for (int eidx : fg.outgoing_edges) {
        acc = (int)std::lcm<long long>(acc, ag.edges[eidx].producer_side_view.last3[g]);
      }
      lcm_last3[g] = acc;
    }
    // Safety: lcm_last3 must divide producer grid_dim.
    dim3 const &pg = ag.layers[i].op->bgraph.grid_dim;
    if (lcm_last3[0] > (int)pg.x || (int)pg.x % lcm_last3[0] != 0 ||
        lcm_last3[1] > (int)pg.y || (int)pg.y % lcm_last3[1] != 0 ||
        lcm_last3[2] > (int)pg.z || (int)pg.z % lcm_last3[2] != 0) {
      std::ostringstream msg;
      msg << "build_annotated_graph: fork LCM last3 ("
          << lcm_last3[0] << "," << lcm_last3[1] << "," << lcm_last3[2]
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
    jg.incoming_edges = ag.layers[i].in_edges;

    std::array<int, 3> lcm_last3{};
    for (int g = 0; g < 3; g++) {
      int acc = 1;
      for (int eidx : jg.incoming_edges) {
        acc = (int)std::lcm<long long>(acc, ag.edges[eidx].consumer_side_view.last3[g]);
      }
      lcm_last3[g] = acc;
    }
    dim3 const &cg = ag.layers[i].op->bgraph.grid_dim;
    if (lcm_last3[0] > (int)cg.x || (int)cg.x % lcm_last3[0] != 0 ||
        lcm_last3[1] > (int)cg.y || (int)cg.y % lcm_last3[1] != 0 ||
        lcm_last3[2] > (int)cg.z || (int)cg.z % lcm_last3[2] != 0) {
      std::ostringstream msg;
      msg << "build_annotated_graph: join LCM last3 doesn't divide consumer "
             "grid_dim at layer " << i;
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
  os << "AnnotatedGraph: " << ag.layers.size() << " layers, "
     << ag.edges.size() << " edges ("
     << ag.stripped_residual_edges.size() << " residuals stripped), "
     << ag.fork_groups.size() << " fork groups, "
     << ag.join_groups.size() << " join groups\n";
  for (int i = 0; i < (int)ag.layers.size(); i++) {
    auto const &L = ag.layers[i];
    os << "  layer " << i << " in=" << L.in_edges.size()
       << " out=" << L.out_edges.size()
       << (L.is_fork_producer ? " [FORK]" : "")
       << (L.is_join_consumer ? " [JOIN]" : "")
       << (L.fork_parent_group >= 0 ? " [fork-consumer]" : "")
       << "\n";
  }
  return os.str();
}

} // namespace kernel
} // namespace mirage
