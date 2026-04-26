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

#include "mirage/kernel/annotated_graph.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/task_register.h"
#include "mirage/transpiler/utils.h"
#include "mirage/utils/json_utils.h"
#include <queue>

namespace mirage {
namespace kernel {

using namespace mirage::runtime;
namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

bool is_nvshmem_event(size_t event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

struct Dim3Comparator {
  bool operator()(dim3 const &a, dim3 const &b) const {
    if (a.x != b.x) {
      return a.x < b.x;
    }
    if (a.y != b.y) {
      return a.y < b.y;
    }
    return a.z < b.z;
  }
};

/**
 * Get number of subtasks for a task type.
 *
 * For each (bid.x, bid.y, bid.z), there could be multiple subtasks associated
 * with it. These subtasks can be executed on different workers, just like
 * normal tasks. The biggest difference is that these subtasks share the same
 * input/output tensors.
 *
 * For example, for NVSHMEM_ALLGATHER_STRIDED_PUT task, each (bid.x, bid.y,
 * bid.z) will have (num_gpus - 1) subtasks, each subtask is responsible for
 * putting data to one of the other GPUs.
 */
int get_num_subtasks(int num_gpus, TaskType task_type) {
  // TODO(Zepeng) Re-consider this design. Try if task coalescing can result in
  // better performance.
  if (task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT) {
    return num_gpus - 1;
  } else {
    return 1;
  }
}

void dfs_create_events_add_tasks(
    int depth,
    int const my_gpu_id,
    int const num_gpus,
    std::vector<int> const &event_dims,
    int3 const input_map,
    int3 const output_map,
    dim3 const consumer_grid_dim,
    dim3 const producer_grid_dim,
    dim3 consumer_lo_bid,
    dim3 consumer_hi_bid,
    dim3 producer_lo_bid,
    dim3 producer_hi_bid,
    std::vector<EventDesc> &all_events,
    std::vector<FullTaskDesc> &all_tasks,
    std::vector<FullTaskDesc> const &cur_op_tasks,
    std::map<dim3, std::vector<TaskId>, Dim3Comparator> const &pre_task_map,
    std::map<dim3, std::vector<TaskId>, Dim3Comparator> &cur_task_map,
    std::unordered_set<size_t> &nvshmem_events_idx,
    bool nvshmem_event,
    bool multigpu_task) {
  if (depth >= mirage::config::MAX_TENSOR_DIMS) {
    EventDesc event_desc;
    event_desc.num_triggers = 0;
    event_desc.first_task_id = all_tasks.size();
    // Add consumer tasks
    dim3 bid;
    for (bid.x = consumer_lo_bid.x; bid.x < consumer_hi_bid.x; bid.x++) {
      for (bid.y = consumer_lo_bid.y; bid.y < consumer_hi_bid.y; bid.y++) {
        for (bid.z = consumer_lo_bid.z; bid.z < consumer_hi_bid.z; bid.z++) {
          int block_offset = bid.x * consumer_grid_dim.y * consumer_grid_dim.z +
                             bid.y * consumer_grid_dim.z + bid.z;
          if (multigpu_task) {
            cur_task_map[bid] = std::vector<TaskId>();
            for (int i = 0; i < num_gpus - 1; i++) {
              cur_task_map[bid].push_back(all_tasks.size());
              all_tasks.push_back(
                  cur_op_tasks[block_offset * (num_gpus - 1) + i]);
            }
          } else {
            cur_task_map[bid] = std::vector<TaskId>{all_tasks.size()};
            all_tasks.push_back(cur_op_tasks[block_offset]);
          }
        }
      }
    }
    event_desc.last_task_id = all_tasks.size();
    // Set producer tasks
    for (bid.x = producer_lo_bid.x; bid.x < producer_hi_bid.x; bid.x++) {
      for (bid.y = producer_lo_bid.y; bid.y < producer_hi_bid.y; bid.y++) {
        for (bid.z = producer_lo_bid.z; bid.z < producer_hi_bid.z; bid.z++) {
          assert(pre_task_map.find(bid) != pre_task_map.end());
          std::vector<TaskId> const &task_ids = pre_task_map.find(bid)->second;
          if (all_tasks[task_ids[0]].task_type ==
              TASK_NVSHMEM_ALLGATHER_STRIDED_PUT) {
            assert(task_ids.size() == (size_t)num_gpus - 1);
            for (int tgt_gpu_id = 0; tgt_gpu_id < num_gpus; tgt_gpu_id++) {
              if (tgt_gpu_id == my_gpu_id) {
                continue;
              }
              size_t idx = tgt_gpu_id < my_gpu_id ? tgt_gpu_id : tgt_gpu_id - 1;
              assert(all_tasks[task_ids[idx]].task_type ==
                     TASK_NVSHMEM_ALLGATHER_STRIDED_PUT);
              all_tasks[task_ids[idx]].trigger_event =
                  get_event_id(tgt_gpu_id,
                               all_events.size(),
                               nvshmem_event /*nvshmem_event*/);
              event_desc.num_triggers++;
            }
          } else {
            assert(task_ids.size() == 1);
            all_tasks[task_ids[0]].trigger_event = get_event_id(
                my_gpu_id, all_events.size(), nvshmem_event /*nvshmem_event*/);
            event_desc.num_triggers++;
          }
        }
      }
    }
    if (nvshmem_event) {
      // NVSHMEM events need to be triggered by all other GPUs
      nvshmem_events_idx.insert(all_events.size());
    }
    event_desc.event_type =
        event_desc.last_task_id >= event_desc.first_task_id + 8
            ? EVENT_LAUNCH_MASSIVE_TASKS
            : EVENT_LAUNCH_TASKS;
    all_events.push_back(event_desc);
  } else {
    for (int i = 0; i < event_dims[depth]; i++) {
      dim3 new_consumer_lo_bid = consumer_lo_bid;
      dim3 new_consumer_hi_bid = consumer_hi_bid;
      dim3 new_producer_lo_bid = producer_lo_bid;
      dim3 new_producer_hi_bid = producer_hi_bid;
      if (depth == input_map.x) {
        int factor = consumer_grid_dim.x / event_dims[depth];
        new_consumer_lo_bid.x = i * factor;
        new_consumer_hi_bid.x = (i + 1) * factor;
      }
      if (depth == input_map.y) {
        int factor = consumer_grid_dim.y / event_dims[depth];
        new_consumer_lo_bid.y = i * factor;
        new_consumer_hi_bid.y = (i + 1) * factor;
      }
      if (depth == input_map.z) {
        int factor = consumer_grid_dim.z / event_dims[depth];
        new_consumer_lo_bid.z = i * factor;
        new_consumer_hi_bid.z = (i + 1) * factor;
      }
      if (depth == output_map.x) {
        int factor = producer_grid_dim.x / event_dims[depth];
        new_producer_lo_bid.x = i * factor;
        new_producer_hi_bid.x = (i + 1) * factor;
      }
      if (depth == output_map.y) {
        int factor = producer_grid_dim.y / event_dims[depth];
        new_producer_lo_bid.y = i * factor;
        new_producer_hi_bid.y = (i + 1) * factor;
      }
      if (depth == output_map.z) {
        int factor = producer_grid_dim.z / event_dims[depth];
        new_producer_lo_bid.z = i * factor;
        new_producer_hi_bid.z = (i + 1) * factor;
      }
      dfs_create_events_add_tasks(depth + 1,
                                  my_gpu_id,
                                  num_gpus,
                                  event_dims,
                                  input_map,
                                  output_map,
                                  consumer_grid_dim,
                                  producer_grid_dim,
                                  new_consumer_lo_bid,
                                  new_consumer_hi_bid,
                                  new_producer_lo_bid,
                                  new_producer_hi_bid,
                                  all_events,
                                  all_tasks,
                                  cur_op_tasks,
                                  pre_task_map,
                                  cur_task_map,
                                  nvshmem_events_idx,
                                  nvshmem_event,
                                  multigpu_task);
    }
  }
}

void register_mugraph(
    mirage::kernel::Graph const &graph,
    int num_gpus,
    int my_gpu_id,
    std::vector<FullTaskDesc> &all_tasks,
    std::vector<EventDesc> &all_events,
    std::vector<TaskId> &first_tasks,
    std::map<kernel::KNOperator *,
             std::map<dim3, std::vector<TaskId>, Dim3Comparator>>
        &all_task_maps,
    std::unordered_map<kn::KNOperator const *,
                       std::tuple<int, int, TaskType, int>> const
        &task_configs) {
  // Build the AnnotatedGraph. This replaces the old chain-only matching
  // (based on guid equality between pre_output_ops and current input_ops)
  // with a full DAG analysis: residual stripping, fork/join classification,
  // per-edge event_dim via GCD, and N-way LCM unification at fork/join
  // boundaries. Any error (cycle, disallowed role combination, ill-formed
  // LCM) aborts compilation here so we fail early with a clear message
  // instead of silently producing a wrong schedule.
  AnnotatedGraph ag = build_annotated_graph(graph, task_configs);
  if (char const *env = std::getenv("MIRAGE_DUMP_ANNOTATED_GRAPH");
      env != nullptr && std::string(env) != "0") {
    std::string dump = maybe_dump_annotated_graph(ag);
    std::fprintf(stderr, "%s", dump.c_str());
  }

  // push a begin-graph task and a event to launch dependent asks
  {
    EventDesc e(EVENT_LAUNCH_DEPENDENT_TASKS, 1, 0, 0);
    FullTaskDesc t(TASK_BEGIN_TASK_GRAPH, 0 /*variant_id*/);
    t.trigger_event = get_event_id(my_gpu_id, all_events.size(), false);
    all_tasks.push_back(t);
    all_events.push_back(e);
  }

  std::unordered_set<size_t> nvshmem_events_idx;
  int const V = (int)ag.layers.size();
  // Per-layer task map: bid -> task_id(s). Indexed by layer index instead of
  // KNOperator* so that downstream edges can look up a producer's tasks by
  // layer index (the AnnotatedGraph's natural key). We still populate
  // `all_task_maps` at the end for the legacy KNOperator*-keyed consumers
  // (print_task_graph).
  std::vector<std::map<dim3, std::vector<TaskId>, Dim3Comparator>>
      layer_task_maps(V);
  std::vector<bool> layer_is_multigpu(V, false);
  // Each fork group's consumer-bundle is emitted exactly once — when we
  // encounter branch 0 (the head) in the topo walk. Other branches of the
  // same group are skipped because the head already laid out ALL branches'
  // tasks interleaved per fork event.
  std::vector<bool> bundle_emitted(ag.fork_groups.size(), false);

  auto get_tensor_desc = [](tb::TBInputOp *const &tb_op) -> TensorDesc {
    TensorDesc desc;
    assert(tb_op->output_tensors.size() == 1);
    tb::STensor stensor = tb_op->output_tensors[0];
    kn::KNInputOp *kernel_input_op =
        static_cast<kn::KNInputOp *>(tb_op->dtensor.owner_op);
    desc.num_dims = stensor.num_dims;
    desc.data_type = stensor.data_type;
    for (int d = stensor.num_dims - 1; d >= 0; d--) {
      desc.dim[d] = stensor.dim[d];
      desc.stride[d] = kernel_input_op->input_strides[d];
    }
    return desc;
  };

  // Split a customized op's bgraph into input_ops / output_ops (same quirk
  // as before: outputs live as TBInputOps after the num_inputs mark).
  auto split_ops = [](kn::KNCustomizedOp const *op,
                      int num_inputs,
                      int num_outputs,
                      std::vector<tb::TBInputOp *> &input_ops,
                      std::vector<tb::TBInputOp *> &output_ops) {
    input_ops.clear();
    output_ops.clear();
    for (auto const &sub_op : op->bgraph.operators) {
      assert(sub_op->op_type == mirage::type::TB_INPUT_OP);
      if ((int)input_ops.size() < num_inputs) {
        input_ops.push_back(static_cast<tb::TBInputOp *>(sub_op));
      } else {
        output_ops.push_back(static_cast<tb::TBInputOp *>(sub_op));
      }
    }
    assert((int)input_ops.size() == num_inputs);
    assert((int)output_ops.size() == num_outputs);
  };

  // Build the bid-lex ordered task vector for a layer (same metadata logic as
  // the original code).
  auto build_tasks_bid_lex = [&](kn::KNCustomizedOp const *cur_op,
                                 TaskType task_type,
                                 int variant_id,
                                 int num_subtasks,
                                 std::vector<tb::TBInputOp *> const &input_ops,
                                 std::vector<tb::TBInputOp *> const &output_ops)
      -> std::vector<FullTaskDesc> {
    std::vector<FullTaskDesc> tasks;
    tb::Graph const &bgraph = cur_op->bgraph;
    dim3 bid;
    for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
      for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
        for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
          for (int subtask_id = 0; subtask_id < num_subtasks; subtask_id++) {
            FullTaskDesc task(task_type, variant_id);
            // Set request_id for attention and paged_attention
            if ((task_type == TASK_ATTENTION_1) ||
                (task_type == TASK_ATTENTION_2) ||
                (task_type == TASK_SINGLE_BATCH_EXTEND_ATTENTION) ||
                (task_type == TASK_PAGED_ATTENTION_1) ||
                (task_type == TASK_PAGED_ATTENTION_2) ||
                (task_type == TASK_PAGED_ATTENTION_HOPPER) ||
                (task_type == TASK_PAGED_ATTENTION_SPLIT_KV_SM100) ||
                (TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100) ||
                (task_type == TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER) ||
                (task_type == TASK_ATTN_SM100)) {
              // Note that we assume grid_dim.x corresponds to
              // the request dimension
              task.task_metadata.request_id = bid.x;
            }
            // Set expert_offset for MoE tasks
            if (task_type == TASK_MOE_W13_LINEAR_SM100 ||
                task_type == TASK_MOE_W2_LINEAR_SM100 ||
                task_type == TASK_MOE_W13_LINEAR_SM90 ||
                task_type == TASK_MOE_W2_LINEAR_SM90 ||
                task_type == TASK_MOE_W13_FP8_SM100 ||
                task_type == TASK_MOE_W2_FP8_SM100) {
              task.task_metadata.expert_offset = bid.x;
            }
            // Set paged attention split kv task kv_idx
            if (task_type == TASK_PAGED_ATTENTION_SPLIT_KV_SM100 ||
                task_type == TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100 ||
                task_type == TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER) {
              task.task_metadata.kv_idx = bid.z;
              task.task_metadata.merge_task_offset = bid.y;
            }
            // Set MLA decode metadata: request_id=batch (bid.y), kv_idx=split
            // (bid.x)
            if (task_type == TASK_MLA_DECODE_SM100) {
              task.task_metadata.request_id = bid.y; // batch_idx or head group
              task.task_metadata.kv_idx = bid.x;     // split_idx
              task.task_metadata.merge_task_offset = bid.z; // batch for q_len>1
            }
            // Set MLA reduce metadata: request_id=batch (bid.x)
            if (task_type == TASK_MLA_REDUCE_SM100) {
              task.task_metadata.request_id = bid.x; // batch_idx or head group
              task.task_metadata.merge_task_offset = bid.z; // batch for q_len>1
            }
            // Set MLA prefill metadata: request_id=batch (bid.z),
            // kv_idx=q_block (bid.y), merge_task_offset=head (bid.x).
            if (task_type == TASK_MLA_PREFILL_SM100) {
              task.task_metadata.request_id = bid.z;        // batch_idx
              task.task_metadata.kv_idx = bid.y;            // q_block
              task.task_metadata.merge_task_offset = bid.x; // head
            }
            // MLA prefill TP8: grid=(H, num_q_blocks, B)
            //   request_id        = head    (bid.x, fits in int16_t)
            //   kv_idx            = q_block (bid.y, fits in uint16_t)
            //   merge_task_offset = batch   (bid.z) — lives at union offset 4
            //   so it doesn't alias request_id/kv_idx (unlike expert_offset).
            if (task_type == TASK_MLA_PREFILL_TP8_SM100) {
              task.task_metadata.request_id = bid.x;
              task.task_metadata.kv_idx = bid.y;
              task.task_metadata.merge_task_offset = bid.z;
            }
            // MTP decode: grid=(sk, num_head_groups, B)
            // request_id=gi (head_group from bid.y), kv_idx=si (split from
            // bid.x) expert_offset stores hpb for TMA box dimension
            if (task_type == TASK_MLA_MTP_DECODE_SM100) {
              task.task_metadata.kv_idx = bid.x;            // si (split_idx)
              task.task_metadata.request_id = bid.y;        // gi (head_group)
              task.task_metadata.merge_task_offset = bid.z; // batch
            }
            // MTP reduce: grid=(D_V/RD_DV, num_head_groups, B)
            if (task_type == TASK_MLA_MTP_REDUCE_SM100) {
              task.task_metadata.kv_idx = bid.x;            // dv_block_idx
              task.task_metadata.request_id = bid.y;        // gi (head_group)
              task.task_metadata.merge_task_offset = bid.z; // batch
            }
            // MLA-MTP TP variants: decode grid=(num_groups*sk[*2 if TP=4], B,
            // 1) Python layer encodes block_x = gi*sk+si (or
            // (block_x<<1)|v_half for TP=4) into kv_idx, batch into request_id.
            // Kernel unpacks v_half from low bit of block_x for TP=4.
            if (task_type == TASK_MLA_MTP_DECODE_TP2_SM100 ||
                task_type == TASK_MLA_MTP_DECODE_TP4_SM100 ||
                task_type == TASK_MLA_MTP_DECODE_TP8_SM100) {
              task.task_metadata.kv_idx = bid.x;     // (gi*sk+si) or packed
              task.task_metadata.request_id = bid.y; // batch
            }
            if (task_type == TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100 ||
                task_type == TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100 ||
                task_type == TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100) {
              task.task_metadata.kv_idx = bid.x;            // dv_block_idx
              task.task_metadata.request_id = bid.y;        // gi
              task.task_metadata.merge_task_offset = bid.z; // batch
            }
            // MLA KV gather split: request_id = bid.x (builder uses
            // grid=(max_num_batched_requests, 1, 1))
            if (task_type == TASK_MLA_KV_GATHER_SPLIT_SM100) {
              task.task_metadata.request_id = bid.x;
            }
            // MLA KV gather: request_id = bid.x (builder uses
            // grid=(max_num_batched_requests, 1, 1))
            if (task_type == TASK_MLA_KV_GATHER_SM100) {
              task.task_metadata.request_id = bid.x;
            }
            // Set request_id for FP8 quantize (row index for column-major scale
            // output)
            if (task_type == TASK_QUANTIZE_FP8_SM100) {
              task.task_metadata.request_id = bid.x;
            }
            if (task_type == TASK_NVSHMEM_TILE_ALLREDUCE) {
              task.task_metadata.task_offset =
                  bid.x + bid.y * bgraph.grid_dim.x +
                  bid.z * bgraph.grid_dim.x * bgraph.grid_dim.y;
            }
            // Initialize input tensors to the task
            for (auto const &input : input_ops) {
              task.inputs[task.num_inputs++] = get_tensor_desc(input);
            }
            // Initialize output tensors to the task
            for (auto const &output : output_ops) {
              task.outputs[task.num_outputs++] = get_tensor_desc(output);
            }
            tasks.push_back(task);
          }
        }
      }
    }
    return tasks;
  };

  auto bid_offset = [](dim3 bid, dim3 grid) -> int {
    return bid.x * grid.y * grid.z + bid.y * grid.z + bid.z;
  };

  auto axis_of_tensor_dim = [](int3 const &m, int tensor_dim) -> int {
    if (m.x == tensor_dim) {
      return 0;
    }
    if (m.y == tensor_dim) {
      return 1;
    }
    if (m.z == tensor_dim) {
      return 2;
    }
    return -1;
  };

  // Given a (lo, hi) on the consumer grid represented as dim3, push each bid
  // (lex order) into all_tasks, recording in task_map. Honors multigpu
  // num_subtasks in the pre-built tasks vector.
  auto push_bids_lex =
      [&](dim3 lo,
          dim3 hi,
          dim3 grid,
          int num_subtasks,
          bool is_multigpu,
          std::vector<FullTaskDesc> const &bid_lex_tasks,
          std::map<dim3, std::vector<TaskId>, Dim3Comparator> &task_map) {
        dim3 b;
        for (b.x = lo.x; b.x < hi.x; b.x++) {
          for (b.y = lo.y; b.y < hi.y; b.y++) {
            for (b.z = lo.z; b.z < hi.z; b.z++) {
              int off = bid_offset(b, grid);
              if (is_multigpu) {
                task_map[b] = std::vector<TaskId>();
                for (int i = 0; i < num_subtasks; i++) {
                  task_map[b].push_back(all_tasks.size());
                  all_tasks.push_back(bid_lex_tasks[off * num_subtasks + i]);
                }
              } else {
                task_map[b] = std::vector<TaskId>{(TaskId)all_tasks.size()};
                all_tasks.push_back(bid_lex_tasks[off]);
              }
            }
          }
        }
      };

  // Given an event index (ex, ey, ez) in a grid-axis event frame with last3
  // block size, returns (lo, hi) on that grid.
  auto grid_subrange_for_event = [](int ex,
                                    int ey,
                                    int ez,
                                    std::array<int, 3> const &last3,
                                    dim3 grid) -> std::pair<dim3, dim3> {
    dim3 lo(ex * last3[0], ey * last3[1], ez * last3[2]);
    dim3 hi((ex + 1) * last3[0], (ey + 1) * last3[1], (ez + 1) * last3[2]);
    (void)grid;
    return {lo, hi};
  };

  // Map a fork-event index (in producer grid axis frame via lcm_last3) to
  // a consumer branch's bid sub-range.
  //
  // The translation goes: producer grid axis -> (via output_map) tensor
  // dim -> (via input_map) consumer grid axis. For each consumer grid axis
  // g_c: find which tensor dim d = input_map[g_c], then which producer
  // grid axis g_p has output_map[g_p] == d. The event index on g_p
  // equals the event index on g_c. Multiply by post-LCM consumer last3
  // to get the bid offset. Axes unmapped on either side collapse to 0
  // (a single event slot along that axis).
  auto consumer_subrange_from_fork_event =
      [&](int3 const &p_ev_idx,
          EdgeInfo const &edge,
          dim3 cons_grid) -> std::pair<dim3, dim3> {
    int c_ev[3] = {0, 0, 0};
    for (int g_c = 0; g_c < 3; g_c++) {
      int d = (g_c == 0   ? edge.input_map.x
               : g_c == 1 ? edge.input_map.y
                          : edge.input_map.z);
      if (d < 0 || d >= (int)mirage::config::MAX_TENSOR_DIMS) {
        c_ev[g_c] = 0;
        continue;
      }
      int g_p = axis_of_tensor_dim(edge.output_map, d);
      if (g_p < 0) {
        c_ev[g_c] = 0;
      } else {
        c_ev[g_c] = (g_p == 0   ? p_ev_idx.x
                     : g_p == 1 ? p_ev_idx.y
                                : p_ev_idx.z);
      }
    }
    (void)cons_grid;
    dim3 lo(c_ev[0] * edge.consumer_side_view.last3[0],
            c_ev[1] * edge.consumer_side_view.last3[1],
            c_ev[2] * edge.consumer_side_view.last3[2]);
    dim3 hi((c_ev[0] + 1) * edge.consumer_side_view.last3[0],
            (c_ev[1] + 1) * edge.consumer_side_view.last3[1],
            (c_ev[2] + 1) * edge.consumer_side_view.last3[2]);
    return {lo, hi};
  };

  // Map a join-event index (expressed in consumer grid axis frame via
  // jg.lcm_last3) to a producer's bid sub-range, via input_map/output_map
  // and the edge's post-LCM producer-side last3.
  auto producer_subrange_from_join_event =
      [&](int3 const &c_ev_idx,
          EdgeInfo const &edge,
          dim3 prod_grid) -> std::pair<dim3, dim3> {
    int p_ev[3] = {0, 0, 0};
    for (int g_p = 0; g_p < 3; g_p++) {
      int d = (g_p == 0   ? edge.output_map.x
               : g_p == 1 ? edge.output_map.y
                          : edge.output_map.z);
      if (d < 0 || d >= (int)mirage::config::MAX_TENSOR_DIMS) {
        p_ev[g_p] = 0;
        continue;
      }
      int g_c = axis_of_tensor_dim(edge.input_map, d);
      if (g_c < 0) {
        p_ev[g_p] = 0;
      } else {
        p_ev[g_p] = (g_c == 0   ? c_ev_idx.x
                     : g_c == 1 ? c_ev_idx.y
                                : c_ev_idx.z);
      }
    }
    (void)prod_grid;
    dim3 lo(p_ev[0] * edge.producer_side_view.last3[0],
            p_ev[1] * edge.producer_side_view.last3[1],
            p_ev[2] * edge.producer_side_view.last3[2]);
    dim3 hi((p_ev[0] + 1) * edge.producer_side_view.last3[0],
            (p_ev[1] + 1) * edge.producer_side_view.last3[1],
            (p_ev[2] + 1) * edge.producer_side_view.last3[2]);
    return {lo, hi};
  };

  // Walk producer bids in [lo, hi), set trigger_event, count num_triggers.
  // Handles NVSHMEM multigpu (task_ids.size() == num_gpus - 1) exactly like
  // the chain dfs does.
  auto set_producer_triggers =
      [&](dim3 lo,
          dim3 hi,
          std::map<dim3, std::vector<TaskId>, Dim3Comparator> const &prod_map,
          size_t event_pos,
          bool nvshmem_event,
          EventDesc &event_desc) {
        dim3 b;
        for (b.x = lo.x; b.x < hi.x; b.x++) {
          for (b.y = lo.y; b.y < hi.y; b.y++) {
            for (b.z = lo.z; b.z < hi.z; b.z++) {
              auto it = prod_map.find(b);
              assert(it != prod_map.end());
              std::vector<TaskId> const &task_ids = it->second;
              if (all_tasks[task_ids[0]].task_type ==
                  TASK_NVSHMEM_ALLGATHER_STRIDED_PUT) {
                assert(task_ids.size() == (size_t)num_gpus - 1);
                for (int tgt = 0; tgt < num_gpus; tgt++) {
                  if (tgt == my_gpu_id) {
                    continue;
                  }
                  size_t idx = tgt < my_gpu_id ? tgt : tgt - 1;
                  all_tasks[task_ids[idx]].trigger_event =
                      get_event_id(tgt, event_pos, nvshmem_event);
                  event_desc.num_triggers++;
                }
              } else {
                assert(task_ids.size() == 1);
                all_tasks[task_ids[0]].trigger_event =
                    get_event_id(my_gpu_id, event_pos, nvshmem_event);
                event_desc.num_triggers++;
              }
            }
          }
        }
      };

  // -------------------------------------------------------------------
  // Pass: iterate ordered_layers, emitting tasks + events per layer role.
  // -------------------------------------------------------------------
  for (int layer_idx : ag.ordered_layers) {
    LayerInfo const &L = ag.layers[layer_idx];
    kn::KNCustomizedOp const *cur_op = L.op;
    TaskType task_type = L.task_type;
    int variant_id = L.variant_id;
    int num_inputs = L.num_inputs;
    int num_outputs = L.num_outputs;
    tb::Graph const &bgraph = cur_op->bgraph;
    dim3 cur_grid = bgraph.grid_dim;

    std::vector<tb::TBInputOp *> input_ops, output_ops;
    split_ops(cur_op, num_inputs, num_outputs, input_ops, output_ops);

    int cur_num_subtasks = get_num_subtasks(num_gpus, task_type);
    bool cur_is_multigpu = (task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT);
    layer_is_multigpu[layer_idx] = cur_is_multigpu;

    // Skip non-head branches of a fork-consumer bundle. The head (branch
    // index 0) emits tasks for ALL branches interleaved per fork event so
    // each fork event's consumer-range is contiguous in all_tasks. If we
    // let each branch run here, their tasks would land in separate blocks
    // and no single EventDesc could span them.
    if (L.fork_parent_group >= 0 && L.fork_branch_index != 0) {
      continue;
    }

    // ---- First layer (no in-edges after residual stripping): lex emit.
    if (L.in_edges.empty() && L.fork_parent_group < 0) {
      std::vector<FullTaskDesc> tasks = build_tasks_bid_lex(cur_op,
                                                            task_type,
                                                            variant_id,
                                                            cur_num_subtasks,
                                                            input_ops,
                                                            output_ops);
      dim3 b;
      for (b.x = 0; b.x < cur_grid.x; b.x++) {
        for (b.y = 0; b.y < cur_grid.y; b.y++) {
          for (b.z = 0; b.z < cur_grid.z; b.z++) {
            int off = bid_offset(b, cur_grid);
            if (cur_is_multigpu) {
              layer_task_maps[layer_idx][b] = std::vector<TaskId>();
              for (int i = 0; i < cur_num_subtasks; i++) {
                layer_task_maps[layer_idx][b].push_back(all_tasks.size());
                first_tasks.push_back(all_tasks.size());
                all_tasks.push_back(tasks[off * cur_num_subtasks + i]);
              }
            } else {
              layer_task_maps[layer_idx][b] =
                  std::vector<TaskId>{(TaskId)all_tasks.size()};
              first_tasks.push_back(all_tasks.size());
              all_tasks.push_back(tasks[off]);
            }
          }
        }
      }
      all_task_maps.emplace(const_cast<kn::KNOperator *>(
                                static_cast<kn::KNOperator const *>(cur_op)),
                            layer_task_maps[layer_idx]);
      continue;
    }

    // ---- Fork consumer bundle (head branch only reaches here).
    if (L.fork_parent_group >= 0) {
      int fg_id = L.fork_parent_group;
      if (bundle_emitted[fg_id]) {
        continue;
      }
      ForkGroupInfo const &fg = ag.fork_groups[fg_id];
      LayerInfo const &P = ag.layers[fg.producer_layer];
      tb::Graph const &pgraph = P.op->bgraph;
      // Build bid-lex tasks for every branch in advance.
      std::vector<std::vector<FullTaskDesc>> branch_tasks;
      std::vector<std::vector<tb::TBInputOp *>> branch_inputs, branch_outputs;
      std::vector<int> branch_num_subtasks;
      std::vector<bool> branch_is_multigpu;
      branch_tasks.reserve(fg.outgoing_edges.size());
      branch_inputs.reserve(fg.outgoing_edges.size());
      branch_outputs.reserve(fg.outgoing_edges.size());
      for (int eidx : fg.outgoing_edges) {
        EdgeInfo const &e = ag.edges[eidx];
        LayerInfo const &B = ag.layers[e.cons_layer];
        std::vector<tb::TBInputOp *> b_in, b_out;
        split_ops(B.op, B.num_inputs, B.num_outputs, b_in, b_out);
        int b_ns = get_num_subtasks(num_gpus, B.task_type);
        bool b_mg = (B.task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT);
        branch_num_subtasks.push_back(b_ns);
        branch_is_multigpu.push_back(b_mg);
        branch_inputs.push_back(b_in);
        branch_outputs.push_back(b_out);
        branch_tasks.push_back(build_tasks_bid_lex(
            B.op, B.task_type, B.variant_id, b_ns, b_in, b_out));
      }
      // Fork event grid (in producer grid axis frame) is
      // (p_grid / lcm_last3) per axis. The loop below walks events in
      // (ex, ey, ez) order; this order is the one that defines task_ids
      // for consumer tasks, so downstream event emission iterating in the
      // same order yields contiguous consumer ranges by construction.
      int ex_max = pgraph.grid_dim.x / std::max(fg.lcm_last3[0], 1);
      int ey_max = pgraph.grid_dim.y / std::max(fg.lcm_last3[1], 1);
      int ez_max = pgraph.grid_dim.z / std::max(fg.lcm_last3[2], 1);
      for (int ex = 0; ex < ex_max; ex++) {
        for (int ey = 0; ey < ey_max; ey++) {
          for (int ez = 0; ez < ez_max; ez++) {
            int3 p_ev{ex, ey, ez};
            EventDesc event_desc;
            event_desc.num_triggers = 0;
            // Mark the start of this fork event's consumer range. All
            // tasks pushed between here and last_task_id below will be
            // triggered by this one EventDesc — hence the contiguity
            // requirement that forces interleaving of branch tasks.
            event_desc.first_task_id = all_tasks.size();
            // Emit branch consumer tasks interleaved: for this single fork
            // event, push branch 0's per-event sub-block, then branch 1's,
            // ..., then branch N-1's. This keeps all tasks triggered by
            // this event contiguous in all_tasks.
            for (size_t b = 0; b < fg.outgoing_edges.size(); b++) {
              EdgeInfo const &e = ag.edges[fg.outgoing_edges[b]];
              LayerInfo const &B = ag.layers[e.cons_layer];
              auto [c_lo, c_hi] = consumer_subrange_from_fork_event(
                  p_ev, e, B.op->bgraph.grid_dim);
              push_bids_lex(c_lo,
                            c_hi,
                            B.op->bgraph.grid_dim,
                            branch_num_subtasks[b],
                            branch_is_multigpu[b],
                            branch_tasks[b],
                            layer_task_maps[e.cons_layer]);
            }
            event_desc.last_task_id = all_tasks.size();
            // Set trigger_event on producer tasks in the producer sub-range.
            dim3 p_lo(ex * fg.lcm_last3[0],
                      ey * fg.lcm_last3[1],
                      ez * fg.lcm_last3[2]);
            dim3 p_hi((ex + 1) * fg.lcm_last3[0],
                      (ey + 1) * fg.lcm_last3[1],
                      (ez + 1) * fg.lcm_last3[2]);
            bool nvshmem_event_flag = layer_is_multigpu[fg.producer_layer];
            set_producer_triggers(p_lo,
                                  p_hi,
                                  layer_task_maps[fg.producer_layer],
                                  all_events.size(),
                                  nvshmem_event_flag,
                                  event_desc);
            if (nvshmem_event_flag) {
              nvshmem_events_idx.insert(all_events.size());
            }
            event_desc.event_type =
                event_desc.last_task_id >= event_desc.first_task_id + 8
                    ? EVENT_LAUNCH_MASSIVE_TASKS
                    : EVENT_LAUNCH_TASKS;
            all_events.push_back(event_desc);
          }
        }
      }
      // Register each branch's task_map in all_task_maps.
      for (int eidx : fg.outgoing_edges) {
        EdgeInfo const &e = ag.edges[eidx];
        all_task_maps.emplace(
            const_cast<kn::KNOperator *>(static_cast<kn::KNOperator const *>(
                ag.layers[e.cons_layer].op)),
            layer_task_maps[e.cons_layer]);
      }
      bundle_emitted[fg_id] = true;
      continue;
    }

    // ---- Join consumer: emit the single consumer layer's tasks in
    // join-event-major order (one contiguous sub-block per join event),
    // then create ONE EventDesc per join event with num_triggers accumulated
    // from ALL incoming edges (each incoming producer contributes a slice
    // of bids that fire this same event). Consumer tasks are in one layer,
    // so their ranges are naturally contiguous without any interleaving.
    if (L.is_join_consumer) {
      int jg_id = ag.edges[L.in_edges[0]].join_group_id;
      assert(jg_id >= 0);
      JoinGroupInfo const &jg = ag.join_groups[jg_id];
      std::vector<FullTaskDesc> tasks = build_tasks_bid_lex(cur_op,
                                                            task_type,
                                                            variant_id,
                                                            cur_num_subtasks,
                                                            input_ops,
                                                            output_ops);
      int ex_max = cur_grid.x / std::max(jg.lcm_last3[0], 1);
      int ey_max = cur_grid.y / std::max(jg.lcm_last3[1], 1);
      int ez_max = cur_grid.z / std::max(jg.lcm_last3[2], 1);
      for (int ex = 0; ex < ex_max; ex++) {
        for (int ey = 0; ey < ey_max; ey++) {
          for (int ez = 0; ez < ez_max; ez++) {
            int3 c_ev{ex, ey, ez};
            EventDesc event_desc;
            event_desc.num_triggers = 0;
            event_desc.first_task_id = all_tasks.size();
            auto [c_lo, c_hi] =
                grid_subrange_for_event(ex, ey, ez, jg.lcm_last3, cur_grid);
            push_bids_lex(c_lo,
                          c_hi,
                          cur_grid,
                          cur_num_subtasks,
                          cur_is_multigpu,
                          tasks,
                          layer_task_maps[layer_idx]);
            event_desc.last_task_id = all_tasks.size();
            bool any_nvshmem = false;
            for (int eidx : jg.incoming_edges) {
              EdgeInfo const &e = ag.edges[eidx];
              bool nvshmem_event_flag = layer_is_multigpu[e.prod_layer];
              any_nvshmem = any_nvshmem || nvshmem_event_flag;
              auto [p_lo, p_hi] = producer_subrange_from_join_event(
                  c_ev, e, ag.layers[e.prod_layer].op->bgraph.grid_dim);
              set_producer_triggers(p_lo,
                                    p_hi,
                                    layer_task_maps[e.prod_layer],
                                    all_events.size(),
                                    nvshmem_event_flag,
                                    event_desc);
            }
            if (any_nvshmem) {
              nvshmem_events_idx.insert(all_events.size());
            }
            event_desc.event_type =
                event_desc.last_task_id >= event_desc.first_task_id + 8
                    ? EVENT_LAUNCH_MASSIVE_TASKS
                    : EVENT_LAUNCH_TASKS;
            all_events.push_back(event_desc);
          }
        }
      }
      all_task_maps.emplace(const_cast<kn::KNOperator *>(
                                static_cast<kn::KNOperator const *>(cur_op)),
                            layer_task_maps[layer_idx]);
      continue;
    }

    // ---- Chain layer (single distinct producer, not fork-consumer-bundle,
    // not join). This path exercises the exact same dfs_create_events_add_tasks
    // the old code used, just with edge info sourced from AnnotatedGraph
    // rather than rediscovered via guid matching. For graphs that are
    // already chains (qwen3 today), this produces bit-identical output to
    // the previous implementation.
    //
    // There may be multiple in-edges to this layer from the SAME producer
    // (multi-tensor bridge — e.g. a producer with 2 output tensors both
    // feeding this consumer). We use the FIRST non-stripped in-edge as the
    // event driver; the other edges share the same (prod, cons) pair and
    // hence the same event grid, so a single event covers them implicitly.
    assert(!L.in_edges.empty());
    EdgeInfo const &edge = ag.edges[L.in_edges[0]];
    LayerInfo const &P = ag.layers[edge.prod_layer];
    std::vector<FullTaskDesc> tasks = build_tasks_bid_lex(
        cur_op, task_type, variant_id, cur_num_subtasks, input_ops, output_ops);
    std::vector<int> event_dims_v(mirage::config::MAX_TENSOR_DIMS, 1);
    for (int d = 0; d < (int)mirage::config::MAX_TENSOR_DIMS; d++) {
      event_dims_v[d] = edge.event_dim[d];
    }
    bool cur_task_trigger_nvshmem_event = layer_is_multigpu[edge.prod_layer];
    dfs_create_events_add_tasks(0,
                                my_gpu_id,
                                num_gpus,
                                event_dims_v,
                                edge.input_map,
                                edge.output_map,
                                cur_grid,
                                P.op->bgraph.grid_dim,
                                dim3(0, 0, 0),
                                cur_grid,
                                dim3(0, 0, 0),
                                P.op->bgraph.grid_dim,
                                all_events,
                                all_tasks,
                                tasks,
                                layer_task_maps[edge.prod_layer],
                                layer_task_maps[layer_idx],
                                nvshmem_events_idx,
                                cur_task_trigger_nvshmem_event,
                                cur_is_multigpu);
    all_task_maps.emplace(const_cast<kn::KNOperator *>(
                              static_cast<kn::KNOperator const *>(cur_op)),
                          layer_task_maps[layer_idx]);
  }

  // Update the trigger event for all tasks in every LEAF layer's task map
  // (drives EVENT_END_OF_TASK_GRAPH).
  //
  // For chain graphs this is the single final layer (what the old code
  // handled). For DAGs with multiple terminal branches (e.g. an unjoined
  // fork where each branch ends at its own output), EVERY leaf layer's
  // tasks must contribute their completion to the END event; otherwise
  // the non-last leaves' tasks end up with trigger_event = INVALID and
  // the runtime can't correctly count graph completion. This was a bug
  // caught by the fork unit test.
  size_t end_num_triggers = 0;
  for (int i = 0; i < V; i++) {
    if (!ag.layers[i].out_edges.empty()) {
      continue;
    }
    for (auto const &it : layer_task_maps[i]) {
      assert(it.second.size() == 1);
      all_tasks[it.second[0]].trigger_event =
          get_event_id(my_gpu_id, all_events.size(), false /*nvshmem_event*/);
      end_num_triggers++;
    }
  }
  all_events.push_back(
      EventDesc(EVENT_END_OF_TASK_GRAPH, end_num_triggers, 0, 0));

  // Prelaunch all tasks at the begining of an iteration
  all_events[1].first_task_id = 2;
  all_events[1].last_task_id = all_tasks.size();
  for (size_t e = 2; e < all_events.size(); e++) {
    if (all_events[e].event_type == EVENT_LAUNCH_TASKS ||
        all_events[e].event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
      all_events[e].event_type = EVENT_EMPTY;
      bool is_nvshmem_event = false;
      if (nvshmem_events_idx.count(e) > 0) {
        is_nvshmem_event = true;
      }
      for (size_t t = all_events[e].first_task_id;
           t < all_events[e].last_task_id;
           t++) {
        all_tasks[t].dependent_event =
            get_event_id(my_gpu_id, e, is_nvshmem_event /*nvshmem_event*/);
      }
    }
  }
}

bool sanity_check(mirage::kernel::Graph const &graph,
                  std::vector<FullTaskDesc> const &all_tasks,
                  std::vector<EventDesc> const &all_events,
                  std::vector<TaskId> const &first_tasks) {
  std::unordered_set<EventId> triggered_events;
  std::unordered_set<TaskId> executed_tasks;
  std::vector<int> event_counts(all_events.size(), 0);
  for (size_t i = 0; i < all_events.size(); i++) {
    event_counts[i] = all_events[i].num_triggers;
  }
  std::queue<TaskId> task_queue;
  std::queue<EventId> event_queue;
  printf("First tasks: %d\n", (int)first_tasks.size());
  for (size_t i = 0; i < first_tasks.size(); i++) {
    task_queue.push(first_tasks[i]);
  }
  while (!(task_queue.empty() && event_queue.empty())) {
    // Execute tasks
    while (!task_queue.empty()) {
      TaskId task = task_queue.front();
      task_queue.pop();
      assert(executed_tasks.count(task) == 0);
      executed_tasks.insert(task);
      FullTaskDesc desc = all_tasks[task];
      if (desc.trigger_event != EVENT_INVALID_ID) {
        EventId event_id = desc.trigger_event;
        size_t event_pos = event_id & 0xffffffff;
        // event_pos 0 is the end of task graph event
        if (event_pos == 0) {
          continue;
        }
        assert(event_counts[event_pos] > 0);
        event_counts[event_pos]--;
        if (event_counts[event_pos] == 0) {
          event_queue.push(event_id);
        }
      }
    }
    while (!event_queue.empty()) {
      EventId event_id = event_queue.front();
      event_queue.pop();
      assert(triggered_events.count(event_id) == 0);
      triggered_events.insert(event_id);
      size_t event_pos = event_id & 0xffffffff;
      EventDesc desc = all_events[event_pos];
      for (TaskId tid = desc.first_task_id; tid < desc.last_task_id; tid++) {
        task_queue.push(tid);
      }
    }
  }
  printf("Number of all events: %zu\n", all_events.size());
  printf("Number of all tasks: %zu\n", all_tasks.size());
  printf("Number of triggered events: %zu\n", triggered_events.size());
  printf("Number of executed tasks: %zu\n", executed_tasks.size());
  return true;
}

TaskGraphResult print_task_graph(
    mirage::kernel::Graph const &graph,
    int num_gpus,
    int my_gpu_id,
    std::vector<FullTaskDesc> const &all_tasks,
    std::vector<EventDesc> const &all_events,
    std::vector<TaskId> const &first_tasks,
    std::map<kernel::KNOperator *,
             std::map<dim3, std::vector<TaskId>, Dim3Comparator>> const
        &all_task_maps,
    std::unordered_map<kn::KNOperator const *,
                       std::tuple<int, int, TaskType, int>> const &task_configs,
    std::map<mirage::type::GuidType, IODesc> const &io_configs,
    bool use_json_format) {
  using mirage::runtime::IODesc;
  mirage::transpiler::CodeKeeper code;
  mirage::transpiler::CodeKeeper tgbody;
  tgbody.inc_indent();
  code.e("#include \"persistent_kernel.cuh\"");
  if (use_json_format) {
    code.e("#include <nlohmann/json.hpp>");
    code.e("#include <fstream>");
    code.e("#include <filesystem>");
    code.e("using json = nlohmann::json;");
  }
  code.e("using namespace mirage::runtime;");
  code.e("size_t get_event_id(int my_gpu_id, size_t event_pos, bool "
         "nvshmem_event) {");
  code.e("size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | "
         "event_pos);");
  code.e("if (nvshmem_event) {");
  code.e("event_id = event_id | EVENT_NVSHMEM_TAG;");
  code.e("}");
  code.e("return event_id;");
  code.e("}");
  code.e("");

  // function that loads json file and generates task graph
  if (use_json_format) {
    code.e("void construct_task_graph(int num_gpus,");
    code.e("                          int my_gpu_id,");
    code.e("                          std::vector<FullTaskDesc> &all_tasks,");
    code.e("                          std::vector<EventDesc> &all_events,");
    code.e("                          std::vector<TaskId> &first_tasks,");
    code.e("                          std::map<std::string, void*> const "
           "&all_tensors) {");
    code.e("std::filesystem::path file_path(__FILE__);");
    code.e("std::ifstream "
           "json_file(file_path.parent_path().string()+\"/task_graph.json\");");
    code.e("nlohmann::json json_task_graph;");
    code.e("json_file >> json_task_graph;");
    // load tasks
    code.e("for (json const &task : json_task_graph[\"all_tasks\"]) {");
    code.e("FullTaskDesc "
           "task_desc(static_cast<TaskType>(task.at(\"task_type\")),");
    code.e("            task.at(\"variant_id\"));");
    code.e("task_desc.task_metadata.request_id = "
           "task.at(\"request_id\").get<int>();");
    code.e("task_desc.task_metadata.expert_offset = "
           "task.at(\"expert_offset\").get<int>();");
    code.e("task_desc.task_metadata.kv_idx = task.at(\"kv_idx\").get<int>();");
    code.e("task_desc.task_metadata.merge_task_offset = "
           "task.at(\"merge_task_offset\").get<int>();");
    code.e("task_desc.task_metadata.task_offset = "
           "task.at(\"task_offset\").get<int>();");
    code.e("if (task.at(\"trigger_event\").is_number_integer()) {");
    code.e("task_desc.trigger_event = task.at(\"trigger_event\").get<unsigned "
           "long long int>();");
    code.e("}");
    code.e("else {");
    code.e("assert(false);");
    code.e("}");
    code.e("if (task.at(\"dependent_event\").is_number_integer()) {");
    code.e("task_desc.dependent_event = "
           "task.at(\"dependent_event\").get<unsigned long long int>();");
    code.e("}");
    code.e("else {");
    code.e("assert(false);");
    code.e("}");

    // load inputs
    code.e("task_desc.num_inputs = 0;");
    code.e("for (json const &tensor : task[\"inputs\"]) {");
    code.e("TensorDesc input;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e("input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e(
        "assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
    code.e("input.num_dims = tensor.at(\"dims\").size();");
    code.e("input.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < input.num_dims; i++) {");
    code.e("input.dim[i] = tensor[\"dims\"][i].get<int>();");
    code.e("input.stride[i] = tensor[\"strides\"][i].get<int>();");
    code.e("}");

    code.e("task_desc.inputs[task_desc.num_inputs++] = input;");
    code.e("}");
    // load outputs
    code.e("task_desc.num_outputs = 0;");
    code.e("for (json const &tensor : task[\"outputs\"]) {");
    code.e("TensorDesc output;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e(
        "output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e(
        "assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
    code.e("output.num_dims = tensor.at(\"dims\").size();");
    code.e("output.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < output.num_dims; i++) {");
    code.e("output.dim[i] = tensor[\"dims\"][i];");
    code.e("output.stride[i] = tensor[\"strides\"][i];");
    code.e("}");

    code.e("task_desc.outputs[task_desc.num_outputs++] = output;");
    code.e("}");

    // create TMA desc for each task
    code.e("#ifdef MPK_ENABLE_TMA");
    // Hopper Tasks
    code.e("if (task.at(\"task_type\") > TASK_HOPPER_TASK_BEGIN && "
           "task.at(\"task_type\") < TASK_HOPPER_TASK_END) {");
    code.e("create_tma_desc_by_task(task_desc);");
    code.e("}");
    // SM100 Tasks
    code.e("if (task.at(\"task_type\") > TASK_SM100_TMA_START_TASK && "
           "task.at(\"task_type\") < TASK_SM100_TMA_END_TASK) {");
    code.e("create_tma_desc_by_task(task_desc);");
    code.e("}");
    // MLA kernels (outside SM100_TMA range but need TMA)
    code.e("if (task.at(\"task_type\") == TASK_MLA_DECODE_SM100 || "
           "task.at(\"task_type\") == TASK_MLA_MTP_DECODE_SM100 || "
           "task.at(\"task_type\") == TASK_MLA_MTP_DECODE_TP2_SM100 || "
           "task.at(\"task_type\") == TASK_MLA_MTP_DECODE_TP4_SM100 || "
           "task.at(\"task_type\") == TASK_MLA_MTP_DECODE_TP8_SM100 || "
           "task.at(\"task_type\") == TASK_MLA_PREFILL_TP8_SM100) {");
    code.e("create_tma_desc_by_task(task_desc);");
    code.e("}");
    // FP8 linear tasks need TMA (outside SM100_TMA range)
    code.e("if (task.at(\"task_type\") == TASK_LINEAR_FP8_SM100 || "
           "task.at(\"task_type\") == TASK_LINEAR_FP8_WITH_RESIDUAL_SM100) {");
    code.e("create_tma_desc_by_task(task_desc);");
    code.e("}");
    code.e("#endif");
    code.e("all_tasks.push_back(task_desc);");
    code.e("}");
    // load events
    code.e("for (json const &e : json_task_graph[\"all_events\"]) {");
    code.e("EventType event_type = "
           "static_cast<EventType>(e.at(\"event_type\").get<int>());");
    code.e("int num_triggers = e.at(\"num_triggers\").get<int>();");
    code.e("int first_task_id = e.at(\"first_task_id\").get<int>();");
    code.e("int last_task_id = e.at(\"last_task_id\").get<int>();");
    code.e("all_events.push_back(EventDesc(event_type, num_triggers, "
           "first_task_id, last_task_id));");
    code.e("}");
    // load first tasks
    code.e("for (json const &t : json_task_graph[\"first_tasks\"]) {");
    code.e("first_tasks.push_back(t.get<int>());");
    code.e("}");
    code.e("}");
    code.e("");
  }

  code.e("static void _init_persistent_kernel(std::vector<FullTaskDesc> "
         "&all_tasks,");
  code.e("                                    std::vector<EventDesc> "
         "&all_events,");
  code.e("                                  std::vector<TaskId> &first_tasks,");
  code.e("                                  int num_gpus,");
  code.e("                                  int my_gpu_id) {");
  code.e("assert(num_gpus = $);", num_gpus);

  if (use_json_format) {
    code.e("std::map<std::string, void*> all_tensors;");
  }
  for (auto const &iter : io_configs) {
    IODesc desc = iter.second;
    switch (desc.type) {
      case IODesc::TorchTensor: {
        code.e("char *$ = (char*)($);", desc.name, desc.torch_data_ptr);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      case IODesc::FusedTorchTensor: {
        for (auto const &sdesc : desc.sub_descs) {
          code.e("char *$ = (char*)($);", sdesc.name, sdesc.torch_data_ptr);
          if (use_json_format) {
            code.e("all_tensors[\"$\"] = $;", sdesc.name, sdesc.name);
          }
        }
        break;
      }
      case IODesc::CUDAMallocTensor: {
        code.e("void *$;", desc.name);
        size_t size = mirage::type::get_datatype_size(
            static_cast<type::DataType>(desc.tensor.data_type));
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("CUDA_CHECK(cudaMalloc(&$, $));", desc.name, size);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      case IODesc::NVSHMEMMallocTensor: {
        size_t size = mirage::type::get_datatype_size(
            static_cast<type::DataType>(desc.tensor.data_type));
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("void *$ = nvshmem_malloc($);", desc.name, size);
        code.e("assert($ != nullptr);", desc.name);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      case IODesc::ShuffledTorchTensor: {
        code.e("char *$;", desc.name);
        size_t size = mirage::type::get_datatype_size(
            static_cast<type::DataType>(desc.tensor.data_type));
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("CUDA_CHECK(cudaMalloc(&$, $));", desc.name, size);

        size_t bytes_per_row = size / desc.tensor.dim[0];
        size_t bytes_per_group = 0;
        std::vector<size_t> bytes_per_tensor;
        for (int i = 0; i < desc.sub_descs.size(); i++) {
          bytes_per_group +=
              bytes_per_row * desc.sub_descs[i].tensor.dim[0] / desc.num_groups;
          bytes_per_tensor.push_back(bytes_per_row *
                                     desc.sub_descs[i].tensor.dim[0] /
                                     desc.num_groups);
        }
        size_t start_addr_offset = 0;
        for (int i = 0; i < desc.sub_descs.size(); i++) {
          code.e("CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<void *>($ + "
                 "$), $, "
                 "reinterpret_cast<const void *>($), $, $, $, "
                 "cudaMemcpyDeviceToDevice));",
                 desc.name,         /*dst address*/
                 start_addr_offset, /*dst bytes offset between each copy*/
                 bytes_per_group,   /*dst bytes offset between each copy*/
                 desc.sub_descs[i].torch_data_ptr, /*src address*/
                 bytes_per_tensor[i], /*src bytes offset between each copy*/
                 bytes_per_tensor[i], /*width*/
                 desc.num_groups      /*height*/
          );
          start_addr_offset += bytes_per_tensor[i];
        }
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      default:
        assert(false);
    }
  }
  json json_task_graph = {
      {"all_tasks", {}}, {"all_events", {}}, {"first_tasks", {}}};
  // generate task[0]
  {
    tgbody.e("all_tasks.push_back(FullTaskDesc(TASK_TERMINATE));");
    json_task_graph["all_tasks"].push_back(
        json{{"task_type", TASK_TERMINATE},
             {"variant_id", 0},
             {"inputs", {}},
             {"outputs", {}},
             {"trigger_event", EVENT_INVALID_ID},
             {"dependent_event", EVENT_INVALID_ID},
             {"request_id", -1},
             {"expert_offset", -1},
             {"kv_idx", -1},
             {"merge_task_offset", -1},
             {"task_offset", -1}});
  }
  // generate task[1]
  {
    tgbody.e("all_tasks.push_back(FullTaskDesc(TASK_BEGIN_TASK_GRAPH));");
    json_task_graph["all_tasks"].push_back(
        json{{"task_type", TASK_BEGIN_TASK_GRAPH},
             {"variant_id", 0},
             {"inputs", {}},
             {"outputs", {}},
             {"trigger_event",
              get_event_id(my_gpu_id, 1 /*event_pos*/, false /*is_nvshmem*/)},
             {"dependent_event", EVENT_INVALID_ID},
             {"request_id", -1},
             {"expert_offset", -1},
             {"kv_idx", -1},
             {"merge_task_offset", -1},
             {"task_offset", -1}});
  }
  // generate all other tasks
  // Global buffer of (task_id -> json_task).
  //
  // The old code wrote `json_tasks[task_id - starting_task_id] = ...` with
  // json_tasks sized to a single op's grid volume, assuming each op's
  // tasks were a contiguous block in all_tasks. With DAG-level fork
  // emission, a fork-consumer layer's tasks are INTERLEAVED with its
  // siblings (e.g. layer B's tasks live at indices 18, 20, 22, ...; layer
  // C's at 19, 21, 23, ...). Writing into a per-op vector with task_id -
  // starting_task_id offsets then overflows (layer B's offset 30 into a
  // vector sized 16), causing heap corruption — the double-free we saw
  // the first time the fork test ran.
  //
  // Fix: buffer every task's json by global task_id; after visiting every
  // op, flush in strict task_id order so that the JSON's "all_tasks"
  // array index matches the runtime's all_tasks vector index.
  std::vector<json> global_json_tasks(all_tasks.size());
  std::vector<char> global_json_filled(all_tasks.size(), 0);

  size_t task_pos = 2;
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      continue;
    }
    assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
    std::tuple<int, int, TaskType, int> task_config =
        task_configs.find(op)->second;

    assert(all_task_maps.find(op) != all_task_maps.end());
    std::map<dim3, std::vector<TaskId>, Dim3Comparator> const &task_map =
        all_task_maps.find(op)->second;
    // Customized op
    kn::KNCustomizedOp const *cur_op =
        dynamic_cast<kn::KNCustomizedOp const *>(op);
    tb::Graph const &bgraph = cur_op->bgraph;
    dim3 bid;
    std::vector<tb::TBInputOp *> input_ops;
    std::vector<tb::TBInputOp *> output_ops;
    int num_inputs = std::get<0>(task_config);
    // int num_outputs = std::get<1>(task_config);
    TaskType task_type = std::get<2>(task_config);
    for (auto const &op : bgraph.operators) {
      assert(op->op_type == mirage::type::TB_INPUT_OP);
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBInputOp *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBInputOp *>(op));
      }
    }

    unsigned cur_op_num_subtasks = get_num_subtasks(num_gpus, task_type);

    // Tasks may be non-contiguous in all_tasks (DAG fork interleaving), so we
    // write directly into the global task-id-indexed buffer.
    size_t op_total_subtasks = (size_t)bgraph.grid_dim.x * bgraph.grid_dim.y *
                               bgraph.grid_dim.z * cur_op_num_subtasks;
    (void)op_total_subtasks;
    for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
      for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
        for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
          for (int subtask_id = 0; subtask_id < cur_op_num_subtasks;
               subtask_id++) {
            TaskId task_id = task_map.at(bid)[subtask_id];
            FullTaskDesc task_desc = all_tasks[task_id];
            tgbody.e("// task[$]", task_id);
            tgbody.e("{");
            tgbody.e("FullTaskDesc task_desc(static_cast<TaskType>($));",
                     task_desc.task_type);
            size_t event_gpu_id = ((task_desc.trigger_event >> 32) & 0xffff);
            size_t event_pos = (task_desc.trigger_event & 0xffffffff);
            bool is_nvshmem_event =
                ((task_desc.trigger_event & EVENT_NVSHMEM_TAG) > 0);
            // assert(event_gpu_id == my_gpu_id);
            // assert(!is_nvshmem_event);
            json json_task;
            json_task = {
                {"task_type", task_desc.task_type},
                {"variant_id", task_desc.variant_id},
                {"inputs", {}},
                {"outputs", {}},
                {"trigger_event", task_desc.trigger_event},
                {"dependent_event", task_desc.dependent_event},
                {"request_id", task_desc.task_metadata.request_id},
                {"expert_offset", task_desc.task_metadata.expert_offset},
                {"kv_idx", task_desc.task_metadata.kv_idx},
                {"merge_task_offset",
                 task_desc.task_metadata.merge_task_offset},
                {"task_offset", task_desc.task_metadata.task_offset}};

            for (int i = 0; i < task_desc.num_inputs; i++) {
              if (input_ops[i]->dtensor == kernel::DTensor::EMPTY_TENSOR) {
                json json_dims = json::array();
                json json_strides = json::array();
                json_task["inputs"].push_back(
                    json{{"base_ptr", "nullptr"},
                         {"offset", 0},
                         {"data_type", type::DT_UNKNOWN},
                         {"dims", json_dims},
                         {"strides", json_strides}});
                continue;
              }
              off_t offset = 0;
              int num_dims = input_ops[i]->dtensor.num_dims;
              int3 input_map = input_ops[i]->input_map;
              IODesc io_desc =
                  io_configs.find(input_ops[i]->dtensor.guid)->second;
              assert(input_ops[i]->dtensor.owner_op->op_type ==
                     type::KN_INPUT_OP);
              if (io_desc.type == IODesc::FusedTorchTensor) {
                // Currently assert that we fuse the 0-th dim (i.e., 0)
                int fused_group_size = 0;
                std::vector<int> group_sizes;
                for (auto const &sub_desc : io_desc.sub_descs) {
                  assert(sub_desc.tensor.num_dims == num_dims);
                  assert(sub_desc.tensor.dim[0] % io_desc.num_groups == 0);
                  int my_group_size =
                      sub_desc.tensor.dim[0] / io_desc.num_groups;
                  fused_group_size += my_group_size;
                  group_sizes.push_back(my_group_size);
                }
                assert(io_desc.tensor.dim[0] ==
                       fused_group_size * io_desc.num_groups);
                assert(io_desc.tensor.num_dims == num_dims);
                int fused_dim_off = 0;
                if (input_map.x == 0) {
                  fused_dim_off =
                      io_desc.tensor.dim[0] / bgraph.grid_dim.x * bid.x;
                }
                if (input_map.y == 0) {
                  fused_dim_off =
                      io_desc.tensor.dim[0] / bgraph.grid_dim.y * bid.y;
                }
                if (input_map.z == 0) {
                  fused_dim_off =
                      io_desc.tensor.dim[0] / bgraph.grid_dim.z * bid.z;
                }
                int fused_dim_off_in_group = fused_dim_off % fused_group_size;
                size_t index = 0;
                while (index < group_sizes.size()) {
                  if (fused_dim_off_in_group >= group_sizes[index]) {
                    fused_dim_off_in_group -= group_sizes[index];
                    index++;
                  } else {
                    break;
                  }
                }
                IODesc sub_desc = io_desc.sub_descs[index];
                int fused_dim_off_subtensor =
                    fused_dim_off / fused_group_size * group_sizes[index] +
                    fused_dim_off_in_group;
                // Assert that it is within range
                assert(fused_dim_off_subtensor < sub_desc.tensor.dim[0]);
                if (input_map.x > 0) {
                  size_t block_size =
                      sub_desc.tensor.dim[input_map.x] / bgraph.grid_dim.x;
                  offset +=
                      block_size * bid.x * sub_desc.tensor.stride[input_map.x];
                } else if (input_map.x == 0) {
                  offset += fused_dim_off_subtensor *
                            sub_desc.tensor.stride[input_map.x];
                }
                if (input_map.y > 0) {
                  size_t block_size =
                      sub_desc.tensor.dim[input_map.y] / bgraph.grid_dim.y;
                  offset +=
                      block_size * bid.y * sub_desc.tensor.stride[input_map.y];
                } else if (input_map.y == 0) {
                  offset += fused_dim_off_subtensor *
                            sub_desc.tensor.stride[input_map.y];
                }
                if (input_map.z > 0) {
                  size_t block_size =
                      sub_desc.tensor.dim[input_map.z] / bgraph.grid_dim.z;
                  offset +=
                      block_size * bid.z * sub_desc.tensor.stride[input_map.z];
                } else if (input_map.z == 0) {
                  offset += fused_dim_off_subtensor *
                            sub_desc.tensor.stride[input_map.z];
                }
                tgbody.e("TensorDesc input$;", i);
                tgbody.e("input$.base_ptr = static_cast<char*>($) + $;",
                         i,
                         sub_desc.name,
                         offset * type::get_datatype_size(
                                      static_cast<type::DataType>(
                                          sub_desc.tensor.data_type)));
                tgbody.e(
                    "input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
                tgbody.e(
                    "input$.data_type = $;", i, task_desc.inputs[i].data_type);
                json json_dims = json::array();
                json json_strides = json::array();
                for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
                  tgbody.e(
                      "input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
                  tgbody.e(
                      "input$.stride[$] = $;", i, d, sub_desc.tensor.stride[d]);
                  json_dims.push_back(task_desc.inputs[i].dim[d]);
                  json_strides.push_back(sub_desc.tensor.stride[d]);
                }
                tgbody.e("task_desc.inputs[$] = input$;", i, i);
                json_task["inputs"].push_back(json{
                    {"base_ptr", sub_desc.name},
                    {"offset",
                     offset *
                         type::get_datatype_size(static_cast<type::DataType>(
                             sub_desc.tensor.data_type))},
                    {"data_type", task_desc.inputs[i].data_type},
                    {"dims", json_dims},
                    {"strides", json_strides}});
              } else {
                // Non-fused case, use io_desc
                if (input_map.x >= 0) {
                  size_t block_size =
                      io_desc.tensor.dim[input_map.x] / bgraph.grid_dim.x;
                  offset +=
                      block_size * bid.x * io_desc.tensor.stride[input_map.x];
                }
                if (input_map.y >= 0) {
                  size_t block_size =
                      io_desc.tensor.dim[input_map.y] / bgraph.grid_dim.y;
                  offset +=
                      block_size * bid.y * io_desc.tensor.stride[input_map.y];
                }
                if (input_map.z >= 0) {
                  size_t block_size =
                      io_desc.tensor.dim[input_map.z] / bgraph.grid_dim.z;
                  offset +=
                      block_size * bid.z * io_desc.tensor.stride[input_map.z];
                }
                tgbody.e("TensorDesc input$;", i);
                tgbody.e("input$.base_ptr = static_cast<char*>($) + $;",
                         i,
                         io_desc.name,
                         offset * type::get_datatype_size(
                                      static_cast<type::DataType>(
                                          io_desc.tensor.data_type)));
                tgbody.e(
                    "input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
                tgbody.e(
                    "input$.data_type = $;", i, task_desc.inputs[i].data_type);
                json json_dims = json::array();
                json json_strides = json::array();
                for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
                  tgbody.e(
                      "input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
                  tgbody.e("input$.stride[$] = $;",
                           i,
                           d,
                           task_desc.inputs[i].stride[d]);
                  json_dims.push_back(task_desc.inputs[i].dim[d]);
                  json_strides.push_back(task_desc.inputs[i].stride[d]);
                }
                tgbody.e("task_desc.inputs[$] = input$;", i, i);
                json_task["inputs"].push_back(json{
                    {"base_ptr", io_desc.name},
                    {"offset",
                     offset *
                         type::get_datatype_size(static_cast<type::DataType>(
                             io_desc.tensor.data_type))},
                    {"data_type", task_desc.inputs[i].data_type},
                    {"dims", json_dims},
                    {"strides", json_strides}});
              }
            }

            for (int i = 0; i < task_desc.num_outputs; i++) {
              off_t offset = 0;
              if (task_type == runtime::TASK_NVSHMEM_ALLGATHER_STRIDED_PUT) {
                // A special case for buffer-style tensors.
                offset = my_gpu_id * input_ops[0]->dtensor.num_elements();
              }
              int3 output_map = output_ops[i]->input_map;
              IODesc io_desc =
                  io_configs.find(output_ops[i]->dtensor.guid)->second;
              assert(io_desc.type != IODesc::FusedTorchTensor);
              if (output_map.x >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.x] / bgraph.grid_dim.x;
                offset +=
                    block_size * bid.x * io_desc.tensor.stride[output_map.x];
              }
              if (output_map.y >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.y] / bgraph.grid_dim.y;
                offset +=
                    block_size * bid.y * io_desc.tensor.stride[output_map.y];
              }
              if (output_map.z >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.z] / bgraph.grid_dim.z;
                offset +=
                    block_size * bid.z * io_desc.tensor.stride[output_map.z];
              }

              tgbody.e("TensorDesc output$;", i);
              tgbody.e("output$.base_ptr = static_cast<char*>($) + $;",
                       i,
                       io_desc.name,
                       offset *
                           type::get_datatype_size(static_cast<type::DataType>(
                               io_desc.tensor.data_type)));
              tgbody.e(
                  "output$.num_dims = $;", i, task_desc.outputs[i].num_dims);
              tgbody.e(
                  "output$.data_type = $;", i, task_desc.outputs[i].data_type);
              json json_dims = json::array();
              json json_strides = json::array();
              for (int d = 0; d < task_desc.outputs[i].num_dims; d++) {
                tgbody.e(
                    "output$.dim[$] = $;", i, d, task_desc.outputs[i].dim[d]);
                tgbody.e("output$.stride[$] = $;",
                         i,
                         d,
                         task_desc.outputs[i].stride[d]);
                json_dims.push_back(task_desc.outputs[i].dim[d]);
                json_strides.push_back(task_desc.outputs[i].stride[d]);
              }
              tgbody.e("task_desc.outputs[$] = output$;", i, i);
              json_task["outputs"].push_back(json{
                  {"base_ptr", io_desc.name},
                  {"offset",
                   offset * type::get_datatype_size(static_cast<type::DataType>(
                                io_desc.tensor.data_type))},
                  {"data_type", task_desc.outputs[i].data_type},
                  {"dims", json_dims},
                  {"strides", json_strides}});
            }

            tgbody.e("all_tasks.push_back(task_desc);");
            tgbody.e("}");
            assert((size_t)task_id < global_json_tasks.size());
            assert(!global_json_filled[task_id]);
            global_json_tasks[task_id] = json_task;
            global_json_filled[task_id] = 1;
          } // subtask_id
        }   // bid.z
      }     // bid.y
    }       // bid.x

    task_pos += op_total_subtasks;
  }
  assert(task_pos == all_tasks.size());

  // Flush tasks 2..N-1 in task_id order. Tasks 0 (TASK_TERMINATE) and 1
  // (TASK_BEGIN_TASK_GRAPH) are prelude and handled by the generated
  // construct_task_graph loader (not in JSON all_tasks).
  for (size_t task_id = 2; task_id < all_tasks.size(); task_id++) {
    assert(global_json_filled[task_id]);
    json_task_graph["all_tasks"].push_back(global_json_tasks[task_id]);
  }
  // Add all events
  for (auto const &event : all_events) {
    tgbody.e(
        "all_events.push_back(EventDesc(static_cast<EventType>($), $, $, $));",
        event.event_type,
        event.num_triggers,
        event.first_task_id,
        event.last_task_id);
    json_task_graph["all_events"].push_back(
        json{{"event_type", event.event_type},
             {"num_triggers", event.num_triggers},
             {"first_task_id", event.first_task_id},
             {"last_task_id", event.last_task_id}});
  }
  // Add first task
  for (auto const &task : first_tasks) {
    tgbody.e("first_tasks.push_back($);", task);
    json_task_graph["first_tasks"].push_back(task);
  }
  if (use_json_format) {
    // Add nullptr for tensors set as None
    code.e("all_tensors[\"nullptr\"] = nullptr;");
    code.e("construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, "
           "first_tasks, all_tensors);");
  } else {
    code.e(tgbody.to_string());
  }
  // ensure cudaMemcpyAsync is completed
  code.e("cudaDeviceSynchronize();");
  code.e("}");
  code.e("");

  // Generate task implementation
  std::map<TaskType, std::string> task_type_to_name;
  task_type_to_name[TASK_EMBEDDING] = "TASK_EMBEDDING";
  task_type_to_name[TASK_RMS_NORM] = "TASK_RMS_NORM";
  task_type_to_name[TASK_RMS_NORM_LINEAR] = "TASK_RMS_NORM_LINEAR";
  task_type_to_name[TASK_ATTENTION_1] = "TASK_ATTENTION_1";
  task_type_to_name[TASK_SILU_MUL] = "TASK_SILU_MUL";
  task_type_to_name[TASK_IDENTITY] = "TASK_IDENTITY";
  task_type_to_name[TASK_SILU_MUL_LINEAR_WITH_RESIDUAL] =
      "TASK_SILU_MUL_LINEAR_WITH_RESIDUAL";
  task_type_to_name[TASK_LINEAR] = "TASK_LINEAR";
  task_type_to_name[TASK_LINEAR_WITH_RESIDUAL] = "TASK_LINEAR_WITH_RESIDUAL";
  task_type_to_name[TASK_ARGMAX_PARTIAL] = "TASK_ARGMAX_PARTIAL";
  task_type_to_name[TASK_ARGMAX_REDUCE] = "TASK_ARGMAX_REDUCE";
  task_type_to_name[TASK_REDUCE] = "TASK_REDUCE";
  task_type_to_name[TASK_FIND_NGRAM_PARTIAL] = "TASK_FIND_NGRAM_PARTIAL";
  task_type_to_name[TASK_FIND_NGRAM_GLOBAL] = "TASK_FIND_NGRAM_GLOBAL";
  task_type_to_name[TASK_TARGET_VERIFY_GREEDY] = "TASK_TARGET_VERIFY_GREEDY";
  task_type_to_name[TASK_SINGLE_BATCH_EXTEND_ATTENTION] =
      "TASK_SINGLE_BATCH_EXTEND_ATTENTION";
  task_type_to_name[TASK_PAGED_ATTENTION_1] = "TASK_PAGED_ATTENTION_1";
  task_type_to_name[TASK_LINEAR_HOPPER] = "TASK_LINEAR_HOPPER";
  task_type_to_name[TASK_LINEAR_WITH_RESIDUAL_HOPPER] =
      "TASK_LINEAR_WITH_RESIDUAL_HOPPER";
  task_type_to_name[TASK_PAGED_ATTENTION_HOPPER] =
      "TASK_PAGED_ATTENTION_HOPPER";
  task_type_to_name[TASK_RMS_NORM_HOPPER] = "TASK_RMS_NORM_HOPPER";
  task_type_to_name[TASK_LINEAR_SWAPAB_HOPPER] = "TASK_LINEAR_SWAPAB_HOPPER";
  task_type_to_name[TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER] =
      "TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER";
  task_type_to_name[TASK_LINEAR_CUTLASS_HOPPER] = "TASK_LINEAR_CUTLASS_HOPPER";
  task_type_to_name[TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER] =
      "TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER";
  task_type_to_name[TASK_SILU_MUL_HOPPER] = "TASK_SILU_MUL_HOPPER";
  task_type_to_name[TASK_EMBEDDING_HOPPER] = "TASK_EMBEDDING_HOPPER";
  task_type_to_name[TASK_LINEAR_SM100] = "TASK_LINEAR_SM100";
  task_type_to_name[TASK_LINEAR_WITH_RESIDUAL_SM100] =
      "TASK_LINEAR_WITH_RESIDUAL_SM100";
  task_type_to_name[TASK_SPLITK_LINEAR_SM100] = "TASK_SPLITK_LINEAR_SM100";
  task_type_to_name[TASK_ATTN_SM100] = "TASK_ATTN_SM100";
  task_type_to_name[TASK_ARGMAX_PARTIAL_SM100] = "TASK_ARGMAX_PARTIAL_SM100";
  task_type_to_name[TASK_ARGMAX_REDUCE_SM100] = "TASK_ARGMAX_REDUCE_SM100";
  task_type_to_name[TASK_SAMPLING_SM100] = "TASK_SAMPLING_SM100";
  task_type_to_name[TASK_MLA_DECODE_SM100] = "TASK_MLA_DECODE_SM100";
  task_type_to_name[TASK_MLA_REDUCE_SM100] = "TASK_MLA_REDUCE_SM100";
  task_type_to_name[TASK_MLA_PREFILL_SM100] = "TASK_MLA_PREFILL_SM100";
  task_type_to_name[TASK_MLA_PREFILL_TP8_SM100] = "TASK_MLA_PREFILL_TP8_SM100";
  task_type_to_name[TASK_MLA_MTP_DECODE_SM100] = "TASK_MLA_MTP_DECODE_SM100";
  task_type_to_name[TASK_MLA_MTP_REDUCE_SM100] = "TASK_MLA_MTP_REDUCE_SM100";
  task_type_to_name[TASK_MLA_MTP_DECODE_TP2_SM100] =
      "TASK_MLA_MTP_DECODE_TP2_SM100";
  task_type_to_name[TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100] =
      "TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100";
  task_type_to_name[TASK_MLA_MTP_DECODE_TP4_SM100] =
      "TASK_MLA_MTP_DECODE_TP4_SM100";
  task_type_to_name[TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100] =
      "TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100";
  task_type_to_name[TASK_MLA_MTP_DECODE_TP8_SM100] =
      "TASK_MLA_MTP_DECODE_TP8_SM100";
  task_type_to_name[TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100] =
      "TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100";
  task_type_to_name[TASK_MLA_KV_GATHER_SM100] = "TASK_MLA_KV_GATHER_SM100";
  task_type_to_name[TASK_MLA_KV_GATHER_SPLIT_SM100] =
      "TASK_MLA_KV_GATHER_SPLIT_SM100";
  task_type_to_name[TASK_MTP_VERIFY_STRICT] = "TASK_MTP_VERIFY_STRICT";
  task_type_to_name[TASK_MTP_ACCEPT_COMMIT] = "TASK_MTP_ACCEPT_COMMIT";
  task_type_to_name[TASK_MTP_TOKEN_SCATTER] = "TASK_MTP_TOKEN_SCATTER";
  task_type_to_name[TASK_MTP_PREPARE_VERIFY] = "TASK_MTP_PREPARE_VERIFY";
  task_type_to_name[TASK_MTP_BUILD_EMBED_INPUT] = "TASK_MTP_BUILD_EMBED_INPUT";
  task_type_to_name[TASK_QUANTIZE_FP8_SM100] = "TASK_QUANTIZE_FP8_SM100";
  task_type_to_name[TASK_LINEAR_FP8_SM100] = "TASK_LINEAR_FP8_SM100";
  task_type_to_name[TASK_LINEAR_FP8_WITH_RESIDUAL_SM100] =
      "TASK_LINEAR_FP8_WITH_RESIDUAL_SM100";
  task_type_to_name[TASK_TENSOR_INIT] = "TASK_TENSOR_INIT";
  task_type_to_name[TASK_MOE_TOPK_SOFTMAX_SM100] =
      "TASK_MOE_TOPK_SOFTMAX_SM100";
  task_type_to_name[TASK_MOE_TOPK_SIGMOID_SM100] =
      "TASK_MOE_TOPK_SIGMOID_SM100";
  task_type_to_name[TASK_MOE_W13_LINEAR_SM100] = "TASK_MOE_W13_LINEAR_SM100";
  task_type_to_name[TASK_MOE_W2_LINEAR_SM100] = "TASK_MOE_W2_LINEAR_SM100";
  task_type_to_name[TASK_MOE_W13_FP8_SM100] = "TASK_MOE_W13_FP8_SM100";
  task_type_to_name[TASK_MOE_W2_FP8_SM100] = "TASK_MOE_W2_FP8_SM100";
  task_type_to_name[TASK_MOE_MUL_SUM_ADD_SM100] = "TASK_MOE_MUL_SUM_ADD_SM100";
  task_type_to_name[TASK_ELEMENTWISE_ADD_SM100] = "TASK_ELEMENTWISE_ADD_SM100";
  task_type_to_name[TASK_SOFTMAX_GATHER_SM100] = "TASK_SOFTMAX_GATHER_SM100";
  task_type_to_name[TASK_MTP_VERIFY_PROBABILISTIC] =
      "TASK_MTP_VERIFY_PROBABILISTIC";
  task_type_to_name[TASK_PROB_SCATTER_SM100] = "TASK_PROB_SCATTER_SM100";
  task_type_to_name[TASK_MTP_FLOAT_SCATTER] = "TASK_MTP_FLOAT_SCATTER";
  task_type_to_name[TASK_PROB_EXTRACT_SM100] = "TASK_PROB_EXTRACT_SM100";
  task_type_to_name[TASK_MOE_W13_LINEAR_SM90] = "TASK_MOE_W13_LINEAR_SM90";
  task_type_to_name[TASK_MOE_W2_LINEAR_SM90] = "TASK_MOE_W2_LINEAR_SM90";
  task_type_to_name[TASK_SPLITK_LINEAR_SWAPAB_HOPPER] =
      "TASK_SPLITK_LINEAR_SWAPAB_HOPPER";
  task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_SM100] =
      "TASK_PAGED_ATTENTION_SPLIT_KV_SM100";
  task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100] =
      "TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100";
  task_type_to_name[TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER] =
      "TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER";
  // Multi-gpu tasks
  task_type_to_name[TASK_NVSHMEM_ALLGATHER_STRIDED_PUT] =
      "TASK_NVSHMEM_ALLGATHER_STRIDED_PUT";
  task_type_to_name[TASK_NVSHMEM_TILE_ALLREDUCE] =
      "TASK_NVSHMEM_TILE_ALLREDUCE";

  code.e("__device__ __forceinline__");
  code.e("void _execute_task(TaskDesc const* task_desc,");
  code.e("                   RuntimeConfig const &runtime_config) {");
  TaskRegister *task_register = TaskRegister::get_instance();
  bool first_task = true;
  for (auto const &task : task_register->all_task_variants) {
    for (size_t variant_id = 0; variant_id < task.second.size(); variant_id++) {
      std::string cond = first_task ? "if" : "else if";
      assert(task_type_to_name.find(task.first) != task_type_to_name.end());
      code.e("$ (task_desc->task_type == $ && task_desc->variant_id == $) {",
             cond,
             task_type_to_name[task.first],
             variant_id);
      code.e("$", task.second[variant_id]);
      code.e("}");
      first_task = false;
    }
  }
  code.e("}");

  // Write json to output file
  // std::ofstream out("task_graph.json");
  // out << json_task_graph.dump(2);
  // out.close();
  TaskGraphResult result;
  result.cuda_code = code.to_string();
  result.json_file = json_task_graph.dump(2);
  return result;
}

TaskGraphResult Graph::generate_task_graph(int _num_gpus, int _my_gpu_id) {
  std::vector<FullTaskDesc> all_tasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  int num_gpus, my_gpu_id;
  std::map<kernel::KNOperator *,
           std::map<dim3, std::vector<TaskId>, Dim3Comparator>>
      all_task_maps;
  num_gpus = _num_gpus;
  my_gpu_id = _my_gpu_id;
  // add the termination event to the event lists
  EventDesc e(EVENT_TERMINATION, 1, 0, 0);
  all_events.push_back(e);
  FullTaskDesc t(TASK_TERMINATE, 0 /*variant_id*/);
  all_tasks.push_back(t);
  register_mugraph(*this,
                   num_gpus,
                   my_gpu_id,
                   all_tasks,
                   all_events,
                   first_tasks,
                   all_task_maps,
                   task_config);
  assert(sanity_check(*this, all_tasks, all_events, first_tasks));
  return print_task_graph(*this,
                          num_gpus,
                          my_gpu_id,
                          all_tasks,
                          all_events,
                          first_tasks,
                          all_task_maps,
                          task_config,
                          io_config,
                          true /*use_json_format*/);
}

} // namespace kernel

namespace runtime {

IODesc::IODesc(IOType _type,
               std::string _name,
               mirage::kernel::DTensor const &_tensor,
               void *_torch_data_ptr)
    : type(_type), name(_name), torch_data_ptr(_torch_data_ptr) {
  tensor.num_dims = _tensor.num_dims;
  tensor.data_type = _tensor.data_type;
  assert(_tensor.owner_op->op_type == mirage::type::KN_INPUT_OP);
  mirage::kernel::KNInputOp const *op =
      static_cast<mirage::kernel::KNInputOp const *>(_tensor.owner_op);
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = _tensor.dim[i];
    tensor.stride[i] = op->input_strides[i];
  }
}

} // namespace runtime
} // namespace mirage
