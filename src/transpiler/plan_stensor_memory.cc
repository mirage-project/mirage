#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/sched_tb_graph.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpiler.h"
#include "mirage/type.h"

#include <limits>
#include <set>
#include <unordered_set>

namespace mirage {
namespace transpiler {

namespace memory_planner {

// Forced alignment for every STensor
static constexpr size_t ALIGNMENT = 128;

// The declaration of a tensor, including its guid, physical size, and lifecycle
// All sizes & addresses inside the `memory_planner` namespace are in bytes!
//
// The semantic of alloc_time and free_time is as follows:
// Let's partition the time axis into intervals, and each interval is a time
// unit. A tensor with (alloc_time, free_time) = (l, r) means that, the tensor
// is allocated at the beginning of the l-th time unit, and freed at the
// beginning of the r-th time unit, which means that within the time interval
// [l, r), the tensor is allocated and can be used.
struct TensorDecl {
  sguid_t sguid;
  size_t phy_size;
  int alloc_time;
  int free_time;
};

// The allocation result
struct AllocResult {
  size_t peak_memory_usage;
  std::unordered_map<sguid_t, size_t> addrs;
};

// Some helper structs & functions
using Range = std::pair<size_t, size_t>;

// The base class for all memory planners
// See the document (docs/transpiler/transpiler.md) for details on the modeling
// of the memory planning problem
class AbstractMemoryPlanner {
public:
  // Declare a new tensor with a specified lifecycle
  virtual void declare_tensor(TensorDecl const &tensor_decl) = 0;

  // Get the memory allocation plan
  virtual AllocResult get_allocation() = 0;
};

// The base class for all memory planner that use an online allocation algorithm
// Online allocation algorithms decide the address of a tensor instancely after
// the tensor is created. These algorithms are widely used in `malloc`.
// Examples include first-fit, next-fit, best-fit, last-fit, and so on.
class OnlineAllocMemoryPlannerBase : public AbstractMemoryPlanner {
protected:
  enum class event_t { ALLOC, FREE };
  struct Event {
    event_t type;
    sguid_t stensor_guid;
    int time;
    size_t phy_size;
  };
  vector<Event> events;

  size_t cur_peak_usage = 0;
  std::unordered_map<sguid_t, size_t> addrs;

  std::set<Range> free_ranges;
  // The range selector for the online allocation algorithm
  // to be implemented by derived classes
  virtual Range select_range(size_t size) = 0;

  // Free a range, and coalesce it with adjacent free ranges
  // The range being freed should not overlap with any allocated range
  void free_and_coalesce(Range range) {
    auto it = free_ranges.lower_bound(range);
    if (it != free_ranges.begin()) {
      --it;
      if (it->second >= range.first) {
        range.first = it->first;
        free_ranges.erase(it);
      }
    }
    it = free_ranges.lower_bound(range);
    if (it != free_ranges.end() && it->first <= range.second) {
      range.second = it->second;
      it = free_ranges.erase(it);
    }
    free_ranges.insert(range);
  }
  // Rise `cur_peak_usage` (like the `sbrk` syscall in Linux). Coalesce with
  // the last free block when necessary
  void rise_peak_usage(size_t new_peak_usage) {
    assert(new_peak_usage > cur_peak_usage);
    Range new_range = {cur_peak_usage, new_peak_usage};
    auto last = free_ranges.rbegin();
    if (!free_ranges.empty() && last->second == cur_peak_usage) {
      // Coalesce with the last free block
      new_range.first = last->first;
      free_ranges.erase(std::prev(free_ranges.end()));
    }
    free_ranges.insert(new_range);
    cur_peak_usage = new_peak_usage;
  }
  // Mark a range as allocated, and split it if necessary
  // The range being allocated should be a free range
  void alloc_and_split(Range range) {
    auto it = free_ranges.lower_bound(range);
    while (it == free_ranges.end() || it->first > range.first) {
      assert(it != free_ranges.begin());
      it = std::prev(it);
    }
    assert(it->first <= range.first && range.second <= it->second);
    Range orig_range = *it;
    free_ranges.erase(it);
    if (orig_range.first < range.first) {
      free_ranges.insert({orig_range.first, range.first});
    }
    if (range.second < orig_range.second) {
      free_ranges.insert({range.second, orig_range.second});
    }
  }

public:
  void declare_tensor(TensorDecl const &t) override {
    assert(t.phy_size != 0 &&
           "Trying to allocate memory for a STensor with a size of 0");
    events.push_back({event_t::ALLOC, t.sguid, t.alloc_time, t.phy_size});
    events.push_back({event_t::FREE, t.sguid, t.free_time, t.phy_size});
  }

  AllocResult get_allocation() override {
    std::sort(events.begin(), events.end(), [](Event const &a, Event const &b) {
      // Compare by time first, and then by type (FREE first)
      if (a.time != b.time) {
        return a.time < b.time;
      } else {
        return (int)(a.type == event_t::FREE) > (int)(b.type == event_t::FREE);
      }
    });
    for (Event const &event : events) {
      if (event.type == event_t::ALLOC) {
        // Check whether there is enough free space for the new tensor. If there
        // is not, allocate more memory
        bool is_avail_space = std::any_of(
            free_ranges.begin(), free_ranges.end(), [&](Range const &range) {
              return range.second - range.first >= event.phy_size;
            });
        if (!is_avail_space) {
          auto last = free_ranges.rbegin();
          size_t last_chunk_size =
              !free_ranges.empty() && last->second == cur_peak_usage
                  ? last->second - last->first
                  : (size_t)0;
          size_t new_peak_usage =
              cur_peak_usage + (event.phy_size - last_chunk_size);
          rise_peak_usage(new_peak_usage);
        }
        Range selected_range = select_range(event.phy_size);
        assert(addrs.count(event.stensor_guid) == 0 &&
               "The STensor has been allocated before");
        assert(selected_range.second - selected_range.first == event.phy_size &&
               "Logic error: the size of the selected range is not equal to "
               "the size of the tensor");
        addrs[event.stensor_guid] = selected_range.first;
        alloc_and_split(selected_range);
      } else {
        size_t start_addr = addrs.at(event.stensor_guid);
        Range range = {start_addr, start_addr + event.phy_size};
        free_and_coalesce(range);
      }
    }
    return {cur_peak_usage, addrs};
  }
};

class FirstFitMemoryPlanner : public OnlineAllocMemoryPlannerBase {
protected:
  Range select_range(size_t size) override {
    auto it = std::find_if(
        free_ranges.begin(), free_ranges.end(), [&](Range const &range) {
          return range.second - range.first >= size;
        });
    assert(it != free_ranges.end() &&
           "No enough free space for the new tensor");
    return {it->first, it->first + size};
  }
};

class BestFitMemoryPlanner : public OnlineAllocMemoryPlannerBase {
protected:
  Range select_range(size_t size) override {
    size_t best_fit_size = std::numeric_limits<size_t>::max();
    Range best_fit_range = {0, 0};
    for (Range const &range : free_ranges) {
      size_t fit_size = range.second - range.first;
      if (fit_size >= size && fit_size < best_fit_size) {
        best_fit_size = fit_size;
        best_fit_range = range;
      }
    }
    assert(best_fit_size != std::numeric_limits<size_t>::max() &&
           "No enough free space for the new tensor");
    return {best_fit_range.first, best_fit_range.first + size};
  }
};

class WorseFitMemoryPlanner : public OnlineAllocMemoryPlannerBase {
protected:
  Range select_range(size_t size) override {
    size_t worse_fit_size = 0;
    Range worse_fit_range = {0, 0};
    for (Range const &range : free_ranges) {
      size_t fit_size = range.second - range.first;
      if (fit_size >= size && fit_size > worse_fit_size) {
        worse_fit_size = fit_size;
        worse_fit_range = range;
      }
    }
    assert(worse_fit_size != 0 && "No enough free space for the new tensor");
    return {worse_fit_range.first, worse_fit_range.first + size};
  }
};

// Run every memory planner on the given tensor declarations and return the
// one with the smallest peak memory usage
AllocResult plan_memory(vector<TensorDecl> const &tensor_decls) {
  AllocResult final_result;
  final_result.peak_memory_usage =
      std::numeric_limits<decltype(final_result.peak_memory_usage)>::max();
  vector<std::shared_ptr<AbstractMemoryPlanner>> planners = {
      std::make_shared<FirstFitMemoryPlanner>(),
      std::make_shared<BestFitMemoryPlanner>(),
      std::make_shared<WorseFitMemoryPlanner>()};
  for (std::shared_ptr<AbstractMemoryPlanner> &planner : planners) {
    for (TensorDecl const &tensor_decl : tensor_decls) {
      planner->declare_tensor(tensor_decl);
    }
    AllocResult result = planner->get_allocation();
    if (result.peak_memory_usage < final_result.peak_memory_usage) {
      final_result = result;
    }
  }
  return final_result;
}

} // namespace memory_planner

TBMemoryPlan Transpiler::get_threadblock_memory_plan(tb::Graph const &tb_graph,
                                                     TBSched const &tb_sched,
                                                     bool hopper_arch) {
  using memory_planner::Range, memory_planner::TensorDecl,
      memory_planner::AllocResult, memory_planner::ALIGNMENT;
  static constexpr sguid_t PIPELINED_INPUT_BUF_GUID_OFFSET =
      10000000; // Should be larger than the number of STensors

  // Generate all tensor declarations
  // The i-th pre_loop_nodes is executed during time slice #i
  // The i-th loop_nodes is executed during time slice #(i+T)
  // The i-th post_loop_nodes is executed during time slice #(i+2T)
  static constexpr int T = 100000000;
  vector<TensorDecl> tensor_decls;

  auto get_phy_size = [&](tb::STensor const &stensor) {
    STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
    return stensor_meta.num_phy_elems *
           type::get_datatype_size(stensor.data_type);
  };
  // auto find_first_used_time = [](sguid_t sguid,
  //                                vector<TBSchedNode> const &nodes,
  //                                int time_delta) -> int {
  //   int first_used_time = -1;
  //   for (size_t i = 0; i < nodes.size(); ++i) {
  //     TBSchedNode const &node = nodes[i];
  //     if (node.type != tb_sched_node_t::OPERATOR) {
  //       continue;
  //     }
  //     for (tb::STensor const &input_tensor :
  //          node.ops.front().first->input_tensors) {
  //       if (input_tensor.guid == sguid) {
  //         first_used_time = i + time_delta;
  //         break;
  //       }
  //     }
  //   }
  //   return first_used_time;
  // };
  auto find_last_used_time = [](sguid_t sguid,
                                vector<TBSchedNode> const &nodes,
                                int time_delta) -> int {
    int last_used_time = -1;
    for (size_t i = 0; i < nodes.size(); ++i) {
      TBSchedNode const &node = nodes[i];
      if (node.type != tb_sched_node_t::OPERATOR) {
        continue;
      }
      for (tb::STensor const &input_tensor :
           node.ops.front().first->input_tensors) {
        if (input_tensor.guid == sguid) {
          last_used_time = i + time_delta;
        }
      }
    }
    return last_used_time;
  };
  auto find_earlist_free_time =
      [&find_last_used_time](sguid_t sguid,
                             vector<TBSchedNode> const &nodes,
                             int time_delta) -> int {
    int last_used_time = find_last_used_time(sguid, nodes, time_delta);
    if (last_used_time == -1) {
      return -1;
    }
    // Locate the next "syncthread" after `last_used_time`
    // See docs/transpiler/transpiler.md for the reason of this step
    last_used_time -= time_delta;
    while (last_used_time < (int)nodes.size() &&
           nodes[last_used_time].type != tb_sched_node_t::SYNCTHREADS) {
      ++last_used_time;
    }
    return last_used_time + time_delta;
  };

  // Tensors that is an output of a pre-loop input operator
  for (int i = 0; i < (int)tb_sched.pre_loop_nodes.size(); ++i) {
    TBSchedNode const &node = tb_sched.pre_loop_nodes[i];
    // Currently pre_loop_nodes only contain non-fused input operators
    assert(node.type == tb_sched_node_t::OPERATOR);
    assert(node.ops.size() == 1);
    auto [_op, op_meta] = node.ops[0];
    assert(_op->op_type == type::TB_INPUT_OP);
    tb::TBInputOp const *op = dynamic_cast<tb::TBInputOp const *>(_op);
    tb::STensor const &stensor = op->output_tensors.at(0);
    size_t phy_size = get_phy_size(stensor);
    // Check whether the input is used after the loop
    int earlist_free_time =
        find_earlist_free_time(stensor.guid, tb_sched.post_loop_nodes, 2 * T);
    if (earlist_free_time == -1) {
      // Not used after loop. Lifecycle ends at 2T
      tensor_decls.push_back({stensor.guid, phy_size, i, 2 * T});
    } else {
      // Used after loop. Lifecycle ends at last_used_time_after_loop+1
      tensor_decls.push_back({stensor.guid, phy_size, i, earlist_free_time});
    }
  }

  // Accumulator tensors, including register-backed ones and non-reg-backed ones
  for (int i = 0; i < (int)tb_sched.loop_nodes.size(); ++i) {
    TBSchedNode const &node = tb_sched.loop_nodes[i];
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP ||
        last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
        last_op->op_type == type::TB_FORLOOP_ACCUM_MAX_OP) {
      tb::STensor const &accum = last_op->output_tensors.at(0);
      size_t phy_size = get_phy_size(accum);
      int earlist_free_time =
          find_earlist_free_time(accum.guid, tb_sched.post_loop_nodes, 2 * T);
      if (last_op_meta.is_accum_in_reg) {
        // find_first_used_time(accum.guid, tb_sched.post_loop_nodes, 2 * T);
        // assert(first_used_time != -1 &&
        //        "An accumulator is not used after the for loop");

        // buffer X number of pipe_stage
        tensor_decls.push_back(
            {accum.guid, phy_size, 2 * T, earlist_free_time});
      } else {
        tensor_decls.push_back({accum.guid,
                                phy_size,
                                T - 1,
                                earlist_free_time}); // Use T-1 here since we
                                                     // should clear the accum
      }
    }
  }

  // Intermediate tensors for for-loop ops
  for (int i = 0; i < (int)tb_sched.loop_nodes.size(); ++i) {
    TBSchedNode const &node = tb_sched.loop_nodes[i];
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type != type::TB_FORLOOP_ACCUM_NO_RED_OP &&
        last_op->op_type != type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP &&
        last_op->op_type != type::TB_FORLOOP_ACCUM_MAX_OP) {
      for (tb::STensor const &output_tensor : last_op->output_tensors) {
        size_t phy_size = get_phy_size(output_tensor);
        int earlist_free_time =
            find_earlist_free_time(output_tensor.guid, tb_sched.loop_nodes, T);
        assert(earlist_free_time != -1 &&
               "An intermediate tensor produced in the for loop is never used");
        // in hopper the doubule buffer needs to be continously allocated
        if (!(last_op->op_type == type::TB_INPUT_OP &&
              last_op_meta.is_pipelined_input)) {
          // the pipelined input tensor should occupy all ranges in the forloop
          tensor_decls.push_back(
              {output_tensor.guid, phy_size, i + T, earlist_free_time});
        }
      }
    }
  }

  // Intermediate tensors for post-loop ops
  for (int i = 0; i < (int)tb_sched.post_loop_nodes.size(); ++i) {
    TBSchedNode const &node = tb_sched.post_loop_nodes[i];
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    for (tb::STensor const &output_tensor : last_op->output_tensors) {
      size_t phy_size = get_phy_size(output_tensor);
      int earlist_free_time = find_earlist_free_time(
          output_tensor.guid, tb_sched.post_loop_nodes, 2 * T);
      assert(
          earlist_free_time != -1 &&
          "An intermediate tensor produced after the for loop is never used");
      tensor_decls.push_back(
          {output_tensor.guid, phy_size, i + 2 * T, earlist_free_time});
    }
  }

  // Buffers for software-pipelined inputs
  for (int i = 0; i < (int)tb_sched.loop_nodes.size(); ++i) {
    TBSchedNode const &node = tb_sched.loop_nodes[i];
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [op, op_meta] = node.ops.front();
    if (op->op_type == type::TB_INPUT_OP && op_meta.is_pipelined_input) {
      tb::STensor const &stensor = op->output_tensors.at(0);
      size_t phy_size = get_phy_size(stensor);
      if (hopper_arch) {
        if (stensor_metas[stensor.guid].m_input && stensor.dim[0] <= 64) {
          tensor_decls.push_back(
              {stensor.guid,
               phy_size * config.pipeline_stages * (64 / stensor.dim[0]),
               T - 1,
               2 * T});
        } else {
          tensor_decls.push_back(
              {stensor.guid, phy_size * config.pipeline_stages, T - 1, 2 * T});
        }
      } else {
        // double buffer
        tensor_decls.push_back({stensor.guid, phy_size, T - 1, 2 * T});
        tensor_decls.push_back({stensor.guid + PIPELINED_INPUT_BUF_GUID_OFFSET,
                                phy_size,
                                T - 1,
                                2 * T});
      }
    }
  }

  // Run the memory planner
  AllocResult alloc_result = memory_planner::plan_memory(tensor_decls);

  // Sanity check
  {
    // The number of tensors should be the same
    int num_stensors = 0;
    for (TBSchedNode const &node :
         Combine(Combine(tb_sched.pre_loop_nodes, tb_sched.loop_nodes),
                 tb_sched.post_loop_nodes)) {
      if (node.type == tb_sched_node_t::OPERATOR) {
        num_stensors += (int)node.ops.back().first->output_tensors.size();
        if (node.ops.front().second.is_pipelined_input && (!hopper_arch)) {
          num_stensors += 1;
        }
      }
    }
    assert(num_stensors == (int)alloc_result.addrs.size());
  }

  // Generate the memory plan
  TBMemoryPlan plan;
  plan.smem_size = alloc_result.peak_memory_usage;
  plan.addrs = alloc_result.addrs;
  plan.pipelined_input_buf_guid_offset = PIPELINED_INPUT_BUF_GUID_OFFSET;

  // Leave the first 16 bytes of the shared memory for matmul operators
  // TODO(intlsy) Remove this if there is not Matmul op or do not need padding
  assert(ALIGNMENT >= 16);
  plan.smem_size += ALIGNMENT;
  for (auto &kv : plan.addrs) {
    kv.second += ALIGNMENT;
  }

  if (plan.smem_size > mirage::config::MAX_SMEM_SIZE) {
    printf("Warning: planned smem_size(%zu) exceeds MAX_SMEM_SIZE(%zu)\n",
           plan.smem_size,
           mirage::config::MAX_SMEM_SIZE);
    // for (const auto &kv : plan.addrs)
    //   printf("sguid(%zu) offset(%zu)\n", kv.first, kv.second);
  }

  return plan;
}

} // namespace transpiler
} // namespace mirage
