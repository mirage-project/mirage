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

#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/forloop_accum.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/common.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/transpiler.h"

#include <algorithm>
#include <unordered_set>

#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/sched_tb_graph.h"
#include "mirage/transpiler/utils.h"
#include "mirage/type.h"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

namespace mirage {
namespace transpiler {

using std::string;
namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

namespace get_layout_detail {

// Get a CuTe layout from dims and strides
//
// The reason why we reverse the vector is that in CuTe, when mapping from an
// integer to a logical coordinate, the first dimension is consider to be the
// "innermost" (here "innermost" has a different meaning from the innermost dim)
//
// For example, assume the tensor has a shape of (3, 2), then 1 will be mapped
// to (1, 0) instead of (0, 1), which is not the same as the C/C++ convention
static string get_cute_layout(vector<int> dims, vector<size_t> strides) {
  assert(dims.size() == strides.size());
  std::reverse(dims.begin(), dims.end());
  std::reverse(strides.begin(), strides.end());
  return fmt("Layout<Shape<$>, Stride<$>>",
             map_to_cute_int(dims),
             map_to_cute_int(strides));
}

static auto get_cute_layout_array(vector<int> dims,
                                  vector<size_t> strides,
                                  bool swap01 = true)
    -> std::pair<std::vector<int>, std::vector<size_t>> {
  assert(dims.size() == strides.size());

  if (swap01) {
    std::reverse(dims.begin(), dims.end());
    std::reverse(strides.begin(), strides.end());
  }

  return {dims, strides};
}

static string get_reversed_cute_layout(vector<int> dims,
                                       vector<size_t> strides) {
  assert(dims.size() == strides.size());
  return fmt("Layout<Shape<$>, Stride<$>>",
             map_to_cute_int(dims),
             map_to_cute_int(strides));
}

template <typename Tensor_T, typename Meta_T>
static string
    get_cute_layout(Tensor_T const &tensor, Meta_T const &meta, int start_dim) {
  return get_cute_layout(
      vector<int>(tensor.dim + start_dim, tensor.dim + tensor.num_dims),
      vector<size_t>(meta.strides + start_dim, meta.strides + tensor.num_dims));
}

template <typename Tensor_T, typename Meta_T>
static string get_swap_01_layout(Tensor_T const &tensor,
                                 Meta_T const &meta,
                                 int start_dim) {
  return get_reversed_cute_layout(
      vector<int>(tensor.dim + start_dim, tensor.dim + tensor.num_dims),
      vector<size_t>(meta.strides + start_dim, meta.strides + tensor.num_dims));
}

// A helper function
template <typename T>
static std::vector<T> mov_to_last(T const *vec, size_t numel, int idx) {
  std::vector<T> result;
  result.reserve(numel);
  result.insert(result.end(), vec, vec + idx);
  result.insert(result.end(), vec + idx + 1, vec + numel);
  result.push_back(vec[idx]);
  return result;
}
} // namespace get_layout_detail

// Get the layout of a STensor
static string get_stensor_layout(tb::STensor const &stensor,
                                 STensorMeta const &meta,
                                 int start_dim = 0,
                                 bool swap01 = true) {

  if (!swap01) {
    if (!meta.is_xor_swizzled) {
      // Do not need to swizzle
      // (Probably swizzled by SHIFT-based swizzling, but we do not care about
      // that)
      return get_layout_detail::get_swap_01_layout(stensor, meta, start_dim);
    } else {
      // XOR-based swizzling
      return fmt(
          "decltype(composition(Swizzle<$, $, $>{}, ${}))",
          meta.xor_swizzle_b,
          meta.xor_swizzle_m,
          meta.xor_swizzle_s,
          get_layout_detail::get_swap_01_layout(stensor, meta, start_dim));
    }
  }

  if (!meta.is_xor_swizzled) {
    // Do not need to swizzle
    // (Probably swizzled by SHIFT-based swizzling, but we do not care about
    // that)
    return get_layout_detail::get_cute_layout(stensor, meta, start_dim);
  } else {
    return fmt("decltype(composition(Swizzle<$, $, $>{}, ${}))",
               meta.xor_swizzle_b,
               meta.xor_swizzle_m,
               meta.xor_swizzle_s,
               get_layout_detail::get_cute_layout(stensor, meta, start_dim));
  }
}

// Move the innermost dim to the last dim, and format it as a CuTe layout
// string.
//
// Assume the tensor has N dimensions and the innermost dim is i, then the
// function is equivalent to torch.permute(tensor, [0, 1, ..., i-1, i+1, ..., N,
// i])
static string mov_last_get_stensor_layout(tb::STensor const &stensor,
                                          STensorMeta const &meta,
                                          int innermost_dim,
                                          bool swap01 = true) {
  tb::STensor new_stensor = stensor;
  STensorMeta new_meta = meta;
  new_meta.swizzled_dim = -1;
  for (int i = 0; i < stensor.num_dims; ++i) {
    int src_dim = i == stensor.num_dims - 1 ? innermost_dim
                  : i < innermost_dim       ? i
                                            : i + 1;
    new_stensor.dim[i] = stensor.dim[src_dim];
    new_meta.strides[i] = meta.strides[src_dim];
    if (src_dim == meta.swizzled_dim) {
      new_meta.swizzled_dim = i;
    }
  }
  new_meta.innermost_dim = stensor.num_dims - 1;

  return get_stensor_layout(new_stensor, new_meta, 0, swap01);
}

// Move the innermost dim to the last dim, and format it as a CuTe layout
// string.
//
// Assume the tensor has N dimensions and the innermost dim is i, then the
// function is equivalent to torch.permute(tensor, [0, 1, ..., i-1, i+1, ..., N,
// i])
static auto mov_last_get_stensor_shape_stride(tb::STensor const &stensor,
                                              STensorMeta const &meta,
                                              int innermost_dim,
                                              bool swap01 = false)
    -> std::pair<std::vector<int>, std::vector<size_t>> {
  tb::STensor new_stensor = stensor;
  STensorMeta new_meta = meta;
  new_meta.swizzled_dim = -1;
  for (int i = 0; i < stensor.num_dims; ++i) {
    int src_dim = i == stensor.num_dims - 1 ? innermost_dim
                  : i < innermost_dim       ? i
                                            : i + 1;
    new_stensor.dim[i] = stensor.dim[src_dim];
    new_meta.strides[i] = meta.strides[src_dim];
    if (src_dim == meta.swizzled_dim) {
      new_meta.swizzled_dim = i;
    }
  }
  new_meta.innermost_dim = stensor.num_dims - 1;

  return {vector<int>(new_stensor.dim, new_stensor.dim + new_stensor.num_dims),
          vector<size_t>(new_meta.strides,
                         new_meta.strides + new_stensor.num_dims)};
  // return get_stensor_layout(new_stensor, new_meta, swap01);
}

// Get the layout of a DTensor tile for input/output operators
static string get_dtensor_tile_layout(kn::DTensor const &dtensor,
                                      DTensorMeta const &d_meta,
                                      tb::STensor const &stensor,
                                      STensorMeta const &s_meta,
                                      int d_innermost_dim) {
  using namespace get_layout_detail;
  return get_cute_layout(
      mov_to_last(stensor.dim,
                  dtensor.num_dims,
                  d_innermost_dim), // Here we use stensor.dim
      mov_to_last(d_meta.strides, dtensor.num_dims, d_innermost_dim));
}

static string append_epilogue_scalars(
    std::vector<std::pair<tb::TBOperator const *, TBSchedOpMeta>> const
        &chain) {
  string res = "const float scalars[] = {";
  if (chain.size() == 1) {
    return res.append("0.0f};");
  }
  for (size_t i = 1; i < chain.size(); i++) {
    if (i == chain.size() - 1) {
      // last one is EpilogueStore
      res.append("0.0f};");
    } else if (is_threadblock_element_unary(chain.at(i).first->op_type)) {
      tb::TBElementUnaryOp const *tb_unary_op =
          dynamic_cast<tb::TBElementUnaryOp const *>(chain.at(i).first);
      res.append(fmt("$f, ", tb_unary_op->scalar));
    } else {
      res.append("0.0f, ");
    }
  }
  return res;
}

static string get_tb_op_str(type::TBOperatorType type) {
  auto toString = [](type::TBOperatorType type) -> string {
    switch (type) {
      case type::TB_EXP_OP:
        return "EXP";
      case type::TB_SQUARE_OP:
        return "SQUARE";
      case type::TB_SQRT_OP:
        return "SQRT";
      case type::TB_MUL_SCALAR_OP:
        return "MULSCALAR";
      case type::TB_SILU_OP:
        return "SILU";
      case type::TB_GELU_OP:
        return "GELU";
      case type::TB_RELU_OP:
        return "RELU";
      case type::TB_CLAMP_OP:
        return "CLAMP";
      default:
        assert(0);
    }
  };

  return toString(type);
}

static std::pair<bool, std::vector<int64_t>>
    add_loop_node_consumer_wait_if_need(
        tb::TBOperator const *op,
        CodeKeeper &code,
        bool is_in_loop,
        std::map<int64_t, tb::TBInputOp const *> &pipeline_inputs) {
  if (!is_in_loop) {
    return {false, {}};
  }

  std::vector<int64_t> input_ids_waited;
  for (int i = 0; i < op->input_tensors.size(); i++) {
    int64_t input_id = op->input_tensors.at(i).guid;
    if (pipeline_inputs.find(input_id) != pipeline_inputs.end()) {
      code.e("int read_idx_$ = hopper_async_pipeline_$.consumer_wait();",
             input_id,
             input_id);
      // only wait once
      pipeline_inputs.erase(input_id);
      input_ids_waited.push_back(input_id);
    }
  }

  if (!input_ids_waited.empty()) {
    return {true, input_ids_waited};
  }

  return {false, {}};
}

// Transpile a custom KN operator (i.e. a custom block graph) into CUDA code
// Will return a CustomOPTranspileResult object. See comments in transpiler.h
// for more details
CustomOPTranspileResult
    Transpiler::transpile_kn_custom_op_hopper(kn::KNCustomizedOp const *op) {
  bool profiling = config.profiling;

  tb::Graph const &g = op->bgraph;
  int num_threads = g.block_dim.x * g.block_dim.y * g.block_dim.z;

  size_t profiler_buf_size =
      profiling ? (g.grid_dim.x * g.grid_dim.y * g.grid_dim.z *
                   (config.num_consumer_wgs + config.num_producer_wgs)) *
                      1000
                : 0;

  // Allocate a kernel name
  static int custom_kernel_idx_counter = 0;
  int cur_custom_kernel_idx = custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  if (GPU_CC::H100 != config.target_cc ||
      (config::MAX_NUM_WARP_GROUPS <
       config.num_consumer_wgs + config.num_producer_wgs) ||
      (num_threads !=
           (config.num_consumer_wgs + config.num_producer_wgs) * 128 &&
       (g.forloop_range > 1))) {
    assert(false && "compiler assertion failure");
    return CustomOPTranspileResult{CUDA_T_CONFIG_ERROR, func_name, 0, 0, ""};
  }

  // int barrier_size = 16 * config.pipeline_stages;
  int tma_barrier_size = 16 * config.pipeline_stages;

  // int 64
  int cluster_barrier_size = 8;

  // Get the schedule
  TBSched sched = get_threadblock_schedule(g);

  get_threadblock_swizzle_plan_hopper(g, sched);

  // Get the memory allocation plan
  TBMemoryPlan mem_plan = get_threadblock_memory_plan(g, sched, true);

  std::vector<TMAParams> tmaParamsList;

  // Generate code prologue
  CodeKeeper code;
  string thread_idx;
  if (g.block_dim.y > 1 || g.block_dim.z > 1) {
    thread_idx = fmt("threadIdx.x + threadIdx.y * $ + threadIdx.z * $",
                     g.block_dim.x,
                     g.block_dim.x * g.block_dim.y);
  } else {
    thread_idx = "threadIdx.x";
  }
  code.e("int thread_idx = $;", thread_idx);
  code.e("static constexpr int NUM_THREADS = $;", 128);

  code.e("static constexpr int CONSUMER_NUM_THREADS = $;",
         config::NUM_THREADS_PER_GROUP * config.num_consumer_wgs);

  // Define STensor as cute::Tensor
  code.e("// STensors");
  code.e("extern __shared__ char buf[];");
  size_t barrier_addr = mem_plan.smem_size;
  for (auto [guid, addr] : mem_plan.addrs) {
    code.e("$ *stensor$_ptr = ($*)(buf + $);",
           get_datatype_str(op->input_tensors[0].data_type),
           guid,
           get_datatype_str(op->input_tensors[0].data_type),
           addr);
  }

  // Define G2SCopy for all input STensors
  code.e("// G->S copy atoms");
  std::unordered_set<tb::TBInputOp const *>
      pipelined_input_ops; // A list of input ops that are software pipelined
                           // (asynchronously G->S copied)

  std::map<int64_t, tb::TBInputOp const *> pipeline_inputs;

  // for release smem_read;
  std::vector<sguid_t> smem_read_output_guids;
  int pipe_index = 0;
  for (TBSchedNode const &node :
       Combine(Combine(sched.pre_loop_nodes, sched.loop_nodes),
               sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_INPUT_OP) {
      auto [_op, op_meta] = node.ops.front();
      tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(_op);
      tb::TBOperator const *output_op = fusion_chain.at(cur_op).back();
      kn::DTensor const &dtensor = cur_op->dtensor;
      tb::STensor const &stensor = output_op->output_tensors.at(0);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
      assert(dtensor.num_dims == stensor.num_dims);
      assert(dtensor.data_type == stensor.data_type);

      code.e("// Copy for G->S: dtensor $ -> stensor $",
             dtensor.guid,
             stensor.guid);

      // Get the starting address of my tile
      // For input tensor that does not have a forloop_dim, the shape of the
      // tile should be identical to the STensor. Otherwise, it should be the
      // shape of STensor * forloop_range
      string offset = "";
      int3 imap = cur_op->input_map;
      for (int dim = 0; dim < 3; ++dim) {
        int div_dim = dim == 0 ? imap.x : dim == 1 ? imap.y : imap.z;
        if (div_dim >= 0) {
          // Dim `div_dim` is divided along `dim`
          int num_tbs = dim == 0   ? g.grid_dim.x
                        : dim == 1 ? g.grid_dim.y
                                   : g.grid_dim.z;
          offset += fmt(" + blockIdx.$*$*$",
                        (char)"xyz"[dim],
                        dtensor.dim[div_dim] / num_tbs,
                        dtensor_meta.strides[div_dim]);
        }
      }

      bool use_chunked_copy = op_meta.is_chunked_input;
      int real_innermost_dim = op_meta.chunked_input_real_innermost_dim;
      bool use_async_copy = op_meta.is_pipelined_input;

      if (!(use_chunked_copy) || (!use_async_copy)) {
        code.e("const $ *dtensor$_tile_ptr = dtensor$_ptr $;",
               get_datatype_str(dtensor.data_type),
               dtensor.guid,
               dtensor.guid,
               offset);
      }

      // assert(use_chunked_copy && use_async_copy);

      // TODO(intlsy) Support swizzled layout
      // TODO(intlsy) Support TMA
      if (!use_chunked_copy) {
        int d_innermost_dim = dtensor_meta.innermost_dim;
        assert(!use_async_copy);
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, d_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        // Non-chunked, synchronous copy
        code.e(
            "using STensor$InputAtom = tb::InputNonChunkedSyncCopy<$, "
            "$, DTensor$TileLayout, NUM_THREADS>;",
            stensor.guid,
            get_datatype_str(stensor.data_type),
            mov_last_get_stensor_layout(stensor, stensor_meta, d_innermost_dim),
            dtensor.guid);
      } else {
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, real_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        if (!use_async_copy) {
          // Chunked, synchronous copy
          code.e("using STensor$InputAtom = tb::InputChunkedSyncCopy<$, "
                 "$, DTensor$TileLayout, NUM_THREADS>;",
                 stensor.guid,
                 get_datatype_str(stensor.data_type),
                 mov_last_get_stensor_layout(
                     stensor, stensor_meta, real_innermost_dim),
                 dtensor.guid);
        } else {
          pipelined_input_ops.insert(cur_op);
          assert(cur_op->output_tensors.size() == 1);
          // make tma

          // gmem tensor
          string gmem_layout = get_layout_detail::get_cute_layout(
              vector<int>(dtensor.dim, dtensor.dim + dtensor.num_dims),
              vector<size_t>(dtensor_meta.strides,
                             dtensor_meta.strides + dtensor.num_dims));

          // imap;
          int forloop_dim = cur_op->forloop_dim;
          bool m_input = stensor_meta.m_input;
          string smem_layout = mov_last_get_stensor_layout(
              stensor, stensor_meta, real_innermost_dim, !m_input);

          auto [dims, strides] = get_layout_detail::get_cute_layout_array(
              vector<int>(dtensor.dim, dtensor.dim + dtensor.num_dims),
              vector<size_t>(dtensor_meta.strides,
                             dtensor_meta.strides + dtensor.num_dims),
              !m_input);

          std::vector<int> partition_logic = {
              imap.x >= 0 ? (dtensor.num_dims - 1 - imap.x) : -1,
              imap.y >= 0 ? (dtensor.num_dims - 1 - imap.y) : -1,
              imap.z >= 0 ? (dtensor.num_dims - 1 - imap.z) : -1};

          string SrcMNKLayout = generate_partitioned_and_expanded_layout(
              dim3(g.grid_dim.x, g.grid_dim.y, g.grid_dim.z),
              dims,
              strides,
              partition_logic,
              g.forloop_range,
              m_input ? forloop_dim : (dtensor.num_dims - 1 - forloop_dim));
          code.e(
              "tb::HopperAsyncPipeline<$> "
              "hopper_async_pipeline_$((void *) (buf + $), (tb::warpgroup_id() "
              "== $ && tb::warp_id() % mirage::config::NUM_WARPS_PER_GROUP == "
              "0), tb::warpgroup_id() < $, $, $);",
              config.pipeline_stages,
              stensor.guid,
              barrier_addr,
              config.num_consumer_wgs,
              config.num_consumer_wgs,
              stensor_meta.num_phy_elems *
                  type::get_datatype_size(stensor.data_type),
              config.num_consumer_wgs);

          code.e(
              "using STensor$InputAtom = tb::InputTMAAsyncCopy<$, $, "
              "$, decltype(tma_$), decltype(hopper_async_pipeline_$), $, $>;",
              stensor.guid,
              get_datatype_str(stensor.data_type),
              smem_layout,
              SrcMNKLayout,
              dtensor.guid,
              stensor.guid,
              stensor_meta.m_input,
              g.forloop_range);
          pipe_index++;
          barrier_addr += tma_barrier_size;

          pipeline_inputs[stensor.guid] = cur_op;

          tmaParamsList.push_back((TMAParams(
              dtensor_meta.input_idx,
              dtensor.guid,
              stensor.guid,
              SrcMNKLayout,
              smem_layout,
              stensor_meta.m_input,
              fmt("shape(${})", smem_layout),
              {1, 1, 1},
              dims,
              strides,
              partition_logic,
              g.forloop_range,
              m_input ? forloop_dim : (dtensor.num_dims - 1 - forloop_dim))));
        }
      }
    }
  }
  code.e("");
  code.e("__syncthreads();");

  // add tma templates for H100
  if (GPU_CC::H100 == config.target_cc) {

    assert(g.cluster_dim.x > 0 && g.cluster_dim.y > 0 && g.cluster_dim.z > 0);
    string tma;
    string tmplt;
    for (size_t i = 0; i < tmaParamsList.size(); ++i) {
      if (i == 0) {
        tmplt.append("template <");
      }
      TMAParams &params = tmaParamsList.at(i);
      tmplt.append("class TMA_" + std::to_string(params.guid));
      tma.append("CUTE_GRID_CONSTANT TMA_" + std::to_string(params.guid) +
                 " const " + "tma_" + std::to_string(params.guid));

      if (i != tmaParamsList.size() - 1) {
        tmplt.append(", ");
      } else {
        tmplt.append(">");
      }
      tma.append(", ");
    }

    if (profiling) {
      code.e_front(
          "__global__ void  __launch_bounds__($) "
          "$($ $, $, uint64_t *profiler_buffer) {",
          num_threads,
          func_name,
          tma,
          map<kn::DTensor, string>(op->output_tensors,
                                   [](kn::DTensor const &dtensor) -> string {
                                     return fmt(
                                         "$* dtensor$_ptr",
                                         get_datatype_str(dtensor.data_type),
                                         dtensor.guid);
                                   }),
          map<kn::DTensor, string>(
              op->input_tensors, [](kn::DTensor const &dtensor) -> string {
                return fmt("$ const* dtensor$_ptr",
                           get_datatype_str(dtensor.data_type),
                           dtensor.guid);
              }));
    } else {
      code.e_front(
          "__global__ void  __launch_bounds__($) "
          "$($ $, $) {",
          num_threads,
          func_name,
          tma,
          map<kn::DTensor, string>(op->output_tensors,
                                   [](kn::DTensor const &dtensor) -> string {
                                     return fmt(
                                         "$* dtensor$_ptr",
                                         get_datatype_str(dtensor.data_type),
                                         dtensor.guid);
                                   }),
          map<kn::DTensor, string>(
              op->input_tensors, [](kn::DTensor const &dtensor) -> string {
                return fmt("$ const* dtensor$_ptr",
                           get_datatype_str(dtensor.data_type),
                           dtensor.guid);
              }));
    }

    code.e_front(tmplt);
    code.inc_indent();
  }

  // add mem_size based on tma copies
  mem_plan.smem_size += tmaParamsList.size() * config.pipeline_stages * 16;

  // Erase the lowest 16 bytes to 0 for GEMM
  code.e("*((uint128_t*)buf) = 0ul;");
  code.e("");

  code.e("");

  // Launch G->S copy atoms for all pre-loop-ops
  int num_pre_loop_copies = 0;
  for (TBSchedNode const &sched_node : sched.pre_loop_nodes) {
    // Currently only non-fused input ops are allowed to appear in
    // pre_loop_nodes check against this condition
    assert(sched_node.type == tb_sched_node_t::OPERATOR);
    assert(sched_node.ops.size() == 1); // Should not be fused
    tb::TBOperator const *op = sched_node.ops[0].first;
    assert(op->op_type == type::TB_INPUT_OP);
    tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(op);
    tb::STensor const &stensor = cur_op->output_tensors.at(0);
    assert(cur_op->forloop_dim == -1);
    assert(!pipelined_input_ops.count(
        cur_op)); // An input op in pre_loop_nodes should not be software
                  // pipelined since they do not have forloop_dim
    num_pre_loop_copies += 1;
    code.e("STensor$InputAtom::run(stensor$_ptr, "
           "dtensor$_tile_ptr, "
           "thread_idx);",
           stensor.guid,
           stensor.guid,
           cur_op->dtensor.guid);
  }
  code.e("");

  // Define S2GCopy for all output STensors
  code.e("// S->G copy atoms");
  for (TBSchedNode const &node :
       Combine(Combine(sched.pre_loop_nodes, sched.loop_nodes),
               sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_OUTPUT_OP) {
      auto [_op, op_meta] = node.ops.front();
      tb::TBOutputOp const *cur_op = dynamic_cast<tb::TBOutputOp const *>(_op);
      tb::STensor const &stensor = cur_op->input_tensors.at(0);
      kn::DTensor const &dtensor = cur_op->dtensor;
      STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      assert(dtensor.num_dims == stensor.num_dims);
      assert(dtensor.data_type == stensor.data_type);

      code.e("// Copy for S->G: stensor $ -> dtensor $",
             stensor.guid,
             dtensor.guid);

      // Get the starting address of my tile
      // For output tensor that does not have a forloop_dim, the shape of the
      // tile should be identical to the STensor. Otherwise, it should be the
      // shape of STensor * forloop_range
      string offset = "";
      int3 omap = cur_op->output_map;
      for (int dim = 0; dim < 3; ++dim) {
        int div_dim = dim == 0 ? omap.x : dim == 1 ? omap.y : omap.z;
        int num_tbs = dim == 0   ? g.grid_dim.x
                      : dim == 1 ? g.grid_dim.y
                                 : g.grid_dim.z;
        if (num_tbs > 1) {
          // The output tensor MUST be divided along this dimension, as stated
          // in the paper
          assert(div_dim >= 0);
          offset += fmt(" + blockIdx.$*$*$",
                        (char)"xyz"[dim],
                        dtensor.dim[div_dim] / num_tbs,
                        dtensor_meta.strides[div_dim]);
        }
      }
      code.e("$ *dtensor$_tile_ptr = dtensor$_ptr $;",
             get_datatype_str(dtensor.data_type),
             dtensor.guid,
             dtensor.guid,
             offset);

      bool use_chunked_copy = op_meta.is_chunked_output;
      int real_innermost_dim = op_meta.chunked_output_real_innermost_dim;

      if (!use_chunked_copy) {
        int d_innermost_dim = dtensor_meta.innermost_dim;
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, d_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        code.e("using STensor$OutputAtom = tb::OutputNonChunkedSyncCopy<$, "
               "DTensor$TileLayout, $, NUM_THREADS>;",
               stensor.guid,
               get_datatype_str(dtensor.data_type),
               dtensor.guid,
               mov_last_get_stensor_layout(
                   stensor, stensor_meta, d_innermost_dim));
      } else {
        string dtensor_tile_layout = get_dtensor_tile_layout(
            dtensor, dtensor_meta, stensor, stensor_meta, real_innermost_dim);
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        code.e("using STensor$OutputAtom = tb::OutputChunkedSyncCopy<$, "
               "DTensor$TileLayout, $, NUM_THREADS>;",
               stensor.guid,
               get_datatype_str(dtensor.data_type),
               dtensor.guid,
               mov_last_get_stensor_layout(
                   stensor, stensor_meta, real_innermost_dim));
      }
      // TODO(intlsy) Support TMA
    }
  }
  code.e("");

  // Clear all accumulators
  // get all pipeline stensors

  int num_clear_accums = 0;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if ((last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP ||
         last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP) &&
        !last_op_meta.is_accum_in_reg) {
      tb::TBForloopAccumOp const *accum_op =
          dynamic_cast<tb::TBForloopAccumOp const *>(last_op);
      tb::STensor const &accum = accum_op->output_tensors.at(0);
      STensorMeta const &accum_meta = stensor_metas.at(accum.guid);
      size_t num_elems = 0;
      for (int i = 0; i < accum.num_dims; ++i) {
        num_elems = std::max(num_elems, accum.dim[i] * accum_meta.strides[i]);
      }
      code.e("tb::ClearAccumlatorKernel<$, $, "
             "NUM_THREADS>::run(stensor$_ptr, thread_idx);",
             get_datatype_str(accum.data_type),
             num_elems,
             accum.guid);
      num_clear_accums += 1;
    }
  }
  code.e("");

  // Initialize all max accumulators
  int num_init_max_accums = 0;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type == type::TB_FORLOOP_ACCUM_MAX_OP &&
        !last_op_meta.is_accum_in_reg) {
      tb::TBForloopAccumOp const *accum_op =
          dynamic_cast<tb::TBForloopAccumOp const *>(last_op);
      tb::STensor const &accum = accum_op->output_tensors.at(0);
      STensorMeta const &accum_meta = stensor_metas.at(accum.guid);
      size_t num_elems = 0;
      for (int i = 0; i < accum.num_dims; ++i) {
        num_elems = std::max(num_elems, accum.dim[i] * accum_meta.strides[i]);
      }
      code.e("tb::InitMaxAccumulatorKernel<$, $, "
             "NUM_THREADS>::run(stensor$_ptr, thread_idx);",
             get_datatype_str(accum.data_type),
             num_elems,
             accum.guid);
      num_init_max_accums += 1;
    }
  }

  // Pre-define all matmul ops and allocate accumulators (if needed)
  // Since we may want to place the accumulator of a matmul op in register
  // files, we may need to allocate the accumulator in advance, and that
  // requires us to define the kernel (`using Matmul$Kernel = ...`) in advance
  for (TBSchedNode const &node :
       Combine(sched.loop_nodes, sched.post_loop_nodes)) {
    if (node.type == tb_sched_node_t::OPERATOR &&
        node.ops.front().first->op_type == type::TB_MATMUL_OP) {
      tb::TBOperator const *op = node.ops.front().first;
      tb::TBOperator const *output_op = node.ops.back().first;
      tb::STensor const &input0 = op->input_tensors.at(0);
      tb::STensor const &input1 = op->input_tensors.at(1);
      tb::STensor const &output = output_op->output_tensors.at(0);
      STensorMeta meta0 = stensor_metas.at(input0.guid);
      STensorMeta meta1 = stensor_metas.at(input1.guid);
      STensorMeta meta2 = stensor_metas.at(output.guid);
      int num_dims = input0.num_dims;
      assert(input1.num_dims == num_dims && output.num_dims == num_dims);
      int m = output.dim[num_dims - 2];
      int n = output.dim[num_dims - 1];
      int k = input0.dim[num_dims - 1];
      assert(input0.dim[num_dims - 2] == m && input0.dim[num_dims - 1] == k);
      assert(input1.dim[num_dims - 2] == k && input1.dim[num_dims - 1] == n);

      // Pick up MMA atom
      // TODO(intlsy) May calculate AB via (B^T A^T)^T when M is relatively
      // small
      if (GPU_CC::H100 == config.target_cc) {
        // Hopper wgmma
        assert(num_threads >= 128);
      } else {
        // TODO(intlsy): Support more architectures
        assert(0 && "Unsupported GPU Architecture");
      }

      bool is_ldmatrix_avail = config.target_cc >= GPU_CC::T4;
      bool is_stmatrix_avail = false;

      int num_exps_before_store = std::count_if(
          node.ops.begin(), node.ops.end(), [](auto &op_and_meta) {
            return op_and_meta.first->op_type == type::TB_EXP_OP;
          });
      bool is_store_accum =
          node.ops.back().first->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP ||
          node.ops.back().first->op_type ==
              type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
          node.ops.back().first->op_type == type::TB_FORLOOP_ACCUM_MAX_OP;
      bool is_accum_in_reg = node.ops.back().second.is_accum_in_reg;

      // For threadblock matmul, cute requires 2-d matrices as inputs / outputs,
      // we assert that all other leading dimensions are of size 1, and only use
      // the last two dimensions when generating layouts
      code.e("using Matmul$LayoutA = $;",
             output.guid,
             get_stensor_layout(input0, meta0, num_dims - 2 /*start_dim*/));
      code.e("using Matmul$LayoutB = $;",
             output.guid,
             get_stensor_layout(input1, meta1, num_dims - 2 /*start_dim*/));
      code.e("using Matmul$LayoutC = $;",
             output.guid,
             get_stensor_layout(output, meta2, num_dims - 2 /*start_dim*/));

      code.e("using Matmul$Kernel = tb::Hopper_Matmul<$, "
             "$, $, Matmul$LayoutA, Matmul$LayoutB, "
             "Matmul$LayoutC, NUM_THREADS, "
             "$, $, $, $, $, $>;",
             output.guid,
             get_datatype_str(input0.data_type),
             is_ldmatrix_avail,
             is_stmatrix_avail,
             output.guid,
             output.guid,
             output.guid,
             num_exps_before_store,
             is_accum_in_reg ? false : is_store_accum,
             config.num_consumer_wgs > 1 ? true : false,
             meta0.is_pipelined_input,
             meta1.is_pipelined_input,
             config.pipeline_stages);
      if (is_accum_in_reg) {
        code.e("auto matmul_$_accum = Matmul$Kernel::get_mma_rC(thread_idx);",
               output.guid,
               output.guid);
      }
      code.e("");
    }
  }
  code.e("__syncthreads();");

  if (num_pre_loop_copies > 0 || num_clear_accums > 0 ||
      num_init_max_accums > 0) {
    code.e("__syncthreads();");
    code.e("");
  }

  bool pipe_tma = !pipeline_inputs.empty();

  // if there is asyc copy defined
  if (pipe_tma) {
    code.e("int warpgroup_id = tb::warpgroup_id();");
    if (profiling) {
      code.e("PROFILER_CLOSURE_PARAMS_DECL");
      code.e("PROFILER_INIT(profiler_buffer, warpgroup_id, $, (threadIdx.x % "
             "128 == 0));",
             config.num_consumer_wgs + config.num_producer_wgs);
    }
    // run producers
    code.e("if (warpgroup_id == $) {", config.num_consumer_wgs);
    // allocate tma register files
    uint32_t tma_reg = config.num_consumer_wgs == 1 ? 56 : 32;

    // code.e("tb::wg_decrease_regs<$>();", tma_reg);
    code.e("if (tb::warp_id_in_wg() == 0) {");

    code.e("for (uint32_t for_idx = 0; for_idx < $; for_idx++) {",
           g.forloop_range);
    for (auto const &[stensor_id, op] : pipeline_inputs) {
      if (profiling) {
        code.e("PROFILER_EVENT_START($, $);",
               (op->op_type - type::TB_UNKOWN),
               "static_cast<uint32_t>(for_idx)");
      }
      code.e(fmt("STensor$InputAtom::run(tma_$, stensor$_ptr, "
                 " $, $, $, for_idx, hopper_async_pipeline_$);",
                 stensor_id,
                 op->dtensor.guid,
                 stensor_id,
                 op->input_map.x,
                 op->input_map.y,
                 op->input_map.z,
                 stensor_id));
      if (profiling) {
        code.e("PROFILER_EVENT_END($, $);",
               (op->op_type - type::TB_UNKOWN),
               "static_cast<uint32_t>(for_idx)");
      }
    }
    code.e("}");
    code.e("}");
    code.e("}");
  }
  // code.e("}");
  // code.e("}");
  // code.e("}");

  // A lambda function that transpiles a chain of (fusable) operators to an
  // epilogue Will automatically ignore the first operator in the `chain`
  // argument
  auto transpile_fusion_epilogue =
      [&](std::vector<std::pair<tb::TBOperator const *, TBSchedOpMeta>> const
              &chain,
          string dtype) -> string {
    size_t chain_size = chain.size();
    if (chain_size == 1) {
      // Not fused with anything
      return fmt("tb::EpilogueStore<$>", dtype);
    }
    // Deal with the last operator
    string res = fmt("tb::EpilogueStore<$>", dtype);
    for (size_t i = chain_size - 1; i >= 1; --i) {
      tb::TBOperator const *cur_op = chain[i].first;
      if (cur_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
        // Can only occur as the last operator in the chain
        assert(i == chain_size - 1);
        res = fmt("tb::EpilogueStoreAccum<$>", dtype);
      } else if (cur_op->op_type == type::TB_EXP_OP) {
        res = fmt("tb::EpilogueExp<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_SILU_OP) {
        res = fmt("tb::EpilogueSILU<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_GELU_OP) {
        res = fmt("tb::EpilogueGELU<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_RELU_OP) {
        res = fmt("tb::EpilogueRELU<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_CLAMP_OP) {
        res = fmt("tb::EpilogueClamp<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_SQUARE_OP) {
        res = fmt("tb::EpilogueSquare<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_SQRT_OP) {
        res = fmt("tb::EpilogueSqrt<$, $>", dtype, res);
      } else if (cur_op->op_type == type::TB_MUL_SCALAR_OP) {
        res = fmt("tb::EpilogueMulScalar<$, $>", dtype, res);
      } else {
        assert(0 && "Unknown operator type");
      }
    }
    return res;
  };

  // A lambda function that transpiles an TBSchedNode
  auto transpile_tb_sched_node = [&](TBSchedNode const &sched_node,
                                     CodeKeeper &code,
                                     std::map<int64_t, tb::TBInputOp const *>
                                         &pipeline_inputs,
                                     bool is_in_loop) {
    if (sched_node.type == tb_sched_node_t::SYNCTHREADS && is_in_loop) {
      // code.e("tb::wg_sync<128>(9 + warpgroup_id);");
    } else if (sched_node.type == tb_sched_node_t::SYNCTHREADS) {
      code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
    } else {
      auto [op, first_op_meta] = sched_node.ops.front();
      auto [output_op, output_op_meta] = sched_node.ops.back();
      assert(output_op == fusion_chain.at(op).back());
      std::string op_type_str;
      to_json(op_type_str, op->op_type);
      code.e("{");
      code.e("// OP type: $", op_type_str);

      auto [need_advance_pipeline, pipe_ids] =
          add_loop_node_consumer_wait_if_need(
              op, code, is_in_loop, pipeline_inputs);

      // define
      if (pipe_tma && profiling) {
        // 2000 - 2999
        code.e("PROFILER_EVENT_START($, $);",
               (op->op_type - type::TB_UNKOWN),
               is_in_loop ? "static_cast<uint32_t>(for_idx)"
                          : "static_cast<uint32_t>(0)");
      }

      switch (op->op_type) {
        case type::TB_OUTPUT_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          tb::TBOutputOp const *cur_op =
              dynamic_cast<tb::TBOutputOp const *>(op);
          // Currently in Mirage core, an output op must have forloop_dim = -1
          assert(!is_in_loop);
          assert(cur_op->forloop_dim == -1);
          if (cur_op->forloop_dim >= 0) {
            assert(0);
          } else {
            tb::STensor const &stensor = cur_op->input_tensors.at(0);
            kn::DTensor const &dtensor = cur_op->dtensor;
            code.e("STensor$OutputAtom::run(dtensor$_tile_ptr, stensor$_ptr, "
                   "thread_idx);",
                   stensor.guid,
                   dtensor.guid,
                   stensor.guid);
          }
          break;
        }
        case type::TB_MATMUL_OP: {
          tb::STensor const &input0 = op->input_tensors.at(0);
          tb::STensor const &input1 = op->input_tensors.at(1);
          tb::STensor const &output = output_op->output_tensors.at(0);
          sguid_t output_guid = output.guid;

          // always pipeline for MMA
          if (need_advance_pipeline) {
            smem_read_output_guids.push_back(output_guid);
            // code.e(fmt("PipelineState smem_pipe_read_$;", output_guid));
            if (output_op_meta.is_accum_in_reg) {
              // Accumulator is in register
              code.e("Matmul$Kernel::run(matmul_$_accum, stensor$_ptr, "
                     "stensor$_ptr, (char*)(buf+0), thread_idx, read_idx_$);",
                     output_guid,
                     output_guid,
                     input0.guid,
                     input1.guid,
                     pipe_ids.at(0));
            } else {
              code.e("auto mma_rC = Matmul$Kernel::get_mma_rC(thread_idx);",
                     output_guid);
              code.e("Matmul$Kernel::run(mma_rC, stensor$_ptr, stensor$_ptr, "
                     "(char*)(buf+0), thread_idx, read_idx_$);",
                     output_guid,
                     input0.guid,
                     input1.guid,
                     pipe_ids.at(0));
              code.e("Matmul$Kernel::write_back_mma_rC(stensor$_ptr, mma_rC, "
                     "thread_idx);",
                     output_guid,
                     output_guid);
            }
          } else {
            if (output_op_meta.is_accum_in_reg) {
              code.e("Matmul$Kernel::run(matmul_$_accum, stensor$_ptr, "
                     "stensor$_ptr, (char*)(buf+0), thread_idx);",
                     output_guid,
                     output_guid,
                     input0.guid,
                     input1.guid);
            } else {
              code.e("auto mma_rC = Matmul$Kernel::get_mma_rC(thread_idx);",
                     output_guid);
              code.e("Matmul$Kernel::run(mma_rC, stensor$_ptr, stensor$_ptr, "
                     "(char*)(buf+0), thread_idx);",
                     output_guid,
                     input0.guid,
                     input1.guid);
              code.e("Matmul$Kernel::write_back_mma_rC(stensor$_ptr, mma_rC, "
                     "thread_idx);",
                     output_guid,
                     output_guid);
            }
          }

          break;
        }
        case type::TB_EXP_OP:
        case type::TB_SILU_OP:
        case type::TB_GELU_OP:
        case type::TB_RELU_OP:
        case type::TB_CLAMP_OP:
        case type::TB_SQUARE_OP:
        case type::TB_SQRT_OP:
        case type::TB_MUL_SCALAR_OP: {
          tb::TBElementUnaryOp const *cur_op =
              dynamic_cast<tb::TBElementUnaryOp const *>(op);
          tb::STensor const &input = cur_op->input_tensors.at(0);
          tb::STensor const &output = output_op->output_tensors.at(0);
          assert(input.num_dims == output.num_dims);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;

          // at least one dim exists that fullfill the requirement:
          // dim i in input&output tensor == meta.innermost_dim or
          // meta.swizzled_dim
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, output}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string final_out_layout = mov_last_get_stensor_layout(
              output, stensor_metas.at(output.guid), iter_dim);
          code.e("using InLayout = $;", in_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(
              sched_node.ops, get_datatype_str(input.data_type));
          // Define and run the kernel
          code.e("using Kernel = tb::ElementUnaryKernel<$, "
                 "tb::ElementUnaryOpType::$, OutLayout, InLayout, "
                 "CONSUMER_NUM_THREADS, $>;",
                 get_datatype_str(input.data_type),
                 get_tb_op_str(cur_op->op_type),
                 epilogue);
          code.e(append_epilogue_scalars(sched_node.ops));
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx, $, "
                 "scalars);",
                 output.guid,
                 input.guid,
                 cur_op->scalar);
          break;
        }
        case type::TB_ADD_OP:
        case type::TB_MUL_OP:
        case type::TB_DIV_OP:
        case type::TB_SUB_OP:
        case type::TB_POW_OP: {
          tb::STensor const &input0 = op->input_tensors.at(0);
          tb::STensor const &input1 = op->input_tensors.at(1);
          tb::STensor const &output = output_op->output_tensors.at(0);
          assert(input0.num_dims == input1.num_dims &&
                 input0.num_dims == output.num_dims);
          int num_dims = input0.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input0, input1, output}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define op type
          string op_type_str = op->op_type == type::TB_ADD_OP   ? "ADD"
                               : op->op_type == type::TB_MUL_OP ? "MUL"
                               : op->op_type == type::TB_DIV_OP ? "DIV"
                               : op->op_type == type::TB_SUB_OP ? "SUB"
                               : op->op_type == type::TB_POW_OP ? "POW"
                                                                : "";
          assert(op_type_str != "");
          // Define layouts
          string in0_layout = mov_last_get_stensor_layout(
              input0, stensor_metas.at(input0.guid), iter_dim);
          string in1_layout = mov_last_get_stensor_layout(
              input1, stensor_metas.at(input1.guid), iter_dim);
          string final_out_layout = mov_last_get_stensor_layout(
              output, stensor_metas.at(output.guid), iter_dim);
          code.e("using In0Layout = $;", in0_layout);
          code.e("using In1Layout = $;", in1_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(
              sched_node.ops, get_datatype_str(input0.data_type));
          // Define and run the kernel
          code.e("using Kernel = tb::ElementBinaryKernel<$, "
                 "tb::ElementBinaryOpType::$, OutLayout, In0Layout, In1Layout, "
                 "CONSUMER_NUM_THREADS, $>;",
                 get_datatype_str(input0.data_type),
                 op_type_str,
                 epilogue);
          code.e(append_epilogue_scalars(sched_node.ops));
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, stensor$_ptr, "
                 "thread_idx, scalars);",
                 output.guid,
                 input0.guid,
                 input1.guid);
          break;
        }
        case type::TB_REDUCTION_0_OP:
        case type::TB_REDUCTION_1_OP:
        case type::TB_REDUCTION_2_OP:
        case type::TB_REDUCTION_0_TO_DIMX_OP:
        case type::TB_REDUCTION_1_TO_DIMX_OP:
        case type::TB_REDUCTION_2_TO_DIMX_OP: {
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &output = output_op->output_tensors.at(0);
          STensorMeta input_meta = stensor_metas.at(input.guid);
          STensorMeta final_output_meta = stensor_metas.at(output.guid);
          assert(input.num_dims == output.num_dims);
          int num_dims = input.num_dims;
          int reduc_dim = op->op_type >= type::TB_REDUCTION_0_TO_DIMX_OP
                              ? op->op_type - type::TB_REDUCTION_0_TO_DIMX_OP
                              : op->op_type - type::TB_REDUCTION_0_OP;
          assert(0 <= reduc_dim && reduc_dim < num_dims);
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            if (i == reduc_dim) {
              continue;
            }
            bool failed = false;
            for (tb::STensor const &stensor : {input, output}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          assert(iter_dim != reduc_dim);
          // Define layouts
          string in_layout =
              mov_last_get_stensor_layout(input, input_meta, iter_dim);
          string final_out_layout =
              mov_last_get_stensor_layout(output, final_output_meta, iter_dim);
          int cute_reduc_dim = reduc_dim < iter_dim ? num_dims - 1 - reduc_dim
                                                    : num_dims - reduc_dim;
          code.e("using InLayout = $;", in_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(
              sched_node.ops, get_datatype_str(input.data_type));
          // Define and run the kernel
          code.e("using Kernel = tb::ReductionKernel<$, "
                 "OutLayout, InLayout, $, CONSUMER_NUM_THREADS, $>;",
                 get_datatype_str(input.data_type),
                 cute_reduc_dim,
                 epilogue);
          code.e(append_epilogue_scalars(sched_node.ops));
          code.e(
              "Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx, scalars);",
              output.guid,
              input.guid);
          break;
        }
        case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          assert(is_in_loop);
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &accum = op->output_tensors.at(0);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, accum}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string accum_layout = mov_last_get_stensor_layout(
              accum, stensor_metas.at(accum.guid), iter_dim);
          code.e("using Kernel = tb::ForloopAccumKernel<$, $, $, "
                 "NUM_THREADS>;",
                 get_datatype_str(input.data_type),
                 accum_layout,
                 in_layout);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx);",
                 accum.guid,
                 input.guid);
          break;
        }
        case type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          assert(is_in_loop);
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &rescale = op->input_tensors.at(1);
          tb::STensor const &accum = op->output_tensors.at(0);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, rescale, accum}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string rescale_layout = mov_last_get_stensor_layout(
              rescale, stensor_metas.at(rescale.guid), iter_dim);
          string accum_layout = mov_last_get_stensor_layout(
              accum, stensor_metas.at(accum.guid), iter_dim);
          code.e("using Kernel = tb::ForloopAccumRescaleKernel<$, $, $, $, "
                 "NUM_THREADS>;",
                 get_datatype_str(input.data_type),
                 accum_layout,
                 in_layout,
                 rescale_layout);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, stensor$_ptr, "
                 "thread_idx);",
                 accum.guid,
                 input.guid,
                 rescale.guid);
          break;
        }
        case type::TB_FORLOOP_ACCUM_MAX_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          assert(is_in_loop);
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &accum = op->output_tensors.at(0);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, accum}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && meta.swizzled_dim != i) {
                failed = true;
                break;
              }
            }
            if (!failed) {
              iter_dim = i;
              break;
            }
          }
          assert(iter_dim != -1);
          // Define layouts
          string in_layout = mov_last_get_stensor_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string accum_layout = mov_last_get_stensor_layout(
              accum, stensor_metas.at(accum.guid), iter_dim);
          code.e(
              "using Kernel = tb::ForloopAccumMaxKernel<$, $, $, NUM_THREADS>;",
              get_datatype_str(input.data_type),
              accum_layout,
              in_layout);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx);",
                 accum.guid,
                 input.guid);
          break;
        }
        case type::TB_CONCAT_0_OP:
        case type::TB_CONCAT_1_OP:
        case type::TB_CONCAT_2_OP: {
          assert(0 && "Not implemented");
          break;
        }
        case type::TB_CONCAT_THEN_MATMUL_OP: {
          assert(0 && "Not implemented");
          break;
        }
        case type::TB_CUSTOMIZED_OP: {
          assert(0 && "Not implemented");
          break;
        }
        default: {
          assert(fmt("Unknown TB op: $", op->op_type).c_str());
        }
      }
      if (pipe_tma && profiling) {
        code.e("PROFILER_EVENT_END($, $);",
               (op->op_type - type::TB_UNKOWN),
               is_in_loop ? "static_cast<uint32_t>(for_idx)"
                          : "static_cast<uint32_t>(0)");
      }
      code.e("}");
    }

    return CUDA_T_SUCCESS;
  };

  // Declare the for loop
  // TODO(intlsy) Remove the loop when `g.forloop_range` is 1
  // TODO(intlsy) Loop unrolling

  if (pipe_tma) {
    code.e("else {");
    // allocate register files for wgmma
    uint32_t mma_reg = config.num_consumer_wgs == 1
                           ? 256
                           : (config.num_consumer_wgs == 2 ? 232 : 160);
    // code.e("tb::wg_increase_regs<$>();", mma_reg);
    code.e("// Consumer main loop");
  }

  std::map<int64_t, tb::TBInputOp const *> copy_of_inputs = pipeline_inputs;
  assert(g.forloop_range >= 1);

  code.e("for (uint32_t for_idx = 0; for_idx < $; for_idx++) {",
         g.forloop_range);

  // warpgroup_id
  for (TBSchedNode const &sched_node : sched.loop_nodes) {
    if (sched_node.type == tb_sched_node_t::OPERATOR &&
        sched_node.ops[0].first->op_type == type::TB_INPUT_OP &&
        pipelined_input_ops.count(
            dynamic_cast<tb::TBInputOp const *>(sched_node.ops[0].first))) {
      continue;
    }
    CodeKeeper res;

    TranspileErrorType err =
        transpile_tb_sched_node(sched_node, res, pipeline_inputs, true);

    code << res;
    if (err != CUDA_T_SUCCESS) {
      return CustomOPTranspileResult{err, func_name, 0, 0, ""};
    }
  }

  if (!copy_of_inputs.empty()) {
    for (auto const &[pipe_id, op] : copy_of_inputs) {
      code.e("hopper_async_pipeline_$.consumer_release();", pipe_id);
    }
  }

  code.e("}"); // For loop

  // Write back in-register accumulators
  int num_in_reg_accums = 0;
  CodeKeeper in_reg_writeback;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if ((last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP ||
         last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
         last_op->op_type == type::TB_FORLOOP_ACCUM_MAX_OP) &&
        !last_op_meta.is_accum_in_reg) {
      tb::TBForloopAccumOp const *accum_op =
          dynamic_cast<tb::TBForloopAccumOp const *>(last_op);
      tb::STensor const &accum = accum_op->output_tensors.at(0);
      in_reg_writeback.e("Matmul$Kernel::write_back_mma_rC(stensor$_ptr, "
                         "matmul_$_accum, thread_idx);",
                         accum.guid,
                         accum.guid,
                         accum.guid);

      num_in_reg_accums += 1;
    }
  }
  if (num_in_reg_accums > 0) {
    code.e("// Write back in-register accumulators");
    code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
    // code.e("__syncthreads();"); // Need this __syncthreads() to make sure no
    //                             // thread is still in the for loop
    code << in_reg_writeback;
  }

  // Transpile the epilogue of the kernel
  if (!sched.post_loop_nodes.empty()) {
    code.e("// The epilogue (kernels outside the loop)");
    code.e("tb::wg_sync<CONSUMER_NUM_THREADS>(8);");
    for (TBSchedNode const &sched_node : sched.post_loop_nodes) {
      CodeKeeper res;
      TranspileErrorType err =
          transpile_tb_sched_node(sched_node, res, pipeline_inputs, false);
      code << res;
      if (err != CUDA_T_SUCCESS) {
        return CustomOPTranspileResult{err, func_name, 0, 0, "", tmaParamsList};
      }
    }
  }
  if (pipe_tma) {
    code.e("}");
  }

  code.e("}"); // kernel

  // mem_plan.smem_size += tmaParamsList.size() * config.pipeline_stages * 16;
  return CustomOPTranspileResult{CUDA_T_SUCCESS,
                                 func_name,
                                 mem_plan.smem_size,
                                 profiler_buf_size,
                                 code.to_string(),
                                 tmaParamsList};
}

} // namespace transpiler
} // namespace mirage