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

template <typename Tensor_T, typename Meta_T>
static string
    get_cute_layout(Tensor_T const &tensor, Meta_T const &meta, int start_dim) {
  return get_cute_layout(
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
                                 int start_dim = 0) {
  if (!meta.is_xor_swizzled) {
    // Do not need to swizzle
    // (Probably swizzled by SHIFT-based swizzling, but we do not care about
    // that)
    return get_layout_detail::get_cute_layout(stensor, meta, start_dim);
  } else {
    // XOR-based swizzling
    return fmt("decltype(composition(Swizzle<$, $, $>{}, ${}))",
               meta.xor_swizzle_b,
               meta.xor_swizzle_m,
               meta.xor_swizzle_s,
               get_layout_detail::get_cute_layout(stensor, meta, start_dim));
  }
}

// Get the aligned layout for MMA stensor
static string get_mma_stensor_aligned_layout(
    tb::STensor const &stensor,
    STensorMeta const &meta,
    std::tuple<int, int, int> const &mma_atom_mnk,
    bool m,
    bool output,
    int start_dim = 0) {

  tb::STensor new_stensor = stensor;
  STensorMeta new_meta = meta;
  bool n = (!m) && (!output);

  assert(stensor.num_dims - start_dim == 2);
  int aligned_shape = -1;

  for (int i = start_dim; i < stensor.num_dims; i++) {
    // m/n
    if (i == start_dim && m) {
      // K
      aligned_shape = std::max(stensor.dim[i], std::get<2>(mma_atom_mnk));
    } else if (i == start_dim && n) {
      // N
      aligned_shape = std::max(stensor.dim[i], std::get<1>(mma_atom_mnk));
    } else if (i == stensor.num_dims - 1 && m) {
      // M
      aligned_shape = std::max(stensor.dim[i], std::get<0>(mma_atom_mnk));
    } else if (i == stensor.num_dims - 1 && n) {
      // K
      aligned_shape = std::max(stensor.dim[i], std::get<2>(mma_atom_mnk));
    } else if (i == start_dim && output) {
      // N
      aligned_shape = std::max(stensor.dim[i], std::get<1>(mma_atom_mnk));
    } else if (i == stensor.num_dims - 1 && output) {
      // M
      aligned_shape = std::max(stensor.dim[i], std::get<0>(mma_atom_mnk));
    } else {
      assert(false);
    }
    new_stensor.dim[i] = aligned_shape;
  }

  // update strides
  int innermost_dim = new_meta.innermost_dim;
  for (int i = start_dim; i < stensor.num_dims; i++) {
    if (i != innermost_dim) {
      new_meta.strides[i] = new_stensor.dim[innermost_dim];
    }
  }
  return get_stensor_layout(new_stensor, new_meta, new_stensor.num_dims - 2);
}

// Move the innermost dim to the last dim, and format it as a CuTe layout
// string.
//
// Assume the tensor has N dimensions and the innermost dim is i, then the
// function is equivalent to torch.permute(tensor, [0, 1, ..., i-1, i+1, ..., N,
// i])
static string mov_last_get_stensor_layout(tb::STensor const &stensor,
                                          STensorMeta const &meta,
                                          int innermost_dim) {
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
  return get_stensor_layout(new_stensor, new_meta);
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

// Transpile a custom KN operator (i.e. a custom block graph) into CUDA code
// Will return a CustomOPTranspileResult object. See comments in transpiler.h
// for more details
CustomOPTranspileResult
    Transpiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  int num_threads = g.block_dim.x * g.block_dim.y * g.block_dim.z;

  // Get the schedule
  TBSched sched = get_threadblock_schedule(g);

  get_threadblock_swizzle_plan(g, sched);

  // Get the memory allocation plan
  TBMemoryPlan mem_plan = get_threadblock_memory_plan(g, sched);

  // Allocate a kernel name
  static int custom_kernel_idx_counter = 0;
  int cur_custom_kernel_idx = custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  // Generate code prologue
  CodeKeeper code;
  code.e("__global__ void __launch_bounds__($) $($, $) {",
         num_threads,
         func_name,
         map<kn::DTensor, string>(op->output_tensors,
                                  [](kn::DTensor const &dtensor) -> string {
                                    return fmt(
                                        "$* __restrict__ dtensor$_ptr",
                                        get_datatype_str(dtensor.data_type),
                                        dtensor.guid);
                                  }),
         map<kn::DTensor, string>(
             op->input_tensors, [](kn::DTensor const &dtensor) -> string {
               return fmt("$ const* __restrict__ dtensor$_ptr",
                          get_datatype_str(dtensor.data_type),
                          dtensor.guid);
             }));

  // Define thread idx
  string thread_idx;
  if (g.block_dim.y > 1 || g.block_dim.z > 1) {
    thread_idx = fmt("threadIdx.x + threadIdx.y * $ + threadIdx.z * $",
                     g.block_dim.x,
                     g.block_dim.x * g.block_dim.y);
  } else {
    thread_idx = "threadIdx.x";
  }
  code.e("int thread_idx = $;", thread_idx);
  code.e("static constexpr int NUM_THREADS = $;", num_threads);

  // Define STensor as cute::Tensor
  code.e("// STensors");
  code.e("extern __shared__ char buf[];");
  for (auto [guid, addr] : mem_plan.addrs) {
    code.e("$ *stensor$_ptr = ($*)(buf + $);",
           get_datatype_str(op->input_tensors[0].data_type),
           guid,
           get_datatype_str(op->input_tensors[0].data_type),
           addr);
  }
  // Erase the lowest 16 bytes to 0 for GEMM
  code.e("*((uint128_t*)buf) = 0ul;");
  code.e("");

  // Define G2SCopy for all input STensors
  code.e("// G->S copy atoms");
  std::unordered_set<tb::TBInputOp const *>
      pipelined_input_ops; // A list of input ops that are software pipelined
                           // (asynchronously G->S copied)
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

      code.e("const $ *dtensor$_tile_ptr = dtensor$_ptr $;",
             get_datatype_str(dtensor.data_type),
             dtensor.guid,
             dtensor.guid,
             offset);

      bool use_chunked_copy = op_meta.is_chunked_input;
      int real_innermost_dim = op_meta.chunked_input_real_innermost_dim;
      bool use_async_copy = op_meta.is_pipelined_input;

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
          // Chunked, asynchronous copy
          pipelined_input_ops.insert(cur_op);
          code.e("using STensor$InputAtom = tb::InputChunkedAsyncCopy<$, "
                 "$, DTensor$TileLayout, NUM_THREADS>;",
                 stensor.guid,
                 get_datatype_str(stensor.data_type),
                 mov_last_get_stensor_layout(
                     stensor, stensor_meta, real_innermost_dim),
                 dtensor.guid);
          code.e("$ *stensor$_async_copy_buf = stensor$_ptr;",
                 get_datatype_str(stensor.data_type),
                 stensor.guid,
                 stensor.guid + mem_plan.pipelined_input_buf_guid_offset);
        }
      }
    }
  }
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
               get_datatype_str(stensor.data_type),
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
               get_datatype_str(stensor.data_type),
               dtensor.guid,
               mov_last_get_stensor_layout(
                   stensor, stensor_meta, real_innermost_dim));
      }
      // TODO(intlsy) Support TMA
    }
  }
  code.e("");

  // Clear all accumulators
  int num_clear_accums = 0;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP &&
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
      string mma_atom_str;
      std::tuple<int, int, int> mma_atom_mnk;
      int mma_atom_num_threads;
      if (GPU_CC::A100 <= config.target_cc && config.target_cc < GPU_CC::H100) {
        if (k <= 8) {
          mma_atom_str = input0.data_type == type::DT_FLOAT16
                             ? "SM80_16x8x8_F16F16F16F16_TN"
                             : "SM80_16x8x8_F32BF16BF16F32_TN";
          mma_atom_mnk = {16, 8, 8};
          mma_atom_num_threads = 32;
        } else {
          mma_atom_str = input0.data_type == type::DT_FLOAT16
                             ? "SM80_16x8x16_F16F16F16F16_TN"
                             : "SM80_16x8x16_F32BF16BF16F32_TN";
          mma_atom_mnk = {16, 8, 16};
          mma_atom_num_threads = 32;
        }
      } else {
        // TODO(intlsy): Support more architectures
        assert(0 && "Unsupported GPU Architecture");
      }
      auto [mma_atom_m, mma_atom_n, mma_atom_k] = mma_atom_mnk;

      // Pick up TiledMMAThrLayout
      // The algorithm is documented in `docs/transpiler/transpiler.md`
      // TODO(intlsy) Update this algo to be more friendly to small matrix
      // by dropping some threads
      assert(num_threads % mma_atom_num_threads == 0);
      int max_num_tgs = num_threads / mma_atom_num_threads; // tg = thread group
      float best_score = -1.0f;
      int best_num_tg_m = 1, best_num_tg_n = 1;
      for (int num_tg_m = 1; num_tg_m <= max_num_tgs; ++num_tg_m) {
        for (int num_tg_n = 1; num_tg_m * num_tg_n <= max_num_tgs; ++num_tg_n) {
          int tiled_mma_m = mma_atom_m * num_tg_m;
          int tiled_mma_n = mma_atom_n * num_tg_n;
          if (((m > mma_atom_m) && (m % tiled_mma_m != 0)) ||
              ((n > mma_atom_n) && (n % tiled_mma_n != 0))) {
            continue;
          }
          int num_tiles_m = ceil_div(m, tiled_mma_m);
          int num_tiles_n = ceil_div(n, tiled_mma_n);
          int64_t data_moved_A =
              ((int64_t)num_tiles_m * tiled_mma_m) * k * num_tg_n;
          int64_t data_moved_B =
              ((int64_t)num_tiles_n * tiled_mma_n) * k * num_tg_m;
          int64_t data_moved = data_moved_A + data_moved_B;
          float score =
              (1.0f / data_moved) * (num_tg_m * num_tg_n / (float)max_num_tgs);
          if (score > best_score) {
            best_score = score;
            best_num_tg_m = num_tg_m;
            best_num_tg_n = num_tg_n;
          }
        }
      }

      bool is_ldmatrix_avail = config.target_cc >= GPU_CC::T4;
      bool is_stmatrix_avail = config.target_cc >= GPU_CC::H100;

      int num_exps_before_store = std::count_if(
          node.ops.begin(), node.ops.end(), [](auto &op_and_meta) {
            return op_and_meta.first->op_type == type::TB_EXP_OP;
          });
      bool is_store_accum =
          node.ops.back().first->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP;
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

      code.e("using Matmul$LayoutAAligned = $;",
             output.guid,
             get_mma_stensor_aligned_layout(input0,
                                            meta0,
                                            mma_atom_mnk,
                                            true,
                                            false,
                                            num_dims - 2 /*start_dim*/));

      code.e("using Matmul$LayoutBAligned = $;",
             output.guid,
             get_mma_stensor_aligned_layout(input1,
                                            meta1,
                                            mma_atom_mnk,
                                            false,
                                            false,
                                            num_dims - 2 /*start_dim*/));

      code.e("using Matmul$Kernel = tb::Matmul<$, $, Layout<Shape<Int<$>, "
             "Int<$>, _1>>, $, $, Matmul$LayoutA, Matmul$LayoutB, "
             "Matmul$LayoutC, Matmul$LayoutAAligned, Matmul$LayoutBAligned,"
             "NUM_THREADS, "
             "$, $>;",
             output.guid,
             get_datatype_str(input0.data_type),
             mma_atom_str,
             best_num_tg_m,
             best_num_tg_n,
             is_ldmatrix_avail,
             is_stmatrix_avail,
             output.guid,
             output.guid,
             output.guid,
             output.guid,
             output.guid,
             num_exps_before_store,
             is_accum_in_reg ? false : is_store_accum);
      // Allocate accumulators in register files (if needed)
      if (is_accum_in_reg) {
        code.e("auto matmul_$_accum = Matmul$Kernel::get_mma_rC(thread_idx);",
               output.guid,
               output.guid);
      }
      code.e("");
    }
  }

  if (num_pre_loop_copies > 0 || num_clear_accums > 0) {
    code.e("__syncthreads();");
    code.e("");
  }

  // Launch async input operations for all async inputs
  if (!pipelined_input_ops.empty()) {
    code.e("{");
    for (tb::TBInputOp const *input_op : pipelined_input_ops) {
      kn::DTensor const &dtensor = input_op->dtensor;
      tb::STensor const &output = input_op->output_tensors.at(0);
      assert(input_op->forloop_dim >= 0);
      code.e("STensor$InputAtom::run(stensor$_async_copy_buf, "
             "dtensor$_tile_ptr, thread_idx);",
             output.guid,
             output.guid,
             dtensor.guid);
    }
    code.e("cute::cp_async_fence();");
    code.e("}");
    code.e("");
  }

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
                                     bool is_in_loop) {
    if (sched_node.type == tb_sched_node_t::SYNCTHREADS) {
      code.e("__syncthreads();");
    } else {
      auto [op, first_op_meta] = sched_node.ops.front();
      auto [output_op, output_op_meta] = sched_node.ops.back();
      assert(output_op == fusion_chain.at(op).back());
      std::string op_type_str;
      to_json(op_type_str, op->op_type);
      code.e("{");
      code.e("// OP type: $", op_type_str);
      switch (op->op_type) {
        case type::TB_INPUT_OP: {
          // In this lambda function we only accept input ops within the for
          // loop
          assert(sched_node.ops.size() == 1); // Should not be fused
          tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(op);
          assert(is_in_loop);
          assert(cur_op->forloop_dim >= 0);
          kn::DTensor const &dtensor = cur_op->dtensor;
          tb::STensor const &output = cur_op->output_tensors.at(0);
          int tile_side_len = output.dim[cur_op->forloop_dim];
          size_t forloop_dim_stride =
              dtensor_metas.at(dtensor.guid).strides[cur_op->forloop_dim];
          bool is_async_copy = pipelined_input_ops.count(cur_op);
          assert(!is_async_copy); // Async copies should be proceeded separately
          code.e("STensor$InputAtom::run(stensor$_ptr, dtensor$_tile_ptr + "
                 "$*for_idx, thread_idx);",
                 output.guid,
                 output.guid,
                 dtensor.guid,
                 tile_side_len * forloop_dim_stride);
          break;
        }
        case type::TB_OUTPUT_OP: {
          assert(sched_node.ops.size() == 1); // Should not be fused
          tb::TBOutputOp const *cur_op =
              dynamic_cast<tb::TBOutputOp const *>(op);
          // Currently in Mirage core, an output op must have forloop_dim = -1
          assert(!is_in_loop);
          assert(cur_op->forloop_dim == -1);
          if (cur_op->forloop_dim >= 0) {
            assert(0);
#ifdef DEADCODE
            // For output DTensor that has a forloop_dim, copy it
            kn::DTensor const &dtensor = cur_op->dtensor;
            tb::STensor const &stensor = cur_op->input_tensors.at(0);
            int tile_side_len = stensor.dim[cur_op->forloop_dim];
            size_t forloop_dim_stride =
                dtensor_metas.at(dtensor.guid).strides[cur_op->forloop_dim];
            code.e("STensor$OutputAtom::run(stensor$_ptr, dtensor$_tile_ptr + "
                   "$*for_idx, thread_idx);",
                   stensor.guid,
                   stensor.guid,
                   dtensor.guid,
                   tile_side_len * forloop_dim_stride);
#endif
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
          if (output_op_meta.is_accum_in_reg) {
            // Accumulator is in register
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
          break;
        }
        case type::TB_EXP_OP:
        case type::TB_SQUARE_OP:
        case type::TB_SQRT_OP:
        case type::TB_SILU_OP:
        case type::TB_GELU_OP:
        case type::TB_RELU_OP:
        case type::TB_CLAMP_OP:
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
#ifdef DEADCODE
          if (iter_dim == -1) {
            // We cannot find a dim that satisfies our assumption:
            // dim i in input&output tensor == meta.innermost_dim or
            // meta.swizzled_dim
            // We return a CUDA_T_LAYOUT_ERROR
            return CUDA_T_LAYOUT_ERROR;
          }
#endif
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
                 "NUM_THREADS, $>;",
                 get_datatype_str(input.data_type),
                 get_tb_op_str(cur_op->op_type),
                 epilogue);
          // add scalar chains for epilogue
          // code.e("const float scalars[] = {A, B, C}");
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
        case type::TB_DIV_OP: {
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
                 "NUM_THREADS, $>;",
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
                 "OutLayout, InLayout, $, NUM_THREADS, $>;",
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
      code.e("}");
    }
    return CUDA_T_SUCCESS;
  };

  // Declare the for loop
  // TODO(intlsy) Remove the loop when `g.forloop_range` is 1
  // TODO(intlsy) Loop unrolling
  assert(g.forloop_range >= 1);
  code.e("// The main loop");
  code.e("for (int for_idx = 0; for_idx < $; for_idx++) {", g.forloop_range);

  if (!pipelined_input_ops.empty()) {
    code.e("{");
    code.e("// Issue async copies for the next round");
    code.e("if (for_idx+1 != $) {", g.forloop_range);
    for (tb::TBInputOp const *input_op : pipelined_input_ops) {
      assert(input_op->forloop_dim >= 0);
      kn::DTensor const &dtensor = input_op->dtensor;
      tb::STensor const &output = input_op->output_tensors.at(0);
      int tile_side_len = output.dim[input_op->forloop_dim];
      size_t forloop_dim_stride =
          dtensor_metas.at(dtensor.guid).strides[input_op->forloop_dim];
      code.e("STensor$InputAtom::run(stensor$_ptr, dtensor$_tile_ptr + "
             "$*(for_idx+1), thread_idx);",
             output.guid,
             output.guid,
             dtensor.guid,
             tile_side_len * forloop_dim_stride);
    }
    code.e("}");
    code.e("cute::cp_async_fence();");

    code.e("// Wait for the async copies in the last round to finish");
    code.e("cute::cp_async_wait<1>();");

    code.e("// Switch buffers");
    for (tb::TBInputOp const *input_op : pipelined_input_ops) {
      tb::STensor const &output = input_op->output_tensors.at(0);
      sguid_t guid = output.guid;
      code.e("SWAP(stensor$_ptr, stensor$_async_copy_buf);", guid, guid);
    }

    code.e("}");
  }

  for (TBSchedNode const &sched_node : sched.loop_nodes) {
    if (sched_node.type == tb_sched_node_t::OPERATOR &&
        sched_node.ops[0].first->op_type == type::TB_INPUT_OP &&
        pipelined_input_ops.count(
            dynamic_cast<tb::TBInputOp const *>(sched_node.ops[0].first))) {
      continue;
    }
    CodeKeeper res;
    TranspileErrorType err = transpile_tb_sched_node(sched_node, res, true);
    code << res;
    if (err != CUDA_T_SUCCESS) {
      return CustomOPTranspileResult{err, func_name, 0, ""};
    }
  }

  code.e("}"); // For loop
  code.e("");

  // Write back in-register accumulators
  int num_in_reg_accums = 0;
  CodeKeeper in_reg_writeback;
  for (TBSchedNode const &node : sched.loop_nodes) {
    if (node.type != tb_sched_node_t::OPERATOR) {
      continue;
    }
    auto [last_op, last_op_meta] = node.ops.back();
    if (last_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP &&
        last_op_meta.is_accum_in_reg) {
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
    code.e("__syncthreads();"); // Need this __syncthreads() to make sure no
                                // thread is still in the for loop
    code << in_reg_writeback;
  }

  // Transpile the epilogue of the kernel
  if (!sched.post_loop_nodes.empty()) {
    code.e("// The epilogue (kernels outside the loop)");
    code.e("__syncthreads();");
    for (TBSchedNode const &sched_node : sched.post_loop_nodes) {
      CodeKeeper res;
      TranspileErrorType err = transpile_tb_sched_node(sched_node, res, false);
      code << res;
      if (err != CUDA_T_SUCCESS) {
        return CustomOPTranspileResult{err, func_name, 0, ""};
      }
    }
  }

  code.e("}"); // kernel

  return CustomOPTranspileResult{
      CUDA_T_SUCCESS, func_name, mem_plan.smem_size, code.to_string()};
}

} // namespace transpiler
} // namespace mirage
