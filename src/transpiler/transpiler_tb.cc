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

#include "mirage/threadblock/operator.h"
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
static string get_cute_layout(Tensor_T const &tensor, Meta_T const &meta) {
  return get_cute_layout(
      vector<int>(tensor.dim, tensor.dim + tensor.num_dims),
      vector<size_t>(meta.strides, meta.strides + tensor.num_dims));
}

// A helper function for mov_inner_dim_and_get_layout
template <typename T>
static std::vector<T> mov_to_last(T const *vec, size_t numel, int idx) {
  std::vector<T> result;
  result.reserve(numel);
  result.insert(result.end(), vec, vec + idx);
  result.insert(result.end(), vec + idx + 1, vec + numel);
  result.push_back(vec[idx]);
  return result;
}

// Move the innermost dim to the last dim, and format it as a CuTe layout
// string.
//
// Assume the tensor has N dimensions and the innermost dim is i, then the
// function is equivalent to torch.permute(tensor, [0, 1, ..., i-1, i+1, ..., N,
// i])
//
// This function is helpful for element-wise ops, since the
// processing order of elements do not affect the correctness.
template <typename Tensor_T, typename Meta_T>
static string mov_last_and_get_layout(Tensor_T const &tensor,
                                      Meta_T const &meta,
                                      int innermost_dim) {
  assert(0 <= innermost_dim && innermost_dim < tensor.num_dims);
  return get_cute_layout(
      mov_to_last(tensor.dim, tensor.num_dims, innermost_dim),
      mov_to_last(meta.strides, tensor.num_dims, innermost_dim));
}

// Test whether consecutive `chunk_size` elements in layout A are contiguous in layout B
// 
// See docs/transpiler/transpiler.md for more details
static std::pair<bool, int> can_perform_chunked_copy(
  int num_dims,
  const int shape[],
  const size_t stride_dtensor[],
  const size_t stride_stensor[],
  size_t dtype_size
) {
  // Check whether the strides are 16B-aligned
  auto is_strides_aligned_16B = [&](const size_t strides[]) -> bool {
    size_t alignment = 16 / dtype_size;
    bool res = true;
    for (int i = 0; i < num_dims; ++i) {
      size_t stride = strides[i];
      res &= (stride % alignment == 0 || stride == 1);
    }
    return res;
  };
  if(!is_strides_aligned_16B(stride_dtensor)) {
    return {false, 0};
  }
  assert(is_strides_aligned_16B(stride_stensor)); // In our current design, the layout of STensor is always 16B-aligned

  // Check whether the "real innermost dim" is the same
  auto find_real_innermost_dim = [&](const size_t strides[]) -> int {
    for (int i = 0; i < num_dims; ++i) {
      if (strides[i] == 1 && shape[i] != 1) {
        return i;
      }
    }
    return -1;
  };
  int real_innermost_dtensor = find_real_innermost_dim(stride_dtensor);
  int real_innermost_stensor = find_real_innermost_dim(stride_stensor);
  // assert(real_innermost_dtensor != -1);  real_innermost_dtensor can be -1 for input tensors
  assert(real_innermost_stensor != -1);
  return {real_innermost_dtensor == real_innermost_stensor, real_innermost_stensor};
}


// Transpile a custom KN operator (i.e. a custom block graph) into CUDA code
// Will return a CustomOPTranspileResult object. See comments in transpiler.h
// for more details
CustomOPTranspileResult
    Transpiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  tb::ExecutionPlan const &plan = op->plan;
  int num_threads = plan.block_dim.x * plan.block_dim.y * plan.block_dim.z;

  // Get the schedule
  TBSched sched = get_threadblock_schedule(g);

  // Get the memory allocation plan
  TBMemoryPlan mem_plan = get_threadblock_memory_plan(g, sched);
  size_t cur_smem_size = mem_plan.smem_size;  // May increase, e.g. when we allocate buffers for async copy
  auto allocate_buf = [&](size_t size) {
    cur_smem_size = round_to_multiple(cur_smem_size, (size_t)16);
    size_t offset = cur_smem_size;
    cur_smem_size += size;
    return offset;
  };

  // Allocate a kernel name
  static int custom_kernel_idx_counter = 0;
  int cur_custom_kernel_idx = custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  // Generate code prologue
  CodeKeeper code;
  code.e(
      "__global__ void __launch_bounds__($) $($, $) {",
      num_threads,
      func_name,
      map<kn::DTensor, string>(op->output_tensors,
                               [](kn::DTensor const &dtensor) -> string {
                                 return fmt("half_t* __restrict__ dtensor$_ptr",
                                            dtensor.guid);
                               }),
      map<kn::DTensor, string>(
          op->input_tensors, [](kn::DTensor const &dtensor) -> string {
            return fmt("half_t const* __restrict__ dtensor$_ptr", dtensor.guid);
          }));

  // Define thread idx
  string thread_idx;
  if (plan.block_dim.y > 1 || plan.block_dim.z > 1) {
    thread_idx = fmt("threadIdx.x + threadIdx.y * $ + threadIdx.z * $",
                     plan.block_dim.x,
                     plan.block_dim.x * plan.block_dim.y);
  } else {
    thread_idx = "threadIdx.x";
  }
  code.e("int thread_idx = $;", thread_idx);
  code.e("static constexpr int NUM_THREADS = $;", num_threads);

  // Define STensor as cute::Tensor
  code.e("// STensors");
  code.e("extern __shared__ char buf[];");
  for (auto [guid, addr] : mem_plan.addrs) {
    code.e("half_t *stensor$_ptr = (half_t*)(buf + $);", guid, addr);
  }
  // Erase the lowest 16 bytes to 0 for GEMM
  code.e("*((uint128_t*)buf) = 0ul;");
  code.e("");

  // Define G2SCopy for all input STensors
  code.e("// G->S copy atoms");
  std::unordered_set<tb::TBInputOp const*> async_copy_input_ops;  // A list of input ops that are asynchronously G->S copied
  for (tb::TBOperator const *op : g.operators) {
    if (op->op_type == type::TB_INPUT_OP) {
      tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(op);
      tb::TBOperator const *output_op = fusion_chain.at(op).back();
      kn::DTensor const &dtensor = cur_op->dtensor;
      tb::STensor const &stensor = output_op->output_tensors.at(0);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
      assert(dtensor.num_dims == stensor.num_dims);
      assert(dtensor.data_type == stensor.data_type);
      size_t alignment = get_num_elems_in_16B(dtensor.data_type);

      code.e("// Copy for G->S: dtensor $ -> stensor $",
             dtensor.guid,
             stensor.guid);

      // Get the starting address of my tile
      // For input tensor that does not have a forloop_dim, the shape of the
      // tile should be identical to the STensor. Otherwise, it should be the
      // shape of STensor * forloop_range
      string offset = "";
      int3 imap = cur_op->input_map;
      bool is_dtensor_offset_divisible = true;
      for (int dim = 0; dim < 3; ++dim) {
        int div_dim = dim == 0 ? imap.x : dim == 1 ? imap.y : imap.z;
        if (div_dim >= 0) {
          // Dim `div_dim` is divided along `dim`
          int num_tbs = dim == 0   ? plan.grid_dim.x
                        : dim == 1 ? plan.grid_dim.y
                                   : plan.grid_dim.z;
          offset += fmt(" + blockIdx.$*$*$",
                        (char)"xyz"[dim],
                        dtensor.dim[div_dim] / num_tbs,
                        dtensor_meta.strides[div_dim]);
          is_dtensor_offset_divisible &= num_tbs == 1 || (dtensor.dim[div_dim] / num_tbs) % alignment == 0 || dtensor_meta.strides[div_dim] % alignment == 0;
        }
      }
      if (cur_op->forloop_dim >= 0) {
        int forloop_dim = cur_op->forloop_dim;
        int forloop_range = dtensor.dim[forloop_dim];
        size_t forloop_dim_stride = dtensor_meta.strides[forloop_dim];
        int tile_side_len = stensor.dim[forloop_dim];
        is_dtensor_offset_divisible &= forloop_range == 1 || tile_side_len % alignment == 0 || forloop_dim_stride % alignment == 0;
      }

      code.e("const half_t *dtensor$_tile_ptr = dtensor$_ptr $;",
             dtensor.guid,
             dtensor.guid,
             offset);

      auto [use_chunked_copy, real_innermost_dim] = can_perform_chunked_copy(
          stensor.num_dims,
          stensor.dim,
          dtensor_meta.strides,
          stensor_meta.strides,
          type::get_datatype_size(dtensor.data_type)
      );
      use_chunked_copy &= is_dtensor_offset_divisible;
      bool use_async_copy =
          use_chunked_copy && this->config.target_cc >= GPU_CC::A100 &&
          cur_op->forloop_dim != -1;  // Only use async copy when the input tensor has a forloop_dim

      // TODO(intlsy) Support swizzled layout
      // TODO(intlsy) Support TMA
      if (!use_chunked_copy) {
        int d_innermost_dim = dtensor_meta.innermost_dim;
        assert(!use_async_copy);
        string dtensor_tile_layout = get_cute_layout(
            mov_to_last(stensor.dim,
                        dtensor.num_dims,
                        d_innermost_dim), // Here we use stensor.dim
            mov_to_last(dtensor_meta.strides, dtensor.num_dims, d_innermost_dim));
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        // Non-chunked, synchronous copy
        code.e("using STensor$InputAtom = tb::InputNonChunkedSyncCopy<half_t, "
             "$, DTensor$TileLayout, NUM_THREADS>;",
             stensor.guid,
             mov_last_and_get_layout(stensor, stensor_meta, d_innermost_dim),
             dtensor.guid);
      } else {
        string dtensor_tile_layout = get_cute_layout(
            mov_to_last(stensor.dim,
                        dtensor.num_dims,
                        real_innermost_dim), // Here we use stensor.dim
            mov_to_last(dtensor_meta.strides, dtensor.num_dims, real_innermost_dim));
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        if (!use_async_copy) {
          // Chunked, synchronous copy
          code.e("using STensor$InputAtom = tb::InputChunkedSyncCopy<half_t, "
             "$, DTensor$TileLayout, NUM_THREADS>;",
             stensor.guid,
             mov_last_and_get_layout(stensor, stensor_meta, real_innermost_dim),
             dtensor.guid);
        } else {
          // Chunked, asynchronous copy
          async_copy_input_ops.insert(cur_op);
          code.e("using STensor$InputAtom = tb::InputChunkedAsyncCopy<half_t, "
             "$, DTensor$TileLayout, NUM_THREADS>;",
             stensor.guid,
             mov_last_and_get_layout(stensor, stensor_meta, real_innermost_dim),
             dtensor.guid);
          // Allocate a buffer for the async copy since we are going to pipeline it
          size_t async_copy_buf_size = stensor_meta.num_phy_elems * type::get_datatype_size(stensor.data_type);
          size_t async_copy_buf_addr = allocate_buf(async_copy_buf_size);
          code.e("half_t *stensor$_async_copy_buf = (half_t*)(buf + $);",
                 stensor.guid,
                 async_copy_buf_addr);
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
    tb::TBOperator const *op = sched_node.ops[0];
    assert(op->op_type == type::TB_INPUT_OP);
    tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(op);
    tb::STensor const& stensor = cur_op->output_tensors.at(0);
    assert(cur_op->forloop_dim == -1);
    assert(!async_copy_input_ops.count(cur_op)); // An input op in pre_loop_nodes should not be asynchronously copied since they do not have forloop_dim
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
  for (tb::TBOperator const *op : g.operators) {
    if (op->op_type == type::TB_OUTPUT_OP) {
      tb::TBOutputOp const *cur_op = dynamic_cast<tb::TBOutputOp const *>(op);
      tb::STensor const &stensor = cur_op->input_tensors.at(0);
      kn::DTensor const &dtensor = cur_op->dtensor;
      STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      assert(dtensor.num_dims == stensor.num_dims);
      assert(dtensor.data_type == stensor.data_type);
      size_t alignment = get_num_elems_in_16B(dtensor.data_type);

      code.e("// Copy for S->G: stensor $ -> dtensor $",
             stensor.guid,
             dtensor.guid);

      // Get the starting address of my tile
      // For output tensor that does not have a forloop_dim, the shape of the
      // tile should be identical to the STensor. Otherwise, it should be the
      // shape of STensor * forloop_range
      string offset = "";
      int3 omap = cur_op->output_map;
      bool is_dtensor_offset_divisible = true;
      for (int dim = 0; dim < 3; ++dim) {
        int div_dim = dim == 0 ? omap.x : dim == 1 ? omap.y : omap.z;
        int num_tbs = dim == 0   ? plan.grid_dim.x
                      : dim == 1 ? plan.grid_dim.y
                                 : plan.grid_dim.z;
        if (num_tbs > 1) {
          // The output tensor MUST be divided along this dimension, as stated
          // in the paper
          assert(div_dim >= 0);
          offset += fmt(" + blockIdx.$*$*$",
                        (char)"xyz"[dim],
                        dtensor.dim[div_dim] / num_tbs,
                        dtensor_meta.strides[div_dim]);
          is_dtensor_offset_divisible &= (dtensor.dim[div_dim] / num_tbs) % alignment == 0 || dtensor_meta.strides[div_dim] % alignment == 0;
        }
      }
      code.e("half_t *dtensor$_tile_ptr = dtensor$_ptr $;",
             dtensor.guid,
             dtensor.guid,
             offset);

      auto [use_chunked_copy, real_innermost_dim] = can_perform_chunked_copy(
          stensor.num_dims,
          stensor.dim,
          dtensor_meta.strides,
          stensor_meta.strides,
          type::get_datatype_size(dtensor.data_type)
      );
      use_chunked_copy &= is_dtensor_offset_divisible;

      if (!use_chunked_copy) {
        int d_innermost_dim = dtensor_meta.innermost_dim;
        string dtensor_tile_layout = get_cute_layout(
            mov_to_last(stensor.dim,
                        dtensor.num_dims,
                        d_innermost_dim), // Here we use stensor.dim
            mov_to_last(dtensor_meta.strides, dtensor.num_dims, d_innermost_dim));
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        code.e("using STensor$OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, "
              "DTensor$TileLayout, $, NUM_THREADS>;",
              stensor.guid,
              dtensor.guid,
              mov_last_and_get_layout(stensor, stensor_meta, d_innermost_dim));
      } else {
        string dtensor_tile_layout = get_cute_layout(
            mov_to_last(stensor.dim,
                        dtensor.num_dims,
                        real_innermost_dim), // Here we use stensor.dim
            mov_to_last(dtensor_meta.strides, dtensor.num_dims, real_innermost_dim));
        code.e(
            "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);
        code.e("using STensor$OutputAtom = tb::OutputChunkedSyncCopy<half_t, "
              "DTensor$TileLayout, $, NUM_THREADS>;",
              stensor.guid,
              dtensor.guid,
              mov_last_and_get_layout(stensor, stensor_meta, real_innermost_dim));
      }
      // TODO(intlsy) Support TMA
    }
  }
  code.e("");

  // Clear all accumulators
  int num_clear_accums = 0;
  for (tb::TBOperator const *op : g.operators) {
    if (op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      tb::STensor const &accum = op->output_tensors.at(0);
      STensorMeta const &accum_meta = stensor_metas.at(accum.guid);
      size_t num_elems = 0;
      for (int i = 0; i < accum.num_dims; ++i) {
        num_elems = std::max(num_elems, accum.dim[i] * accum_meta.strides[i]);
      }
      code.e("tb::ClearAccumlatorKernel<half_t, $, "
             "NUM_THREADS>::run(stensor$_ptr, thread_idx);",
             num_elems,
             accum.guid);
      num_clear_accums += 1;
    }
  }
  code.e("");

  if (num_pre_loop_copies > 0 || num_clear_accums > 0) {
    code.e("__syncthreads();");
    code.e("");
  }

  // A lambda function that transpiles a chain of (fusable) operators to an
  // epilogue Will automatically ignore the first operator in the `chain`
  // argument
  auto transpile_fusion_epilogue =
      [&](std::vector<tb::TBOperator const *> const &chain) -> string {
    size_t chain_size = chain.size();
    if (chain_size == 1) {
      // Not fused with anything
      return "tb::EpilogueStore<half_t>";
    }
    // Deal with the last operator
    string res = "tb::EpilogueStore<half_t>";
    for (size_t i = chain_size - 1; i >= 1; --i) {
      tb::TBOperator const *cur_op = chain[i];
      if (cur_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
        // Can only occur as the last operator in the chain
        assert(i == chain_size - 1);
        res = "tb::EpilogueStoreAccum<half_t>";
      } else if (cur_op->op_type == type::TB_EXP_OP) {
        res = fmt("tb::EpilogueExp<half_t, $>", res);
      } else {
        assert(0 && "Unknown operator type");
      }
    }
    return res;
  };

  // A lambda function that transpiles an TBSchedNode
  auto transpile_tb_sched_node = [&](TBSchedNode const &sched_node,
                                     bool is_in_loop) {
    CodeKeeper code;
    if (sched_node.type == tb_sched_node_t::SYNCTHREADS) {
      code.e("__syncthreads();");
    } else {
      tb::TBOperator const *op = sched_node.ops[0];
      tb::TBOperator const *output_op = fusion_chain.at(op).back();
      assert(output_op == sched_node.ops.back());
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
          bool is_async_copy = async_copy_input_ops.count(cur_op);
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
        };
        case type::TB_MATMUL_OP: {
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
          assert(input0.dim[num_dims - 2] == m &&
                 input0.dim[num_dims - 1] == k);
          assert(input1.dim[num_dims - 2] == k &&
                 input1.dim[num_dims - 1] == n);

          // Pick up MMA atom
          // TODO(intlsy) May calculate AB via (B^T A^T)^T when M is relatively
          // small
          string mma_atom_str;
          std::tuple<int, int, int> mma_atom_mnk;
          int mma_atom_num_threads;
          if (GPU_CC::A100 <= config.target_cc &&
              config.target_cc < GPU_CC::H100) {
            if (k <= 8) {
              mma_atom_str = "SM80_16x8x8_F16F16F16F16_TN";
              mma_atom_mnk = {16, 8, 8};
              mma_atom_num_threads = 32;
            } else {
              mma_atom_str = "SM80_16x8x16_F16F16F16F16_TN";
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
          int max_num_tgs =
              num_threads / mma_atom_num_threads; // tg = thread group
          float best_score = -1.0f;
          int best_num_tg_m = -1, best_num_tg_n = -1;
          for (int num_tg_m = 1; num_tg_m <= max_num_tgs; ++num_tg_m) {
            for (int num_tg_n = 1; num_tg_m * num_tg_n <= max_num_tgs;
                 ++num_tg_n) {
              int tiled_mma_m = mma_atom_m * num_tg_m;
              int tiled_mma_n = mma_atom_n * num_tg_n;
              int num_tiles_m = ceil_div(m, tiled_mma_m);
              int num_tiles_n = ceil_div(n, tiled_mma_n);
              int64_t data_moved_A =
                  ((int64_t)num_tiles_m * tiled_mma_m) * k * num_tg_n;
              int64_t data_moved_B =
                  ((int64_t)num_tiles_n * tiled_mma_n) * k * num_tg_m;
              int64_t data_moved = data_moved_A + data_moved_B;
              float score = (1.0f / data_moved) *
                            (num_tg_m * num_tg_n / (float)max_num_tgs);
              if (score > best_score) {
                best_score = score;
                best_num_tg_m = num_tg_m;
                best_num_tg_n = num_tg_n;
              }
            }
          }

          bool is_ldmatrix_avail = config.target_cc >= GPU_CC::T4;
          bool is_stmatrix_avail = config.target_cc >= GPU_CC::H100;

          int num_exps_before_store =
              std::count_if(sched_node.ops.begin(),
                            sched_node.ops.end(),
                            [](tb::TBOperator const *op) {
                              return op->op_type == type::TB_EXP_OP;
                            });
          bool is_store_accum =
              sched_node.ops.back()->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP;

          code.e("using LayoutA = $;", get_cute_layout(input0, meta0));
          code.e("using LayoutB = $;", get_cute_layout(input1, meta1));
          code.e("using LayoutC = $;", get_cute_layout(output, meta2));

          code.e("using Kernel = tb::Matmul<half_t, $, Layout<Shape<Int<$>, "
                 "Int<$>, _1>>, $, $, LayoutA, LayoutB, LayoutC, NUM_THREADS, "
                 "$, $>;",
                 mma_atom_str,
                 best_num_tg_m,
                 best_num_tg_n,
                 is_ldmatrix_avail,
                 is_stmatrix_avail,
                 num_exps_before_store,
                 is_store_accum);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, stensor$_ptr, "
                 "(char*)(buf+0), thread_idx);",
                 output.guid,
                 input0.guid,
                 input1.guid);
          break;
        };
        case type::TB_EXP_OP: {
          tb::STensor const &input = op->input_tensors.at(0);
          tb::STensor const &output = output_op->output_tensors.at(0);
          assert(input.num_dims == output.num_dims);
          int num_dims = input.num_dims;
          // Find the iteration dim
          int iter_dim = -1;
          for (int i = 0; i < num_dims; ++i) {
            bool failed = false;
            for (tb::STensor const &stensor : {input, output}) {
              STensorMeta meta = stensor_metas.at(stensor.guid);
              if (i != meta.innermost_dim && !meta.is_dim_swizzled(i)) {
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
          string in_layout = mov_last_and_get_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string final_out_layout = mov_last_and_get_layout(
              output, stensor_metas.at(output.guid), iter_dim);
          code.e("using InLayout = $;", in_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(sched_node.ops);
          // Define and run the kernel
          code.e("using Kernel = tb::ElementUnaryKernel<half_t, "
                 "tb::ElementUnaryOpType::EXP, OutLayout, InLayout, "
                 "NUM_THREADS, $>;",
                 epilogue);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx);",
                 output.guid,
                 input.guid);
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
              if (i != meta.innermost_dim && !meta.is_dim_swizzled(i)) {
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
          string in0_layout = mov_last_and_get_layout(
              input0, stensor_metas.at(input0.guid), iter_dim);
          string in1_layout = mov_last_and_get_layout(
              input1, stensor_metas.at(input1.guid), iter_dim);
          string final_out_layout = mov_last_and_get_layout(
              output, stensor_metas.at(output.guid), iter_dim);
          code.e("using In0Layout = $;", in0_layout);
          code.e("using In1Layout = $;", in1_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(sched_node.ops);
          // Define and run the kernel
          code.e("using Kernel = tb::ElementBinaryKernel<half_t, "
                 "tb::ElementBinaryOpType::$, OutLayout, In0Layout, In1Layout, "
                 "NUM_THREADS, $>;",
                 op_type_str,
                 epilogue);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, stensor$_ptr, "
                 "thread_idx);",
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
          STensorMeta orig_output_meta = stensor_metas.at(output.guid);
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
              if (i != meta.innermost_dim && !meta.is_dim_swizzled(i)) {
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
              mov_last_and_get_layout(input, input_meta, iter_dim);
          string final_out_layout =
              mov_last_and_get_layout(output, final_output_meta, iter_dim);
          int cute_reduc_dim = reduc_dim < iter_dim ? num_dims - 1 - reduc_dim
                                                    : num_dims - reduc_dim;
          code.e("using InLayout = $;", in_layout);
          code.e("using OutLayout = $;", final_out_layout);
          // Get the epilogue
          string epilogue = transpile_fusion_epilogue(sched_node.ops);
          // Define and run the kernel
          code.e("using Kernel = tb::ReductionKernel<half_t, "
                 "OutLayout, InLayout, $, NUM_THREADS, $>;",
                 cute_reduc_dim,
                 epilogue);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx);",
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
              if (i != meta.innermost_dim && !meta.is_dim_swizzled(i)) {
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
          string in_layout = mov_last_and_get_layout(
              input, stensor_metas.at(input.guid), iter_dim);
          string accum_layout = mov_last_and_get_layout(
              accum, stensor_metas.at(accum.guid), iter_dim);
          code.e("using Kernel = tb::ForloopAccumKernel<half_t, $, $, "
                 "NUM_THREADS>;",
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
    return code;
  };

  // Launch async input operations for all async inputs
  if (!async_copy_input_ops.empty()) {
    code.e("{");
    for (tb::TBInputOp const* input_op : async_copy_input_ops) {
      kn::DTensor const &dtensor = input_op->dtensor;
      tb::STensor const &output = input_op->output_tensors.at(0);
      assert(input_op->forloop_dim >= 0);
      code.e("STensor$InputAtom::run(stensor$_async_copy_buf, dtensor$_tile_ptr, thread_idx);",
                 output.guid,
                 output.guid,
                 dtensor.guid);
    }
    code.e("cute::cp_async_fence();");
    code.e("}");
  }

  // Declare the for loop
  // TODO(intlsy) Remove the loop when `plan.forloop_range` is 1
  // TODO(intlsy) Loop unrolling
  assert(plan.forloop_range >= 1);
  code.e("// The main loop");
  code.e("for (int for_idx = 0; for_idx < $; for_idx++) {", plan.forloop_range);

  if (!async_copy_input_ops.empty()) {
    code.e("{");
    code.e("// Issue async copies for the next round");
    code.e("if (for_idx+1 != $) {", plan.forloop_range);
    for (tb::TBInputOp const* input_op : async_copy_input_ops) {
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
    for (tb::TBInputOp const* input_op : async_copy_input_ops) {
      tb::STensor const &output = input_op->output_tensors.at(0);
      sguid_t guid = output.guid;
      code.e("SWAP(stensor$_ptr, stensor$_async_copy_buf);", guid, guid);
    }

    code.e("}");
  }

  for (TBSchedNode const &sched_node : sched.loop_nodes) {
    if (sched_node.type == tb_sched_node_t::OPERATOR &&
        sched_node.ops[0]->op_type == type::TB_INPUT_OP &&
        async_copy_input_ops.count(dynamic_cast<tb::TBInputOp const *>(sched_node.ops[0]))) {
      continue;
    }
    CodeKeeper res = transpile_tb_sched_node(sched_node, true);
    code << res;
  }
  
  code.e("}"); // For loop
  code.e("");

  if (!sched.post_loop_nodes.empty()) {
    code.e("__syncthreads();");
    code.e("// The epilogue (kernels outside the loop)");
    for (TBSchedNode const &sched_node : sched.post_loop_nodes) {
      CodeKeeper res = transpile_tb_sched_node(sched_node, false);
      code << res;
    }
  }

  code.e("}"); // kernel

  return CustomOPTranspileResult{
      func_name, cur_smem_size, code.to_string()};
}

} // namespace transpiler
} // namespace mirage