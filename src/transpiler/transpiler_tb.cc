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

#include "mirage/transpiler/transpiler.h"

#include <algorithm>
#include <unordered_set>

#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/utils.h"

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

// The following code are related to threadblock graph transpilation

// Transpile a custom KN operator (i.e. a custom block graph) into CUDA code
// Will return a CustomOPTranspileResult object. See comments in transpiler.h
// for more details
Transpiler::CustomOPTranspileResult
    Transpiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  tb::ExecutionPlan const &plan = op->plan;
  int num_threads = plan.block_dim.x * plan.block_dim.y * plan.block_dim.z;

  // Get a list of all STensors in the current kernel
  vector<tb::STensor> all_stensors;
  std::unordered_set<sguid_t> processed_sguids;
  for (tb::TBOperator *const op : g.operators) {
    for (tb::STensor const &stensor :
         Combine(op->input_tensors, op->output_tensors)) {
      if (processed_sguids.count(stensor.guid) == 0) {
        processed_sguids.insert(stensor.guid);
        all_stensors.push_back(stensor);
      }
    }
  }

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
  for (tb::STensor const &stensor : all_stensors) {
    sguid_t guid = stensor.guid;
    STensorMeta const &meta = stensor_metas.at(guid);
    code.e("half_t *stensor$_ptr = (half_t*)(buf + $);", guid, meta.addr);
  }
  code.e("");

  // Define G2SCopy for all input STensors, and copy STensors that do not have
  // forloop_dim
  code.e("// G->S copy atoms");
  for (tb::TBOperator const *op : g.operators) {
    if (op->op_type == type::TB_INPUT_OP) {
      tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(op);
      kn::DTensor const &dtensor = cur_op->dtensor;
      tb::STensor const &stensor = cur_op->output_tensors.at(0);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      STensorMeta const &stensor_meta = stensor_metas.at(stensor.guid);
      assert(dtensor.num_dims == stensor.num_dims);
      assert(dtensor.data_type == stensor.data_type);
      int num_dims = dtensor.num_dims;
      int d_innermost_dim = dtensor_meta.innermost_dim;

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
          int num_tbs = dim == 0   ? plan.grid_dim.x
                        : dim == 1 ? plan.grid_dim.y
                                   : plan.grid_dim.z;
          offset += fmt(" + blockIdx.$*$*$",
                        (char)"xyz"[dim],
                        dtensor.dim[div_dim] / num_tbs,
                        dtensor_meta.strides[div_dim]);
        }
      }
      code.e("const half_t *dtensor$_tile_ptr = dtensor$_ptr $;",
             dtensor.guid,
             dtensor.guid,
             offset);
      string dtensor_tile_layout = get_cute_layout(
          mov_to_last(stensor.dim,
                      dtensor.num_dims,
                      d_innermost_dim), // Here we use stensor.dim
          mov_to_last(dtensor_meta.strides, dtensor.num_dims, d_innermost_dim));
      code.e(
          "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);

      // Decide the copy atom to use
      bool is_all_stride_aligned_16B = true;
      size_t alignment = get_num_elems_in_16B(dtensor.data_type);
      for (int i = 0; i < num_dims; ++i) {
        size_t stride = dtensor_meta.strides[i];
        is_all_stride_aligned_16B &= (stride % alignment == 0 || stride == 1);
      }
      bool use_chunked_copy = is_all_stride_aligned_16B;
      bool use_async_copy =
          is_all_stride_aligned_16B && this->config.target_cc >= GPU_CC::A100;

      // TODO(intlsy) Support chunked copy and async copy
      // TODO(intlsy) Support swizzled layout
      // TODO(intlsy) Support TMA
      code.e("using STensor$InputAtom = tb::InputNonChunkedSyncCopy<half_t, "
             "$, DTensor$TileLayout, NUM_THREADS>;",
             stensor.guid,
             mov_last_and_get_layout(stensor, stensor_meta, d_innermost_dim),
             dtensor.guid);

      if (!use_chunked_copy) {
        assert(!use_async_copy);
        // Non-chunked, synchronous copy
      } else if (!use_async_copy) {
        // Chunked, synchronous copy
      } else {
        // Chunked, asynchronous copy
      }

      if (cur_op->forloop_dim < 0) {
        // For input STensor that does not have a forloop_dim, copy it
        code.e("STensor$InputAtom::run(stensor$_ptr, dtensor$_tile_ptr, "
               "thread_idx);",
               stensor.guid,
               stensor.guid,
               dtensor.guid);
      }
    }
  }
  code.e("");

  // Define S2GCopy for all output STensors
  code.e("// S->G copy atoms");
  vector<string> code_after_forloop;
  for (tb::TBOperator const *op : g.operators) {
    if (op->op_type == type::TB_OUTPUT_OP) {
      tb::TBOutputOp const *cur_op = dynamic_cast<tb::TBOutputOp const *>(op);
      // For output ops that have a forloop_dim, we copy the result at the end
      // of a forloop iteration, which means that the source should be the input
      // STensor, while for output ops that do not have a forloop_dim, we
      // accumulate the result during the forloop iteration, which means that
      // the source should be the accumulator
      tb::STensor const &input_stensor = cur_op->input_tensors.at(0);
      tb::STensor const &accum_stensor = cur_op->output_tensors.at(0);
      tb::STensor const &src_stensor = cur_op->forloop_dim >= 0
                                           ? input_stensor
                                           : accum_stensor; // Copy source
      kn::DTensor const &dtensor = cur_op->dtensor;
      STensorMeta const &stensor_meta = stensor_metas.at(src_stensor.guid);
      DTensorMeta const &dtensor_meta = dtensor_metas.at(dtensor.guid);
      assert(dtensor.num_dims == src_stensor.num_dims);
      assert(dtensor.data_type == src_stensor.data_type);
      int num_dims = dtensor.num_dims;
      int d_innermost_dim = dtensor_meta.innermost_dim;

      code.e("// Copy for S->G: stensor $ -> dtensor $",
             src_stensor.guid,
             dtensor.guid);

      // Get the starting address of my tile
      // For output tensor that does not have a forloop_dim, the shape of the
      // tile should be identical to the STensor. Otherwise, it should be the
      // shape of STensor * forloop_range
      string offset = "";
      int3 omap = cur_op->output_map;
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
        }
      }
      code.e("half_t *dtensor$_tile_ptr = dtensor$_ptr $;",
             dtensor.guid,
             dtensor.guid,
             offset);
      string dtensor_tile_layout = get_cute_layout(
          mov_to_last(src_stensor.dim,
                      dtensor.num_dims,
                      d_innermost_dim), // Here we use stensor.dim
          mov_to_last(dtensor_meta.strides, dtensor.num_dims, d_innermost_dim));
      code.e(
          "using DTensor$TileLayout = $;", dtensor.guid, dtensor_tile_layout);

      // Decide the copy atom
      bool is_all_stride_aligned_16B = true;
      size_t alignment = get_num_elems_in_16B(dtensor.data_type);
      for (int i = 0; i < num_dims; ++i) {
        size_t stride = dtensor_meta.strides[i];
        is_all_stride_aligned_16B &= (stride % alignment == 0 || stride == 1);
      }
      bool use_chunked_copy = is_all_stride_aligned_16B;
      // Since the layout of the output tensor is designed by us (the
      // transpiler), it should always have a friendly layout
      assert(use_chunked_copy);

      // TODO(intlsy) Support chunked copy
      // TODO(intlsy) Support TMA
      code.e(
          "using STensor$OutputAtom = tb::OutputNonChunkedSyncCopy<half_t, "
          "DTensor$TileLayout, $, NUM_THREADS>;",
          src_stensor.guid,
          dtensor.guid,
          mov_last_and_get_layout(src_stensor, stensor_meta, d_innermost_dim));

      if (cur_op->forloop_dim < 0) {
        // For output tensors that do not have a forloop_dim, generate the
        // instruction for saving the result to the output dtensor, and clear
        // the accumulator
        code_after_forloop.push_back(fmt("STensor$OutputAtom::run(dtensor$_"
                                         "tile_ptr, stensor$_ptr, thread_idx);",
                                         src_stensor.guid,
                                         dtensor.guid,
                                         src_stensor.guid));
        size_t num_elems = 0;
        for (int i = 0; i < src_stensor.num_dims; ++i) {
          num_elems =
              std::max(num_elems, src_stensor.dim[i] * stensor_meta.strides[i]);
        }
        code.e("tb::ClearOutputAccumKernel<half_t, $, "
               "NUM_THREADS>::run(stensor$_ptr, thread_idx);",
               num_elems,
               accum_stensor.guid);
      }
    }
  }

  code.e("__syncthreads();");

  // Declare the for loop
  // TODO(intlsy) Remove the loop when `plan.forloop_range` is 1
  // TODO(intlsy) Loop unrolling
  assert(plan.forloop_range >= 1);
  code.e("// The main loop");
  code.e("for (int for_idx = 0; for_idx < $; for_idx++) {", plan.forloop_range);

  for (tb::TBOperator const *op : g.operators) {
    std::string op_type_str;
    to_json(op_type_str, op->op_type);
    code.e("{");
    code.e("// OP type: $", op_type_str);
    switch (op->op_type) {
      case type::TB_INPUT_OP: {
        // For input STensor that has a forloop_dim, copy it
        tb::TBInputOp const *cur_op = dynamic_cast<tb::TBInputOp const *>(op);
        if (cur_op->forloop_dim >= 0) {
          kn::DTensor const &dtensor = cur_op->dtensor;
          tb::STensor const &stensor = cur_op->output_tensors.at(0);
          int tile_side_len = stensor.dim[cur_op->forloop_dim];
          size_t forloop_dim_stride =
              dtensor_metas.at(dtensor.guid).strides[cur_op->forloop_dim];
          code.e("STensor$InputAtom::run(stensor$_ptr, dtensor$_tile_ptr + "
                 "$*for_idx, thread_idx);",
                 stensor.guid,
                 stensor.guid,
                 dtensor.guid,
                 tile_side_len * forloop_dim_stride);
        }
        break;
      }
      case type::TB_OUTPUT_OP: {
        tb::TBOutputOp const *cur_op = dynamic_cast<tb::TBOutputOp const *>(op);
        if (cur_op->forloop_dim >= 0) {
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
        } else {
          // For output DTensor that does not have a forloop_dim, accumulate it
          tb::STensor const &stensor_input = cur_op->input_tensors.at(0);
          tb::STensor const &stensor_accum = cur_op->output_tensors.at(0);
          STensorMeta const &stensor_input_meta =
              stensor_metas.at(stensor_input.guid);
          STensorMeta const &stensor_accum_meta =
              stensor_metas.at(stensor_accum.guid);
          // The two tensors should have the same layout
          assert(vector<int>(stensor_input.dim,
                             stensor_input.dim + stensor_input.num_dims) ==
                 vector<int>(stensor_accum.dim,
                             stensor_accum.dim + stensor_accum.num_dims));
          assert(vector<size_t>(stensor_input_meta.strides,
                                stensor_input_meta.strides +
                                    stensor_input.num_dims) ==
                 vector<size_t>(stensor_accum_meta.strides,
                                stensor_accum_meta.strides +
                                    stensor_accum.num_dims));
          string layout =
              mov_last_and_get_layout(stensor_accum,
                                      stensor_accum_meta,
                                      stensor_accum_meta.innermost_dim);
          code.e(
              "using Kernel = tb::AccumOutputKernel<half_t, $, NUM_THREADS>;",
              layout);
          code.e("Kernel::run(stensor$_ptr, stensor$_ptr, thread_idx);",
                 stensor_accum.guid,
                 stensor_input.guid);
        }
        break;
      };
      case type::TB_MATMUL_OP: {
        break;
      };
      case type::TB_EXP_OP: {
        tb::STensor const &input = op->input_tensors.at(0);
        tb::STensor const &output = op->output_tensors.at(0);
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
        string out_layout = mov_last_and_get_layout(
            output, stensor_metas.at(output.guid), iter_dim);
        code.e("using InLayout = $;", in_layout);
        code.e("using OutLayout = $;", out_layout);
        // Define and run the kernel
        code.e(
            "using Kernel = tb::ElementUnaryKernel<half_t, "
            "tb::ElementUnaryOpType::EXP, OutLayout, InLayout, NUM_THREADS>;");
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
        tb::STensor const &output = op->output_tensors.at(0);
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
        string out_layout = mov_last_and_get_layout(
            output, stensor_metas.at(output.guid), iter_dim);
        code.e("using In0Layout = $;", in0_layout);
        code.e("using In1Layout = $;", in1_layout);
        code.e("using OutLayout = $;", out_layout);
        // Define and run the kernel
        code.e("using Kernel = tb::ElementBinaryKernel<half_t, "
               "tb::ElementBinaryOpType::$, OutLayout, In0Layout, In1Layout, "
               "NUM_THREADS>;",
               op_type_str);
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
    code.e("__syncthreads();");
  }
  code.e("}"); // For loop

  // For output ops that do not have a forloop dim, save the result in the
  // accumulator to the output dtensor
  for (string const &line : code_after_forloop) {
    code.e(line);
  }

  code.e("}"); // kernel

  return Transpiler::CustomOPTranspileResult{func_name, code.to_string()};
}

} // namespace transpiler
} // namespace mirage
