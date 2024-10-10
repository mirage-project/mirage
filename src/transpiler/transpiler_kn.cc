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

// The following code are related to kernel graph transpilation

// Get the pointer of a DTensor. The tensor may be an input/output/intermediate
// one Return: (pointer_var_name, code_to_get_the_pointer) For example, when
// requesting for an input tensor, may return:
// ("dtensor1000000", "half_t* dtensor1000000 = input_tensors[0];")
std::pair<string, string>
    Transpiler::get_dtensor_ptr(kn::DTensor const &dtensor) {
  auto guid = dtensor.guid;
  DTensorMeta const &meta = dtensor_metas.at(guid);
  string pointer_var_name = fmt("dtensor$", guid);
  string code = "";
  if (meta.is_input) {
    code = fmt("half_t *$ = (half_t*)input_tensors.at($);",
               pointer_var_name,
               meta.input_idx);
  } else if (meta.is_output) {
    code = fmt("half_t *$ = (half_t*)output_tensors.at($);",
               pointer_var_name,
               meta.output_idx);
  } else {
    code = fmt(
        "half_t *$ = (half_t*)((char*)buf + $);", pointer_var_name, meta.addr);
  }
  return {pointer_var_name, code};
}

static string get_kn_op_str(type::KNOperatorType type) {
  auto toString = [](type::KNOperatorType type) -> string {
    switch (type) {
      case type::KN_EXP_OP:
        return "EXP";
      case type::KN_SILU_OP:
        return "SILU";
      case type::KN_SQUARE_OP:
        return "SQUARE";
      case type::KN_SQRT_OP:
        return "SQRT";
      default:
        assert(0);
    }
  };
  return toString(type);
}

TranspileResult Transpiler::transpile_ugraph() {
  // Generate header
  CodeKeeper header;
  header.e("#define NUM_GPUS $", num_gpus);
  header.e("#define USE_NVSHMEM $", use_nvshmem);
  header.e("#include \"runtime.h\"");
  header.e("using namespace cute;");

  CodeKeeper custom_kernels; // This keeps all code for custom kernels
                             // (KNCustomizedOp)
  CodeKeeper init; // This keeps all code in the `_init` function (e.g.
                   // cudaFuncSetAttribute)
  CodeKeeper exec; // This keeps all code in the `_execute_mugraph` function

  init.e("static void _init() {");
  exec.e(
      "static void _execute_mugraph(std::vector<void const *> input_tensors, "
      "std::vector<void*> output_tensors"
      ", void* buf) {");
  for (kn::KNOperator *const op : g->operators) {
    std::string op_type_str;
    to_json(op_type_str, op->op_type);
    exec.e("{");
    exec.e("// OP type: $", op_type_str);
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP:
      case type::KNOperatorType::KN_OUTPUT_OP: {
        // Input/Output op
        break;
      }
      case type::KNOperatorType::KN_MATMUL_OP: {
        // Matrix multiplication. Going to call cuBLAS
        kn::DTensor &in0 = op->input_tensors.at(0);
        kn::DTensor &in1 = op->input_tensors.at(1);
        kn::DTensor &out0 = op->output_tensors.at(0);
        DTensorMeta meta_in0 = dtensor_metas.at(in0.guid);
        DTensorMeta meta_in1 = dtensor_metas.at(in1.guid);
        DTensorMeta meta_out0 = dtensor_metas.at(out0.guid);
        // Both tensors should have their innermost dim as the last two dims
        assert(meta_in0.innermost_dim >= in0.num_dims - 2);
        assert(meta_in1.innermost_dim >= in1.num_dims - 2);
        assert(meta_out0.innermost_dim >= out0.num_dims - 2);
        // Get tensor ptrs
        auto [in0_ptr_name, in0_ptr_code] = get_dtensor_ptr(in0);
        auto [in1_ptr_name, in1_ptr_code] = get_dtensor_ptr(in1);
        auto [out0_ptr_name, out0_ptr_code] = get_dtensor_ptr(out0);
        exec.e(in0_ptr_code);
        exec.e(in1_ptr_code);
        exec.e(out0_ptr_code);
        // Calculate info about matrix shape
        int m = in0.dim[in0.num_dims - 2];
        int k = in0.dim[in0.num_dims - 1];
        int n = in1.dim[in1.num_dims - 1];
        assert(k == in1.dim[in1.num_dims - 2]);
        assert(m == out0.dim[out0.num_dims - 2]);
        assert(n == out0.dim[out0.num_dims - 1]);
        // Calculate info about batching
        assert(in0.num_dims == in1.num_dims);
        assert(in0.num_dims == out0.num_dims);
        size_t batch_size = 1;
        for (int i = 0; i < in0.num_dims - 2; i++) {
          assert(in0.dim[i] == in1.dim[i] && in0.dim[i] == out0.dim[i]);
          batch_size *= in0.dim[i];
        }
        size_t batch_stride_A =
            in0.num_dims == 2 ? 0 : meta_in0.strides[in0.num_dims - 3];
        size_t batch_stride_B =
            in1.num_dims == 2 ? 0 : meta_in1.strides[in1.num_dims - 3];
        size_t batch_stride_C =
            out0.num_dims == 2 ? 0 : meta_out0.strides[out0.num_dims - 3];
        // Run GEMM
        exec.e("kn::gemm<CUBLAS_COMPUTE_16F>($,$,$, $,$,$, $,$, $,$, $,$, $, "
               "$,$,$);",
               out0_ptr_name,
               in0_ptr_name,
               in1_ptr_name,
               m,
               n,
               k,
               meta_in0.strides[in0.num_dims - 2],
               meta_in0.strides[in0.num_dims - 1],
               meta_in1.strides[in1.num_dims - 2],
               meta_in1.strides[in1.num_dims - 1],
               meta_out0.strides[out0.num_dims - 2],
               meta_out0.strides[out0.num_dims - 1],
               batch_size,
               batch_stride_A,
               batch_stride_B,
               batch_stride_C);
        break;
      }
      case type::KNOperatorType::KN_EXP_OP:
      case type::KNOperatorType::KN_SILU_OP:
      case type::KNOperatorType::KN_SQUARE_OP:
      case type::KNOperatorType::KN_SQRT_OP: {
        // Elemwise unary op
        kn::DTensor &in0 = op->input_tensors.at(0);
        kn::DTensor &out0 = op->output_tensors.at(0);
        DTensorMeta meta_in0 = dtensor_metas.at(in0.guid);
        DTensorMeta meta_out0 = dtensor_metas.at(out0.guid);
        // The two tensors must have the same shape
        assert(in0.num_dims == out0.num_dims);
        for (int i = 0; i < in0.num_dims; i++) {
          assert(in0.dim[i] == out0.dim[i]);
        }
        if (meta_in0.innermost_dim != meta_out0.innermost_dim) {
          printf("Warning: In the current implementation of the elementwise "
                 "unary kernel, global memory access won't be coalesced when "
                 "input tensor's innermost_dim (%d) != output tensor's "
                 "innermost_dim (%d)"
                 "This may cause performance degration\n",
                 meta_in0.innermost_dim,
                 meta_out0.innermost_dim);
        }
        // Assemble the new shape and stride
        // We move the innermost dim to the first dim to coalesce global mem
        // access
        int innermost_dim = meta_in0.innermost_dim;
        string in0_layout =
            mov_last_and_get_layout(in0, meta_in0, innermost_dim);
        string out0_layout =
            mov_last_and_get_layout(in0, meta_in0, innermost_dim);
        // Get tensor ptrs
        auto [in0_ptr_name, in0_ptr_code] = get_dtensor_ptr(in0);
        auto [out0_ptr_name, out0_ptr_code] = get_dtensor_ptr(out0);
        exec.e(in0_ptr_code);
        exec.e(out0_ptr_code);
        // Create kernel instance
        exec.e("using kernel = kn::ElementUnaryKernel<half_t, "
               "kn::ElementUnaryOpType::$, $, $>;",
               get_kn_op_str(op->op_type),
               in0_layout,
               out0_layout);
        // Launch kernel
        exec.e("kernel::run($, $);", out0_ptr_name, in0_ptr_name);
        break;
      }
      case type::KNOperatorType::KN_ADD_OP:
      case type::KNOperatorType::KN_MUL_OP:
      case type::KNOperatorType::KN_DIV_OP: {
        // Elemwise binary op
        kn::DTensor &in0 = op->input_tensors.at(0);
        kn::DTensor &in1 = op->input_tensors.at(1);
        kn::DTensor &out0 = op->output_tensors.at(0);
        DTensorMeta meta_in0 = dtensor_metas.at(in0.guid);
        DTensorMeta meta_in1 = dtensor_metas.at(in1.guid);
        DTensorMeta meta_out0 = dtensor_metas.at(out0.guid);
        // The three tensors must have the same shape except broadcasting
        // dimensions
        assert(in0.num_dims == in1.num_dims && in0.num_dims == out0.num_dims);
        for (int i = 0; i < in0.num_dims; i++) {
          assert(in0.dim[i] == out0.dim[i] || in0.dim[i] == 1);
          assert(in1.dim[i] == out0.dim[i] || in1.dim[i] == 1);
        }
        if (meta_in0.innermost_dim != meta_in1.innermost_dim ||
            meta_in0.innermost_dim != meta_out0.innermost_dim) {
          printf("Warning: In the current implementation of the elementwise "
                 "binary kernel, global memory access won't be coalesced when "
                 "input tensors' innermost_dim (%d and %d) != output tensor's "
                 "innermost_dim (%d)"
                 "This may cause performance degration\n",
                 meta_in0.innermost_dim,
                 meta_in1.innermost_dim,
                 meta_out0.innermost_dim);
        }
        // Assemble the new shape and stride
        // We move the innermost dim to the first dim to coalesce global mem
        // access
        int innermost_dim = meta_in0.innermost_dim;
        string in0_layout =
            mov_last_and_get_layout(in0, meta_in0, innermost_dim);
        string in1_layout =
            mov_last_and_get_layout(in1, meta_in1, innermost_dim);
        string out0_layout =
            mov_last_and_get_layout(out0, meta_out0, innermost_dim);
        // Get tensor ptrs
        auto [in0_ptr_name, in0_ptr_code] = get_dtensor_ptr(in0);
        auto [in1_ptr_name, in1_ptr_code] = get_dtensor_ptr(in1);
        auto [out0_ptr_name, out0_ptr_code] = get_dtensor_ptr(out0);
        exec.e(in0_ptr_code);
        exec.e(in1_ptr_code);
        exec.e(out0_ptr_code);
        // Get OPType
        string op_type_str = op->op_type == type::KN_ADD_OP   ? "ADD"
                             : op->op_type == type::KN_MUL_OP ? "MUL"
                             : op->op_type == type::KN_DIV_OP ? "DIV"
                                                              : "";
        assert(op_type_str != "");
        // Create kernel instance
        exec.e("using kernel = kn::ElementBinaryKernel<half_t, "
               "kn::ElementBinaryOpType::$, $, $, $>;",
               op_type_str,
               in0_layout,
               in1_layout,
               out0_layout);
        // Launch kernel
        exec.e(
            "kernel::run($, $, $);", out0_ptr_name, in0_ptr_name, in1_ptr_name);
        break;
      }
      case type::KNOperatorType::KN_REDUCTION_0_OP:
      case type::KNOperatorType::KN_REDUCTION_1_OP:
      case type::KNOperatorType::KN_REDUCTION_2_OP: {
        // Reduction op
        int reduction_dim = op->op_type - type::KN_REDUCTION_0_OP;
        kn::DTensor &in0 = op->input_tensors.at(0);
        kn::DTensor &out0 = op->output_tensors.at(0);
        DTensorMeta meta_in0 = dtensor_metas.at(in0.guid);
        DTensorMeta meta_out0 = dtensor_metas.at(out0.guid);
        // The two tensors must have the same shape (except the reduction dim)
        assert(in0.num_dims == out0.num_dims);
        for (int i = 0; i < in0.num_dims; i++) {
          if (i != reduction_dim) {
            assert(in0.dim[i] == out0.dim[i]);
          }
        }
        // Currently we require them to have the same innermost_dim
        assert(meta_in0.innermost_dim == meta_out0.innermost_dim);
        // The size of the reduction dim must be divisible
        assert(in0.dim[reduction_dim] % out0.dim[reduction_dim] == 0);
        // Assemble the new shape and stride
        if (meta_in0.innermost_dim == reduction_dim) {
          printf(
              "Warning: In the current implementation of the reduction "
              "kernel, "
              "global memory access won't be coalesced when reduction_dim == "
              "innermost_dim. This may cause performance degration\n");
        }
        // We move the innermost dim to the first dim to coalesce global mem
        // access
        vector<int> new_shape_in0(in0.num_dims);
        vector<int> new_shape_out0(in0.num_dims);
        vector<size_t> new_strides_in0(in0.num_dims);
        vector<size_t> new_strides_out0(in0.num_dims);
        for (int i = 0; i < in0.num_dims; i++) {
          int src_dim = (i + meta_in0.innermost_dim) % in0.num_dims;
          new_shape_in0[i] = in0.dim[src_dim];
          new_shape_out0[i] = out0.dim[src_dim];
          new_strides_in0[i] = meta_in0.strides[src_dim];
          new_strides_out0[i] = meta_out0.strides[src_dim];
        }
        int new_reduction_dim =
            (reduction_dim + in0.num_dims - meta_in0.innermost_dim) %
            in0.num_dims;
        string layout_in0 = fmt("Layout<Shape<$>, Stride<$>>",
                                map_to_cute_int(new_shape_in0),
                                map_to_cute_int(new_strides_in0));
        string layout_out0 = fmt("Layout<Shape<$>, Stride<$>>",
                                 map_to_cute_int(new_shape_out0),
                                 map_to_cute_int(new_strides_out0));
        // Get tensor ptrs
        auto [in0_ptr_name, in0_ptr_code] = get_dtensor_ptr(in0);
        auto [out0_ptr_name, out0_ptr_code] = get_dtensor_ptr(out0);
        exec.e(in0_ptr_code);
        exec.e(out0_ptr_code);
        // Create kernel instance
        exec.e("using kernel = kn::ReductionKernel<half_t, $, $, $>;",
               layout_in0,
               layout_out0,
               new_reduction_dim);
        // Launch kernel
        exec.e("kernel::run($, $);", out0_ptr_name, in0_ptr_name);
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        // Customized op
        kn::KNCustomizedOp const *cur_op =
            dynamic_cast<kn::KNCustomizedOp const *>(op);
        // tb::ExecutionPlan const &plan = cur_op->plan;
        tb::Graph const &bgraph = cur_op->bgraph;
        // Get DTensor ptrs
        // We make the aggrement that, when calling a custom kernel, the
        // arguments are in the order of "output_tensors, input_tensors"
        vector<string> ptr_names;
        for (kn::DTensor const &dtensor :
             Combine(cur_op->output_tensors, cur_op->input_tensors)) {
          auto [ptr_name, ptr_code] = get_dtensor_ptr(dtensor);
          exec.e(ptr_code);
          ptr_names.push_back(ptr_name);
        }
        // Transpile
        CustomOPTranspileResult result = transpile_kn_custom_op(cur_op);
        // Checkings against grid dim and block dim
        if (config.target_cc <= GPU_CC::H100) {
          // According to
          // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
          // all GPUs up to H100 have the same restriction
          assert(bgraph.grid_dim.x <= (1LL << 31) - 1);
          assert(bgraph.grid_dim.y <= 65535);
          assert(bgraph.grid_dim.z <= 65535);
          assert((long long)bgraph.grid_dim.x * bgraph.grid_dim.y *
                     bgraph.grid_dim.z <=
                 (1LL << 31) - 1);
          assert(bgraph.block_dim.x <= 1024);
          assert(bgraph.block_dim.y <= 1024);
          assert(bgraph.block_dim.z <= 64);
          assert((long long)bgraph.block_dim.x * bgraph.block_dim.y *
                     bgraph.block_dim.z <=
                 1024);
        } else {
          // In the future, we may need to update this part for GPUs later
          // than H100
          assert(0);
        }
        // Launch kernel
        exec.e("dim3 grid_dim($, $, $);",
               bgraph.grid_dim.x,
               bgraph.grid_dim.y,
               bgraph.grid_dim.z);
        exec.e("dim3 block_dim($, $, $);",
               bgraph.block_dim.x,
               bgraph.block_dim.y,
               bgraph.block_dim.z);
        exec.e("size_t smem_size = $;", result.smem_size);
        exec.e("$<<<grid_dim, block_dim, smem_size>>>($);",
               result.func_name,
               ptr_names);
        custom_kernels.e(result.code);
        init.e("cudaFuncSetAttribute($, "
               "cudaFuncAttributeMaxDynamicSharedMemorySize, $);",
               result.func_name,
               result.smem_size);
        break;
      }
      default:
        assert(false && ("Unsupported operator type: " +
                         std::to_string(int(op->op_type)))
                            .c_str());
    }
    exec.e("}");
  }
  init.e("}");
  exec.e("}");

  string code = fmt("$\n$\n$\n$\n",
                    header.to_string(),
                    custom_kernels.to_string(),
                    init.to_string(),
                    exec.to_string());
  vector<OutputTensorDirective> output_directives;
  for (kn::DTensor const &dtensor : this->mugraph_output_tensors) {
    assert(dtensor_metas.find(dtensor.guid) != dtensor_metas.end());

    DTensorMeta meta = dtensor_metas.at(dtensor.guid);
    output_directives.push_back(OutputTensorDirective{
        meta.num_phy_elems,
        vector<int>(dtensor.dim, dtensor.dim + dtensor.num_dims),
        vector<size_t>(meta.strides, meta.strides + dtensor.num_dims)});
  }
  return TranspileResult{code, this->d_buf_size, output_directives};
}

} // namespace transpiler
} // namespace mirage
