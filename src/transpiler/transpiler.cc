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

#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace transpiler {

using std::string;
namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

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

TranspileResult Transpiler::generate_code() {
  // Generate header
  CodeKeeper header;
  header.e("#define NUM_GPUS $", num_gpus);
  header.e("#define USE_NVSHMEM $", use_nvshmem);
  header.e("#include \"runtime.h\"");
  header.e("using namespace cute;");

  CodeKeeper exec;
  exec.e("void execute_mugraph(std::vector<void const *> input_tensors, "
         "std::vector<void*> output_tensors"
         ", void* buf) {");
  for (kn::KNOperator *const op : g->operators) {
    std::string op_type_str;
    to_json(op_type_str, op->op_type);
    exec.e("{");
    exec.e("// OP type: $", op_type_str);
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        // Input op
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
        exec.e(
            "gemm<CUBLAS_COMPUTE_16F>($,$,$, $,$,$, $,$, $,$, $,$, $, $,$,$);",
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
      case type::KNOperatorType::KN_EXP_OP: {
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
        int shift_amount = meta_in0.innermost_dim;
        string in0_layout = shift_and_get_layout(in0, meta_in0, shift_amount);
        string out0_layout = shift_and_get_layout(in0, meta_in0, shift_amount);
        // Get tensor ptrs
        auto [in0_ptr_name, in0_ptr_code] = get_dtensor_ptr(in0);
        auto [out0_ptr_name, out0_ptr_code] = get_dtensor_ptr(out0);
        exec.e(in0_ptr_code);
        exec.e(out0_ptr_code);
        // Create kernel instance
        exec.e("using kernel = ElementUnaryKernel<half_t, "
               "ElementUnaryOpType::EXP, $, $>;",
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
        string in0_layout =
            shift_and_get_layout(in0, meta_in0, meta_in0.innermost_dim);
        string in1_layout =
            shift_and_get_layout(in1, meta_in1, meta_in1.innermost_dim);
        string out0_layout =
            shift_and_get_layout(out0, meta_out0, meta_out0.innermost_dim);
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
        exec.e("using kernel = ElementBinaryKernel<half_t, "
               "ElementBinaryOpType::$, $, $, $>;",
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
              "Warning: In the current implementation of the reduction kernel, "
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
        exec.e("using kernel = ReductionKernel<half_t, $, $, $>;",
               layout_in0,
               layout_out0,
               new_reduction_dim);
        // Launch kernel
        exec.e("kernel::run($, $);", out0_ptr_name, in0_ptr_name);
        break;
      }
      default:
        assert(false && ("Unsupported operator type: " +
                         std::to_string(int(op->op_type)))
                            .c_str());
    }
    exec.e("}");
  }
  exec.e("}");

  string result = header.to_string() + "\n" + exec.to_string() + "\n";
  vector<vector<size_t>> output_strides;
  for (kn::DTensor const *output_tensor : output_tensors) {
    DTensorMeta const &meta = dtensor_metas.at(output_tensor->guid);
    output_strides.push_back(
        vector<size_t>(meta.strides, meta.strides + output_tensor->num_dims));
  }
  return {result, this->d_buf_size, output_strides};
}

} // namespace transpiler
} // namespace mirage
