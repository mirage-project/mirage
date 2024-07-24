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

// A helper function for mov_inner_dim_and_get_layout
template <typename T>
static std::vector<T>
    mov_to_last_and_reverse(T const *vec, size_t numel, int idx) {
  std::vector<T> result;
  result.reserve(numel);
  result.insert(result.end(), vec, vec + idx);
  result.insert(result.end(), vec + idx + 1, vec + numel);
  result.push_back(vec[idx]);
  std::reverse(result.begin(), result.end());
  return result;
}

// Move the innermost dim to the last dim, and then flip the order of the dims
// and strides, and finally format it as a string.
// If innermost_dim is -1, do not move the innermost dim.
// Assume the tensor has N dimensions and the innermost dim is i, then the
// function is equivalent to torch.permute(tensor, [i, N, N-1, ..., i+1, i-1,
// i-2, ..., 0]) This function is helpful for element-wise ops, since the
// processing order of elements do not affect the correctness.
static string mov_last_reverse_and_get_layout(kernel::DTensor const &dtensor,
                                              DTensorMeta const &meta,
                                              int innermost_dim) {
  if (innermost_dim == -1) {
    innermost_dim = dtensor.num_dims - 1;
  }
  assert(0 <= innermost_dim && innermost_dim < dtensor.num_dims);
  return fmt("Layout<Shape<$>, Stride<$>>",
             map_to_cute_int(mov_to_last_and_reverse(
                 dtensor.dim, dtensor.num_dims, innermost_dim)),
             map_to_cute_int(mov_to_last_and_reverse(
                 meta.strides, dtensor.num_dims, innermost_dim)));
}

static string get_cute_layout(kernel::DTensor const &dtensor,
                              DTensorMeta const &meta) {
  return mov_last_reverse_and_get_layout(dtensor, meta, -1);
}

// The following code are related to threadblock graph transpilation

// Transpile a custom KN operator (i.e. a custom block graph) into CUDA code
// Will return a CustomOPTranspileResult object. See comments in transpiler.h
// for more details
Transpiler::CustomOPTranspileResult
    Transpiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  tb::ExecutionPlan const &plan = op->plan;

  static int custom_kernel_idx_counter = 0;
  int cur_custom_kernel_idx = custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  CodeKeeper code;
  code.e(
      "__global__ void $($, $) {",
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
  code.e("extern __shared__ char* buf[];");

  // Define DTensor as cute::Tensor
  for (kn::DTensor const &dtensor :
       Combine(op->output_tensors, op->input_tensors)) {
    DTensorMeta const &meta = dtensor_metas.at(dtensor.guid);
    dguid_t guid = dtensor.guid;
    code.e("using DTensor$Layout = $;", guid, get_cute_layout(dtensor, meta));
    code.e("Tensor dtensor$ = make_tensor(make_gmem_ptr(dtensor$_ptr), "
           "DTensor$Layout{});",
           guid,
           guid,
           guid);
  }

  // Define G2SCopy for all input STensors
  // For input STensor that does not have a forloop_dim, read it and save in
  // shared mem
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
      if (cur_op->forloop_dim < 0) {
      }
    }
  }

  // Declare the for loop
  // TODO Remove the loop when `plan.forloop_range` is 1
  assert(plan.forloop_range >= 1);
  code.e("for (int for_idx = 0; for_idx < $; for_idx++) {", plan.forloop_range);

  // Declare STensor fragments
  code.e("}"); // For loop

  code.e("}"); // kernel

  return Transpiler::CustomOPTranspileResult{func_name, code.to_string()};
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

TranspileResult Transpiler::generate_code() {
  // Generate header
  CodeKeeper header;
  header.e("#define NUM_GPUS $", num_gpus);
  header.e("#define USE_NVSHMEM $", use_nvshmem);
  header.e("#include \"runtime.h\"");
  header.e("using namespace cute;");

  CodeKeeper
      custom_kernels; // This keeps all code for custom kernels (KNCustomizedOp)
  CodeKeeper init;    // This keeps all code in the `_init` function (e.g.
                      // cudaFuncSetAttribute)
  CodeKeeper exec;    // This keeps all code in the `_execute_mugraph` function

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
        int innermost_dim = meta_in0.innermost_dim;
        string in0_layout =
            mov_last_reverse_and_get_layout(in0, meta_in0, innermost_dim);
        string out0_layout =
            mov_last_reverse_and_get_layout(in0, meta_in0, innermost_dim);
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
        int innermost_dim = meta_in0.innermost_dim;
        string in0_layout =
            mov_last_reverse_and_get_layout(in0, meta_in0, innermost_dim);
        string in1_layout =
            mov_last_reverse_and_get_layout(in1, meta_in1, innermost_dim);
        string out0_layout =
            mov_last_reverse_and_get_layout(out0, meta_out0, innermost_dim);
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
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        // Customized op
        kn::KNCustomizedOp const *cur_op =
            dynamic_cast<kn::KNCustomizedOp const *>(op);
        tb::ExecutionPlan const &plan = cur_op->plan;
        assert(custom_op_metas.count(cur_op));
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
        // Launch kernel
        exec.e("dim3 grid_dim($, $, $);",
               plan.grid_dim.x,
               plan.grid_dim.y,
               plan.grid_dim.z);
        exec.e("dim3 block_dim($, $, $);",
               plan.block_dim.x,
               plan.block_dim.y,
               plan.block_dim.z);
        exec.e("size_t smem_size = $;", custom_op_metas[cur_op].smem_size);
        exec.e("$<<<grid_dim, block_dim, smem_size>>>($);",
               result.func_name,
               ptr_names);
        custom_kernels.e(result.code);
        init.e("cudaFuncSetAttribute($, "
               "cudaFuncAttributeMaxDynamicSharedMemorySize, $);",
               result.func_name,
               custom_op_metas[cur_op].smem_size);
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
  for (kn::DTensor const *dtensor : this->output_tensors) {
    DTensorMeta meta = dtensor_metas.at(dtensor->guid);
    output_directives.push_back(OutputTensorDirective{
        meta.phy_size,
        vector<int>(dtensor->dim, dtensor->dim + dtensor->num_dims),
        vector<size_t>(meta.strides, meta.strides + dtensor->num_dims)});
  }
  return TranspileResult{code, this->d_buf_size, output_directives};
}

} // namespace transpiler
} // namespace mirage
