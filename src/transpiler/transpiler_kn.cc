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
#include <mirage/kernel/device_tensor.h>
#include <unordered_set>

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

// The following code are related to kernel graph transpilation

// Get the pointer of a DTensor. The tensor may be an input/output/intermediate
// one Return: (pointer_var_name, code_to_get_the_pointer) For example, when
// requesting for an input tensor, may return:
// ("dtensor1000000", "half_t* dtensor1000000 = input_tensors[0];")
std::pair<string, string>
    Transpiler::get_dtensor_ptr(kn::DTensor const &dtensor, 
                                CodeKeeper *comm_buffers, 
                                size_t *next_comm_buffer_idx) {
  auto guid = dtensor.guid;
  DTensorMeta const &meta = dtensor_metas.at(guid);
  string pointer_var_name = fmt("dtensor$", guid);
  string code = "";
  // string signal_code = "";
  // string malloc_code = "";
  if (dtensor.prologue == mirage::type::TBPrologueType::TB_PROLOGUE_ALLGATHER) {
    assert(comm_buffers && next_comm_buffer_idx);
    size_t comm_buffer_idx = (*next_comm_buffer_idx)++;
    comm_buffers->e("sizeof($) * $,\t// Buffer $", \
            get_datatype_str(dtensor.data_type),
            meta.num_phy_elems,
            comm_buffer_idx);
    code = fmt("$ *$ = ($*)comm_buffers.at($);", \
            get_datatype_str(dtensor.data_type),
            pointer_var_name,
            get_datatype_str(dtensor.data_type),
            comm_buffer_idx);
    /*
    code = fmt("$ *$ = ($*)nvshmem_malloc(sizeof($) * $);",
                      get_datatype_str(dtensor.data_type),
                      pointer_var_name,
                      get_datatype_str(dtensor.data_type),
                      get_datatype_str(dtensor.data_type),
                      meta.num_phy_elems);
    */
  }
  else if (meta.is_input) {
    code = fmt("$ *$ = ($*)input_tensors.at($);",
               get_datatype_str(dtensor.data_type),
               pointer_var_name,
               get_datatype_str(dtensor.data_type),
               meta.input_idx);
  } else if (dtensor.is_nvshmem_tensor) {
    assert(comm_buffers && next_comm_buffer_idx);
    if (dtensor.epilogue == type::TBEpilogueType::TB_EPILOGUE_REDUCESCATTER) {
      if (comm_buffer_metas.find(guid) == comm_buffer_metas.end()) {
        size_t comm_buffer_idx = (*next_comm_buffer_idx)++;
        comm_buffers->e("sizeof($) * $,\t// Buffer $", \
                get_datatype_str(dtensor.data_type),
                meta.num_phy_elems * g->gpu_dim.x,
                comm_buffer_idx);
        comm_buffer_metas[guid] = comm_buffer_idx;
        code = fmt("$ *$ = ($*)comm_buffers.at($);", \
                get_datatype_str(dtensor.data_type),
                pointer_var_name,
                get_datatype_str(dtensor.data_type),
                comm_buffer_metas[guid]);
      } else if (meta.is_output) {
        code = fmt("$ *$ = ($*)output_tensors.at($);", \
                get_datatype_str(dtensor.data_type),
                pointer_var_name,
                get_datatype_str(dtensor.data_type),
                meta.output_idx);
      } else {
        code = fmt("$ *$ = ($*)comm_buffers.at($);", \
                get_datatype_str(dtensor.data_type),
                pointer_var_name,
                get_datatype_str(dtensor.data_type),
                comm_buffer_metas[guid]);
      }
      /*
      code = fmt("$ *$ = to_nvshmem_ptr<$>($);",
                 get_datatype_str(dtensor.data_type),
                 pointer_var_name,
                 get_datatype_str(dtensor.data_type),
                 meta.num_phy_elems * g->gpu_dim.x);
      */
    } else {
      if (comm_buffer_metas.find(guid) == comm_buffer_metas.end()) {
        size_t comm_buffer_idx = (*next_comm_buffer_idx)++;
        comm_buffers->e("sizeof($) * $,\t// Buffer $", \
                get_datatype_str(dtensor.data_type),
                meta.num_phy_elems,
                comm_buffer_idx);
        comm_buffer_metas[guid] = comm_buffer_idx;
      }
      code = fmt("$ *$ = ($*)comm_buffers.at($);", \
              get_datatype_str(dtensor.data_type),
              pointer_var_name,
              get_datatype_str(dtensor.data_type),
              comm_buffer_metas[guid]);
      /*
      code = fmt("$ *$ = to_nvshmem_ptr<$>($);",
                 get_datatype_str(dtensor.data_type),
                 pointer_var_name,
                 get_datatype_str(dtensor.data_type),
                 meta.num_phy_elems);
      */
    }
  } else if (meta.is_output) {
    code = fmt("$ *$ = ($*)output_tensors.at($);",
               get_datatype_str(dtensor.data_type),
               pointer_var_name,
               get_datatype_str(dtensor.data_type),
               meta.output_idx);
  } else {
    code = fmt("$ *$ = ($*)((char*)buf + $);",
               get_datatype_str(dtensor.data_type),
               pointer_var_name,
               get_datatype_str(dtensor.data_type),
               meta.addr);
  }
  return {pointer_var_name, code};
}

std::pair<string, string>
    Transpiler::get_profiling_ptr(int const customized_idx) {
  string pointer_var_name = fmt("profiler_buffer_$", customized_idx);
  string code = "";
  code = fmt("uint64_t *$ = (uint64_t*)profiler_buffer;", pointer_var_name);
  return {pointer_var_name, code};
}

static string get_kn_op_str(type::KNOperatorType type) {
  auto toString = [](type::KNOperatorType type) -> string {
    switch (type) {
      case type::KN_EXP_OP:
        return "EXP";
      case type::KN_SILU_OP:
        return "SILU";
      case type::KN_GELU_OP:
        return "GELU";
      case type::KN_SQUARE_OP:
        return "SQUARE";
      case type::KN_SQRT_OP:
        return "SQRT";
      case type::KN_RELU_OP:
        return "RELU";
      case type::KN_CLAMP_OP:
        return "CLAMP";
      default:
        assert(0);
    }
  };
  return toString(type);
}

TranspileResult Transpiler::transpile_ugraph() {
  size_t max_smem_size = 0;
  size_t next_comm_buffer_idx = 0;
  size_t profiler_buf_size = 0;
  // Generate header

  CodeKeeper header;
  header.e("#define NUM_GPUS $", num_gpus);
  header.e("#define USE_NVSHMEM $", use_nvshmem);
  header.e("#define USE_NCCL $", use_nccl);
  if (config.target_cc == GPU_CC::H100) {
    header.e("#define MIRAGE_GRACE_HOPPER");
  }
  header.e("#include \"runtime.h\"");
  header.e("using namespace cute;");

  CodeKeeper custom_kernels; // This keeps all code for custom kernels
                             // (KNCustomizedOp)
  CodeKeeper init; // This keeps all code in the `_init` function (e.g.
                   // cudaFuncSetAttribute)
  CodeKeeper exec; // This keeps all code in the `_execute_mugraph` function

  CodeKeeper comm_buffers;

  CodeKeeper hopper_tma;

  init.e("static void _init() {");

  exec.e(
      "static void _execute_mugraph(std::vector<void const *> input_tensors, "
      "std::vector<void*> output_tensors, "
      "std::vector<void*> comm_buffers, "
      "void* buf, "
      "cudaStream_t stream, "
      "void* profiler_buffer){");

  comm_buffers.e("static std::vector<size_t> get_comm_sizes() {");
  comm_buffers.e("int npes = nvshmem_n_pes();");
  comm_buffers.e("return {");

  //Assume that we don't use both nccl and nvshmem
  assert(!(use_nccl && use_nvshmem));
  if (use_nccl) {
    //exec.e("// Initialize MPI");
    //exec.e("int argc = 1;");
    //exec.e("const char* argv[] = {\"_execute_mugraph\"};");
    //exec.e("char** argv_ptr = const_cast<char**>(argv);");
    //exec.e("MPI_Init(&argc, &argv_ptr);");
    exec.e("int my_rank, world_size;");
    exec.e("MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);");
    exec.e("MPI_Comm_size(MPI_COMM_WORLD, &world_size);");
    exec.e("");
    exec.e("// Set device to local rank");
    exec.e("cudaSetDevice(my_rank);");
    exec.e("");
    exec.e("// Initialize NCCL");
    exec.e("cudaStream_t s;");
    // TODO (linsj20) Why doesn't this work?
    //exec.e("cudaStreamCreate(&s);");
    exec.e("cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);");
    exec.e("ncclComm_t comm;");
    exec.e("ncclUniqueId id;");
    exec.e("if (my_rank == 0) ncclGetUniqueId(&id);");
    exec.e("MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);");
    exec.e("NCCLCHECK(ncclCommInitRank(&comm, world_size, id, my_rank));");
  }        
  else if (use_nvshmem) { // define nvshemem
    //exec.e("initialize_mpi_nvshmem(rank);");
    exec.e("int mype = nvshmem_my_pe();");
    exec.e("int npes = nvshmem_n_pes();");
  }

  for (kn::KNOperator *const op : g->operators) {
    std::string op_type_str;
    to_json(op_type_str, op->op_type);
    exec.e("{");
    exec.e("// OP type: $", op_type_str);
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP:
        //TODO: multiGPU division using NVSHMEM
        exec.e("// Input Shape: <$, $, $, $>",
               op->output_tensors[0].dim[0],
               op->output_tensors[0].dim[1],
               op->output_tensors[0].dim[2],
               op->output_tensors[0].dim[3]);
        exec.e("// Layout: $",
               get_cute_layout(op->output_tensors[0], dtensor_metas.at(op->output_tensors[0].guid)));
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
        string compute_type =
            (in0.data_type == type::DT_FLOAT16 ? "CUBLAS_COMPUTE_16F"
                                               : "CUBLAS_COMPUTE_32F");
        exec.e("kn::gemm<$>($,$,$, $,$,$, $,$, $,$, $,$, $, "
               "$,$,$);",
               compute_type,
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
      case type::KNOperatorType::KN_GELU_OP:
      case type::KNOperatorType::KN_RELU_OP:
      case type::KNOperatorType::KN_CLAMP_OP:
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
                 "This may cause performance degradation\n",
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
            mov_last_and_get_layout(out0, meta_out0, innermost_dim);
        // Get tensor ptrs
        auto [in0_ptr_name, in0_ptr_code] = get_dtensor_ptr(in0);
        auto [out0_ptr_name, out0_ptr_code] = get_dtensor_ptr(out0);
        exec.e(in0_ptr_code);
        exec.e(out0_ptr_code);
        // Create kernel instance
        exec.e("using kernel = kn::ElementUnaryKernel<$, "
               "kn::ElementUnaryOpType::$, $, $>;",
               get_datatype_str(in0.data_type),
               get_kn_op_str(op->op_type),
               in0_layout,
               out0_layout);
        // Launch kernel
        exec.e("kernel::run($, $);", out0_ptr_name, in0_ptr_name);
        break;
      }
      case type::KNOperatorType::KN_ADD_OP:
      case type::KNOperatorType::KN_MUL_OP:
      case type::KNOperatorType::KN_DIV_OP:
      case type::KNOperatorType::KN_POW_OP: {
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
                             : op->op_type == type::KN_POW_OP ? "POW"
                                                              : "";
        assert(op_type_str != "");
        // Create kernel instance
        exec.e("using kernel = kn::ElementBinaryKernel<$, "
               "kn::ElementBinaryOpType::$, $, $, $>;",
               get_datatype_str(in0.data_type),
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
        exec.e("using kernel = kn::ReductionKernel<$, $, $, $>;",
               get_datatype_str(in0.data_type),
               layout_in0,
               layout_out0,
               new_reduction_dim);
        // Launch kernel
        exec.e("kernel::run($, $);", out0_ptr_name, in0_ptr_name);
        break;
      }
      case type::KNOperatorType::KN_ALLREDUCE_OP: {
        // Allreduce op
        kn::DTensor &in0 = op->input_tensors.at(0);
        kn::DTensor &out0 = op->output_tensors.at(0);
        DTensorMeta meta_in0 = dtensor_metas.at(in0.guid);
        DTensorMeta meta_out0 = dtensor_metas.at(out0.guid);
        // Input and output tensor should have the same amount of elements
        assert(meta_in0.num_phy_elems == meta_out0.num_phy_elems);
        auto [in0_ptr_name, in0_ptr_code] = get_dtensor_ptr(in0);
        auto [out0_ptr_name, out0_ptr_code] = get_dtensor_ptr(out0);
        exec.e(in0_ptr_code);
        exec.e(out0_ptr_code);
        exec.e("NCCLCHECK(ncclAllReduce((const void*)$, "
               "(void*)$, $, ncclFloat16, ncclSum, comm, s));", 
               in0_ptr_name, 
               out0_ptr_name, 
               meta_in0.num_phy_elems);
        // Synchronous Commnunication
        exec.e("cudaStreamSynchronize(s);");
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {

        // Customized op
        kn::KNCustomizedOp const *cur_op =
            dynamic_cast<kn::KNCustomizedOp const *>(op);
        // tb::ExecutionPlan const &plan = cur_op->plan;
        tb::Graph const &bgraph = cur_op->bgraph;
        vector<string> nvshmem_to_free;
        vector<string> nvshmem_as_param;
        vector<string> streams;
        // For epilogue nvshmem allocation
        if (use_nvshmem) {
          for (kn::DTensor const &dtensor : cur_op->output_tensors) {
            DTensorMeta meta = dtensor_metas.at(dtensor.guid);
            if (dtensor.epilogue == type::TBEpilogueType::TB_EPILOGUE_ALLTOALL) {
              //TODO: TB alltoall
              size_t comm_buffer_idx = next_comm_buffer_idx++;
              comm_buffers.e("sizeof($) * $,\t// Buffer $", \
                      get_datatype_str(dtensor.data_type),
                      meta.num_phy_elems,
                      comm_buffer_idx);
              exec.e("$ *alltoall_buf_$ = ($ *)comm_buffers.at($);", \
                      get_datatype_str(dtensor.data_type),
                      dtensor.guid,
                      get_datatype_str(dtensor.data_type),
                      comm_buffer_idx);

              comm_buffer_idx = next_comm_buffer_idx++;
              comm_buffers.e("sizeof(uint64_t),\t// Buffer $", \
                      comm_buffer_idx);
              exec.e("uint64_t *alltoall_signal_$ = (uint64_t *)comm_buffers.at($);", \
                      dtensor.guid, comm_buffer_idx);
              /*
              exec.e("$ *alltoall_buf_$ = ($ *)nvshmem_malloc(sizeof($) * $);", \
                      get_datatype_str(dtensor.data_type),
                      dtensor.guid,
                      get_datatype_str(dtensor.data_type),
                      get_datatype_str(dtensor.data_type),
                      meta.num_phy_elems);
              */
              nvshmem_to_free.push_back(fmt("alltoall_buf_$", dtensor.guid));
              nvshmem_as_param.push_back(fmt("alltoall_buf_$", dtensor.guid));
              nvshmem_as_param.push_back(fmt("alltoall_signal_$", dtensor.guid));
            } else if (dtensor.epilogue == type::TBEpilogueType::TB_EPILOGUE_REDUCESCATTER) {
              //TODO: TB reduce_scatter
              //TODO: support multi-dim gpu mesh
              size_t comm_buffer_idx = next_comm_buffer_idx++;
              comm_buffers.e("sizeof($) * $,\t// Buffer $", \
                      get_datatype_str(dtensor.data_type),
                      meta.num_phy_elems * cur_op->bgraph.gpu_dim.x,
                      comm_buffer_idx);
              exec.e("$ *reduce_scatter_buf_$ = ($ *)comm_buffers.at($);", \
                      get_datatype_str(dtensor.data_type),
                      dtensor.guid,
                      get_datatype_str(dtensor.data_type),
                      comm_buffer_idx);

              comm_buffer_idx = next_comm_buffer_idx++;
              comm_buffers.e("sizeof(uint64_t),\t// Buffer $", \
                      comm_buffer_idx);
              exec.e("uint64_t *reduce_scatter_signal_$ = (uint64_t *)comm_buffers.at($);", \
                      dtensor.guid, comm_buffer_idx);
              /*
              exec.e("$ *reduce_scatter_buf_$ = ($ *)nvshmem_malloc(sizeof($) * $);", \
                      get_datatype_str(dtensor.data_type),
                      dtensor.guid,
                      get_datatype_str(dtensor.data_type),
                      get_datatype_str(dtensor.data_type),
                      meta.num_phy_elems * cur_op->bgraph.gpu_dim.x);
              */
              nvshmem_to_free.push_back(fmt("reduce_scatter_buf_$", dtensor.guid));
              nvshmem_as_param.push_back(fmt("reduce_scatter_buf_$", dtensor.guid));
              nvshmem_as_param.push_back(fmt("reduce_scatter_signal_$", dtensor.guid));
            } else if (dtensor.epilogue == type::TBEpilogueType::TB_EPILOGUE_ALLREDUCE) {
              assert(false && "TB allreduce is not supported yet"); 
              //TODO: TB allreduce
            }
          }
        }
        // Get DTensor ptrs
        // We make the agreement that, when calling a custom kernel, the
        // arguments are in the order of "output_tensors, input_tensors, nvshmem_to_free"
        vector<string> ptr_names;
        for (kn::DTensor const &dtensor :
             Combine(cur_op->output_tensors, cur_op->input_tensors)) {
          auto [ptr_name, ptr_code] = get_dtensor_ptr(dtensor, &comm_buffers, &next_comm_buffer_idx);
          exec.e(ptr_code);
          ptr_names.push_back(ptr_name);
        }
        // For prologue
        for (tb::TBOperator const *tb_op : bgraph.operators) {
          if (tb_op->op_type == mirage::type::TB_INPUT_OP) {
            tb::TBInputOp const *input_op =
                dynamic_cast<tb::TBInputOp const *>(tb_op);
            mirage::kernel::DTensor const *dtensor = &(input_op->dtensor);
            int64_t original_guid = dtensor->original_guid;
            int64_t guid = dtensor->guid;
            DTensorMeta const &meta = dtensor_metas.at(dtensor->guid);
            if (input_op->prologue == mirage::type::TBPrologueType::TB_PROLOGUE_ALLGATHER) {
              //TODO: TB allgather (Jianan)
              size_t comm_buffer_idx = next_comm_buffer_idx++;
              comm_buffers.e("sizeof($) * $,\t// Buffer $", \
                      get_datatype_str(dtensor->data_type),
                      meta.num_phy_elems,
                      comm_buffer_idx);
              exec.e("$ *allgather_buf_$ = ($ *)comm_buffers.at($);", \
                      get_datatype_str(dtensor->data_type),
                      dtensor->guid,
                      get_datatype_str(dtensor->data_type),
                      comm_buffer_idx);
              /*
              exec.e("$ *allgather_buf_$ = ($ *)nvshmem_malloc(sizeof($) * $);", \
                      get_datatype_str(dtensor->data_type),
                      guid,
                      get_datatype_str(dtensor->data_type),
                      get_datatype_str(dtensor->data_type),
                      meta.num_phy_elems);
              */
              // repoint input ptr to allgather_buf
              // TODO: Use tiles instead of whole tensor
              comm_buffer_idx = next_comm_buffer_idx++;
              comm_buffers.e("sizeof(uint64_t) * npes,\t// Buffer $", \
                      comm_buffer_idx);
              exec.e("uint64_t *allgather_signal_$ = (uint64_t *)comm_buffers.at($);", \
                      guid, comm_buffer_idx);
              /*
              exec.e("uint64_t *allgather_signal_$ = (uint64_t*)nvshmem_malloc(sizeof(uint64_t) * npes);",
                      guid);
              */
              exec.e(fmt("cudaStream_t allgather_stream_$;", guid));
              exec.e(fmt("cudaStreamCreate(&allgather_stream_$);", guid));
              streams.push_back(fmt("allgather_stream_$", guid));
              exec.e(fmt("tb::allgather_host<$, $>(allgather_buf_$, dtensor$, allgather_signal_$, $, mype, npes, 0, false, allgather_stream_$);", \
                      get_datatype_str(dtensor->data_type),
                      get_cute_layout(*dtensor, dtensor_metas.at(guid)),
                      guid,
                      original_guid,
                      guid,
                      dtensor_metas.at(original_guid).num_phy_elems,
                      guid));
              exec.e("dtensor$ = allgather_buf_$;",
                     original_guid,
                     guid);
              nvshmem_to_free.push_back(fmt("allgather_buf_$", guid));
              nvshmem_to_free.push_back(fmt("allgather_signal_$", guid));
              nvshmem_as_param.push_back(fmt("allgather_signal_$", guid));
            }
          }
        }

        if (config.profiling) {
          auto [ptr_name, ptr_code] = get_profiling_ptr(0);
          ptr_names.push_back(ptr_name);
          exec.e(ptr_code);
        }

        // Transpile
        CustomOPTranspileResult result;
        if (config.target_cc == GPU_CC::H100) {
          result = transpile_kn_custom_op_hopper(cur_op);
          // only generate for first tb graph now
          config.profiling = false;
        } else {
          result = transpile_kn_custom_op(cur_op);
          config.profiling = false;
        }

        if (result.error_type != CUDA_T_SUCCESS) {
          vector<OutputTensorDirective> output_directives;
          return TranspileResult{
              result.error_type, "", 0, 0, 0, output_directives};
        }
        if (result.smem_size > max_smem_size) {
          max_smem_size = result.smem_size;
        }
        profiler_buf_size += result.profiler_buf_size;

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
        // init

        exec.e("");

        // get tma params;
        if (config.target_cc >= GPU_CC::H100) {
          exec.e("// define tmas");

          // for inputs that needs tma async copy, init the TMAs
          std::string tmas;
          std::string tma_tmps;
          std::string m_inputs;
          for (int i = 0; i < result.tmaParamsList.size(); i++) {
            auto const &tmaParams = result.tmaParamsList.at(i);
            m_inputs.append(tmaParams.m_input ? "true" : "false");

            tmas.append(fmt("tma_$, ", tmaParams.guid));
            tma_tmps.append(fmt("decltype(tma_$)", tmaParams.guid));

            if (i != result.tmaParamsList.size() - 1) {
              tma_tmps.append(", ");
              m_inputs.append(", ");
            }
          }

          exec.e(fmt("std::vector<bool> minputs = {$};", m_inputs));
          for (int i = 0; i < result.tmaParamsList.size(); i++) {
            auto const &tmaParams = result.tmaParamsList.at(i);

            if (tmaParams.m_input) {
              exec.e(fmt("static constexpr cute::GMMA::Major GmmaMajor_$ = "
                         "GMMA::Major::K;",
                         tmaParams.guid));
            } else {
              exec.e(fmt("static constexpr cute::GMMA::Major GmmaMajor_$ = "
                         "GMMA::Major::MN;",
                         tmaParams.guid));
            }
            exec.e(fmt("using DstMNKLayout_$ = $;",
                       tmaParams.guid,
                       tmaParams.dstLayout));

            exec.e(fmt("using SrcMNKLayout_$ = $;",
                       tmaParams.guid,
                       tmaParams.srcLayout));

            exec.e(fmt(
                "using SmemLayoutAtom_$ = "
                "decltype(cutlass::gemm::collective::detail::ss_smem_selector<"
                "GmmaMajor_$, $, decltype(get<0>(DstMNKLayout_${})), "
                "decltype(get<1>(DstMNKLayout_${}))>());",
                tmaParams.guid,
                tmaParams.guid,
                get_datatype_str(cur_op->input_tensors[0].data_type),
                tmaParams.guid,
                tmaParams.guid));
            exec.e(fmt("using DstPipeLayout_$ = "
                       "decltype(tile_to_shape(SmemLayoutAtom_${}, "
                       "make_shape(shape<0>(DstMNKLayout_${}), "
                       "shape<1>(DstMNKLayout_${}), Int<$>{}), "
                       "Step<_1, _2, _3>{}));",
                       tmaParams.guid,
                       tmaParams.guid,
                       tmaParams.guid,
                       tmaParams.guid,
                       config.pipeline_stages));
            exec.e(fmt("auto g_tensor_$ = "
                       "make_tensor(make_gmem_ptr<$>(dtensor$), "
                       "SrcMNKLayout_${});",
                       tmaParams.guid,
                       get_datatype_str(cur_op->input_tensors[0].data_type),
                       tmaParams.guid,
                       tmaParams.guid));
            exec.e(
                fmt("auto tma_$ = make_tma_copy(SM90_TMA_LOAD{}, g_tensor_$, "
                    "DstPipeLayout_${}(_, _, Int<0>{}));",
                    tmaParams.guid,
                    tmaParams.guid,
                    tmaParams.guid));

            exec.e("");
          }

          if (result.tmaParamsList.size() > 0) {
            exec.e("cudaFuncSetAttribute($<$>, "
                   "cudaFuncAttributeMaxDynamicSharedMemorySize, $);",
                   result.func_name,
                   tma_tmps,
                   result.smem_size);
          } else {
            exec.e("cudaFuncSetAttribute($, "
                   "cudaFuncAttributeMaxDynamicSharedMemorySize, $);",
                   result.func_name,
                   result.smem_size);
          }
          if(!use_nvshmem) {
            exec.e("$<<<grid_dim, block_dim, smem_size, stream>>>($ $ $);",
                 result.func_name,
                 tmas,
                 ptr_names,
                 nvshmem_as_param);
          }
          else {
            exec.e("$<<<grid_dim, block_dim, smem_size>>>($ $ $, mype, npes);",
                 result.func_name,
                 tmas,
                 ptr_names,
                 nvshmem_as_param);
          }
        } else {
          exec.e("cudaFuncSetAttribute($, "
                 "cudaFuncAttributeMaxDynamicSharedMemorySize, $);",
                 result.func_name,
                 result.smem_size);
          if(!use_nvshmem) {
            exec.e("$<<<grid_dim, block_dim, smem_size, stream>>>($ $);",
                 result.func_name,
                 ptr_names,
                 nvshmem_as_param);
          }
          else {
            exec.e("$<<<grid_dim, block_dim, smem_size>>>($, $, mype, npes);",
                 result.func_name,
                 ptr_names,
                 nvshmem_as_param);
          }
        }

        custom_kernels.e(result.code);
        if (use_nvshmem) {
          // copy result from comm_buf to dtensor
          // TODO: assuming only one output tensor and one comm_buf
          // TODO: handle commbufs including prologue and epilogue
          kn::DTensor const &output_dtensor = cur_op->output_tensors[0];
          auto output_guid = output_dtensor.guid;
          DTensorMeta const &output_meta = dtensor_metas.at(output_guid);
          if (!nvshmem_as_param.empty() && nvshmem_as_param[0].find("alltoall") != string::npos) {
          
            for (auto& nvshmem_param : nvshmem_as_param) {
              if (nvshmem_param.find("signal") != std::string::npos) {
                exec.e("nvshmemx_uint64_wait_until_on_stream($, NVSHMEM_CMP_EQ,  $ * $ * $, stream);",
                       nvshmem_param,
                       num_gpus,
                       bgraph.grid_dim.y,
                       bgraph.grid_dim.z);
              }
            }
            exec.e("cudaMemcpy((void *)output_tensors.at(0), "
                  "(const void *)nvshmem_ptr($, mype), "
                  "$ * sizeof($), "
                  "cudaMemcpyDeviceToDevice);", 
                  nvshmem_as_param[0], 
                  output_meta.num_phy_elems, 
                  get_datatype_str(output_dtensor.data_type));
           }
           // TODO: Inconsistency between nvshmem_as_param and comm_buf_names may still need to check
           // TODO (linsj20) 
          else if (output_dtensor.epilogue == type::TBEpilogueType::TB_EPILOGUE_REDUCESCATTER) {
            /*
            for (kn::KNOperator *_op : g->operators) {
              if (op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
                kn::KNCustomizedOp *tmp_op =
                    dynamic_cast<kn::KNCustomizedOp *>(_op);
                if (tmp_op->output_tensors[0].guid == output_guid) {
                  kn::DTensor &tmp_dtensor = tmp_op->output_tensors[0];
                  DTensorMeta &tmp_meta = dtensor_metas.at(output_guid);
                  tmp_dtensor.dim[1] /= cur_op->bgraph.gpu_dim.x;
                  for (int i = 1; i < kernel::MAX_TENSOR_DIMS; i++) {
                    tmp_meta.strides[i] /= cur_op->bgraph.gpu_dim.x;
                  }
                }
              }
            }
            */

            auto dims = vector<int>(output_dtensor.dim, output_dtensor.dim + output_dtensor.num_dims);
            //TODO support reduce_scatter on other dims
            dims[0] *= cur_op->bgraph.gpu_dim.x;
            for (auto& nvshmem_param : nvshmem_as_param) {
              if (nvshmem_param.find("signal") != std::string::npos) {
                exec.e("nvshmemx_uint64_wait_until_on_stream($, NVSHMEM_CMP_EQ,  $ * $ * $, stream);",
                       nvshmem_param,
                       num_gpus,
                       bgraph.grid_dim.y,
                       bgraph.grid_dim.z);
              }
            }
            exec.e("using reduction_kernel = kn::ReductionKernel<$, $, $, 1, true>;",
                   get_datatype_str(output_dtensor.data_type),
                   get_cute_layout(dims,
                                   vector<size_t>(output_meta.strides,
                                                  output_meta.strides + output_dtensor.num_dims)),
                   get_cute_layout(output_dtensor, output_meta));
            // Assuming the first param is the buffer
            if (output_meta.is_output) {
              exec.e("reduction_kernel::run(($*)output_tensors.at($), $);",
                    get_datatype_str(output_dtensor.data_type),
                    output_meta.output_idx,
                    nvshmem_as_param[0]);
            } else {
              exec.e("reduction_kernel::run(($*)dtensor$, $);",
                    get_datatype_str(output_dtensor.data_type),
                    output_dtensor.guid,
                    nvshmem_as_param[0]);
            }
          }

          // Free nvshmem allocated memory
          /*
          for (kn::DTensor const &dtensor :
               Combine(cur_op->output_tensors, cur_op->input_tensors)) {
            auto guid = dtensor.guid;
            DTensorMeta const &meta = dtensor_metas.at(guid);
            if (meta.is_output && dtensor.is_nvshmem_tensor) {
              string ptr_name = fmt("dtensor$", guid);
              exec.e("nvshmem_free($);", ptr_name);
            }
          }
          for (auto const &comm_buf_name : nvshmem_to_free) {
            std::cout << "freeing " << comm_buf_name << std::endl;
            exec.e("nvshmem_free($);", comm_buf_name);
          }
          */

          // Free streams
          for (auto const &stream_name : streams) {
            exec.e("cudaStreamDestroy($);", stream_name);
          }
        }

        break;
      }
      default:
        assert(false && ("Unsupported operator type: " +
                         std::to_string(int(op->op_type)))
                            .c_str());
    }
    exec.e("}");
  }

  if (use_nccl) {
    exec.e("// Finalize NCCL and MPI");
    exec.e("ncclCommDestroy(comm);");
    exec.e("cudaStreamDestroy(s);");
    //exec.e("MPI_Finalize();");
  }
  else if (use_nvshmem) {
    //exec.e("finalize_mpi_nvshmem();");
  }

  init.e("}");
  exec.e("}");
  comm_buffers.e("};");
  comm_buffers.e("}");
  string code;
  if (use_nvshmem) {
    code = fmt("$\n$\n$\n$\n$\n$\n",
                      header.to_string(),
                      custom_kernels.to_string(),
                      comm_buffers.to_string(),
                      init.to_string(),
                      hopper_tma.to_string(),
                      exec.to_string());
  } else {
    code = fmt("$\n$\n$\n$\n$\n",
                      header.to_string(),
                      custom_kernels.to_string(),
                      init.to_string(),
                      hopper_tma.to_string(),
                      exec.to_string());
  }
  vector<OutputTensorDirective> output_directives;
  for (kn::DTensor const &dtensor : this->mugraph_output_tensors) {
    assert(dtensor_metas.find(dtensor.guid) != dtensor_metas.end());

    DTensorMeta meta = dtensor_metas.at(dtensor.guid);
    output_directives.push_back(OutputTensorDirective{
        meta.num_phy_elems,
        vector<int>(dtensor.dim, dtensor.dim + dtensor.num_dims),
        vector<size_t>(meta.strides, meta.strides + dtensor.num_dims)});
  }

  return TranspileResult{CUDA_T_SUCCESS,
                         code,
                         this->d_buf_size,
                         max_smem_size,
                         profiler_buf_size,
                         output_directives};
}

} // namespace transpiler
} // namespace mirage