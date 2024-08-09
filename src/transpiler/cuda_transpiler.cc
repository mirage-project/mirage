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

#include "mirage/transpiler/cuda_transpiler.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/serializer/concat_serializer.h"
#include "mirage/threadblock/serializer/element_binary_serializer.h"
#include "mirage/threadblock/serializer/element_unary_serializer.h"
#include "mirage/threadblock/serializer/forloop_accum_serializer.h"
#include "mirage/threadblock/serializer/input_loader_serializer.h"
#include "mirage/threadblock/serializer/matmul_serializer.h"
#include "mirage/threadblock/serializer/output_saver_serializer.h"
#include "mirage/threadblock/serializer/reduction_serializer.h"

#include <fstream>
#include <iostream>
#include <map>

namespace mirage {
namespace transpiler {

// We need to specialize the to_string function for char* type, so we have
// my_to_string function here
template <typename T>
static std::string my_to_string(T const &value) {
  return std::to_string(value);
}

template <>
[[maybe_unused]] std::string my_to_string(char const *const &value) {
  return value;
}

template <>
[[maybe_unused]] std::string my_to_string(std::string const &value) {
  return value;
}

// A function that takes a template string and a list of elements
// It replaces each marker in the template string with the corresponding element
// and returns the resulting string. We can use this to generate code elegantly
template <typename... Args>
static std::string fmt(std::string const &fmt_str, Args... args) {
  std::string result = fmt_str;
  int num_args = sizeof...(args);
  int num_markers = std::count(result.begin(), result.end(), '$');
  if (num_args != num_markers) {
    std::cerr << "Error encountered during transpiling.";
    std::cerr << "The number of arguments does not match the number of markers "
                 "in the template string.";
    std::cerr << "fmt_str: " << fmt_str << std::endl;
    std::cerr << "args: ";
    ((std::cerr << args << " "), ...);
    std::cerr << std::endl;
    std::cerr << "num_args: " << num_args << std::endl;
    std::cerr << "num_markers: " << num_markers << std::endl;
    exit(1);
  }
  (result.replace(result.find("$"), 1, my_to_string(args)), ...);
  return result;
}

// A helper class for keeping all generated code, and provide some utility
// functions
class CodeKeeper {
private:
  std::vector<std::string> lines;

public:
  // Emit a new line
  template <typename... Args>
  void e(std::string const &fmt_str, Args... args) {
    lines.push_back(fmt(fmt_str, args...));
  }

  // Emit a new line
  void operator<<(std::string const &line) {
    lines.push_back(line);
  }

  // Merge two CodeKeeper objects
  friend void operator<<(CodeKeeper &target, CodeKeeper const &source) {
    for (auto const &line : source.lines) {
      target.lines.push_back(line);
    }
  }

  // Return the generated code as a string
  std::string to_string() const {
    std::string result;
    for (auto const &line : lines) {
      result += line + "\n";
    }
    return result;
  }
};

std::string CudaTranspiler::generate_header_code(std::string indent) {
  CodeKeeper ck;
  ck.e("#define USE_NVSHMEM $", use_nvshmem);
  ck.e("#include \"runtime.h\"");
  return ck.to_string();
}

std::string gen_cuda_block_offset_calculation(int off_x,
                                              int off_y,
                                              int off_z,
                                              int off_i = 0) {
  if (off_x == 0 && off_y == 0 && off_z == 0 && off_i == 0) {
    return "0";
  }
  std::stringstream ret;
  if (off_x > 0) {
    ret << off_x << " * blockIdx.x";
  }
  if (off_y > 0) {
    if (off_x > 0) {
      ret << " + ";
    }
    ret << off_y << " * blockIdx.y";
  }
  if (off_z > 0) {
    if (off_x > 0 || off_y > 0) {
      ret << " + ";
    }
    ret << off_z << " * blockIdx.z";
  }
  if (off_i > 0) {
    if (off_x > 0 || off_y > 0 || off_z > 0) {
      ret << " + ";
    }
    ret << off_i << " * i";
  }
  return ret.str();
}

void CudaTranspiler::define_stensor_from_offset(std::stringstream &ss,
                                                int offset,
                                                std::string name,
                                                std::string ind,
                                                mirage::type::DataType type) {
  switch (type) {
    case mirage::type::DT_FLOAT16: {
      ss << ind << "cutlass::half_t *" << name << " =\n"
         << ind << "    (cutlass::half_t *)(";
      if (input_loader_smem_offsets.find(offset) !=
          input_loader_smem_offsets.end()) {
        ss << "smem_buffer + (i % 2) * " << bgraph->smem_offset << " + "
           << offset;
      } else {
        ss << "smem_buffer + " << offset;
      }
      ss << ");\n";
      break;
    }
    default:
      assert(false && "Unsupported data type");
  }
}

void CudaTranspiler::gen_cuda_code_input_loader(std::string dtensor_name,
                                                std::string ind) {
  int3 input_matrix_row_offset_block_stride;
  int3 input_matrix_column_offset_block_stride;
  int input_matrix_row_offset_forloop_stride;
  int input_matrix_column_offset_forloop_stride;
  int3 global_offset_block_stride;
  int global_offset_forloop_stride;
  int2 dtensor_matrix_shape, stensor_matrix_shape;
  int input_smem_offset;
  mirage::layout::DmemLayout dtensor_layout;
  mirage::layout::SmemLayout stensor_layout;
  mirage::threadblock::deserialize_input_loader_parameters(
      params.parameters,
      param_idx,
      input_matrix_row_offset_block_stride,
      input_matrix_column_offset_block_stride,
      input_matrix_row_offset_forloop_stride,
      input_matrix_column_offset_forloop_stride,
      global_offset_block_stride,
      global_offset_forloop_stride,
      dtensor_matrix_shape,
      stensor_matrix_shape,
      dtensor_layout,
      stensor_layout,
      input_smem_offset);
  // Add input_smem_offset into input_loader_smem_offsets
  input_loader_smem_offsets.insert(input_smem_offset);

  input_loader_func << ind << "int tb_offset_row = "
                    << gen_cuda_block_offset_calculation(
                           input_matrix_row_offset_block_stride.x,
                           input_matrix_row_offset_block_stride.y,
                           input_matrix_row_offset_block_stride.z,
                           input_matrix_row_offset_forloop_stride)
                    << ";\n";
  input_loader_func << ind << "int tb_offset_column = "
                    << gen_cuda_block_offset_calculation(
                           input_matrix_column_offset_block_stride.x,
                           input_matrix_column_offset_block_stride.y,
                           input_matrix_column_offset_block_stride.z,
                           input_matrix_column_offset_forloop_stride)
                    << ";\n";
  input_loader_func << ind << "int global_offset = "
                    << gen_cuda_block_offset_calculation(
                           global_offset_block_stride.x,
                           global_offset_block_stride.y,
                           global_offset_block_stride.z,
                           global_offset_forloop_stride)
                    << ";\n";
  // input_loader_func << ind << "cutlass::MatrixCoord matrix_offset"
  //                   << " = {tb_offset_row, tb_offset_column};\n";
  input_loader_func << ind << "cutlass::half_t *stensor_ptr =\n"
                    << ind << "    (cutlass::half_t*)(smem_buffer + (i % 2) * "
                    << bgraph->smem_offset << " + " << input_smem_offset
                    << ");\n";

  int kRow = stensor_matrix_shape.x;
  int kColumn = stensor_matrix_shape.y;
  if (kColumn == dtensor_matrix_shape.y) {
    input_loader_func << ind
                      << "int base_offset = global_offset + tb_offset_row * "
                      << dtensor_matrix_shape.y << " + tb_offset_column;\n";
  } else {
    assert(kRow == dtensor_matrix_shape.x);
    input_loader_func << ind
                      << "int base_offset = global_offset + tb_offset_column * "
                      << dtensor_matrix_shape.x << " + tb_offset_row;\n";
  }
  // Each thread loads 16 bytes using cp.async
  input_loader_func << ind << "for (int _idx = threadIdx.x * 8; _idx < "
                    << kRow * kColumn << "; _idx += 8 * blockDim.x) {\n";
  input_loader_func
      << ind << "  unsigned stensor_int_ptr =\n"
      << ind
      << "      cutlass::arch::cutlass_get_smem_pointer(stensor_ptr + _idx);\n";
  input_loader_func << ind
                    << "  cutlass::half_t *_dtensor_ptr = " << dtensor_name
                    << " + base_offset + _idx;\n";
  input_loader_func
      << ind << "  asm volatile(\n"
      << ind << "      "
      << "\"cp.async.ca.shared.global.L2::128B [\%0], [\%1], \%2, \%3;\\n\"::"
      << "\n"
      << ind << "      \"r\"(stensor_int_ptr),\n"
      << ind << "      \"l\"(_dtensor_ptr),\n"
      << ind << "      \"n\"(16),\n"
      << ind << "      \"r\"(16));\n";
  input_loader_func << ind << "} // end of for-loop\n";
}

void CudaTranspiler::gen_cuda_code_forloop_accum(std::string ind) {
  int num_elements;
  int input_smem_offset, accum_smem_offset;
  mirage::threadblock::deserialize_forloop_accum_parameters(
      params.parameters,
      param_idx,
      num_elements,
      input_smem_offset,
      accum_smem_offset);
  // FIXME: currently we do nothing for forloop accumulation
  // since we expect it to be implemented as an epilogue of
  // the previous operator
}

void CudaTranspiler::gen_cuda_code_output_saver(std::string dtensor_name,
                                                std::string ind) {
  int3 output_matrix_row_offset_block_stride;
  int3 output_matrix_column_offset_block_stride;
  int output_matrix_row_offset_forloop_stride;
  int output_matrix_column_offset_forloop_stride;
  int3 global_offset_block_stride;
  int global_offset_forloop_stride;
  int2 dtensor_matrix_shape, stensor_matrix_shape;
  int input_smem_offset;
  mirage::layout::DmemLayout dtensor_layout;
  mirage::layout::SmemLayout stensor_layout;
  mirage::type::TBEpilogueType epilogue;
  mirage::threadblock::deserialize_output_saver_parameters(
      params.parameters,
      param_idx,
      output_matrix_row_offset_block_stride,
      output_matrix_column_offset_block_stride,
      output_matrix_row_offset_forloop_stride,
      output_matrix_column_offset_forloop_stride,
      global_offset_block_stride,
      global_offset_forloop_stride,
      dtensor_matrix_shape,
      stensor_matrix_shape,
      dtensor_layout,
      stensor_layout,
      input_smem_offset,
      epilogue);
  std::stringstream code;
  code << ind << "int tb_offset_row = "
       << gen_cuda_block_offset_calculation(
              output_matrix_row_offset_block_stride.x,
              output_matrix_row_offset_block_stride.y,
              output_matrix_row_offset_block_stride.z)
       << ";\n";
  code << ind << "int tb_offset_column = "
       << gen_cuda_block_offset_calculation(
              output_matrix_column_offset_block_stride.x,
              output_matrix_column_offset_block_stride.y,
              output_matrix_column_offset_block_stride.z)
       << ";\n";
  code << ind << "int global_offset = "
       << gen_cuda_block_offset_calculation(global_offset_block_stride.x,
                                            global_offset_block_stride.y,
                                            global_offset_block_stride.z)
       << ";\n";
  code << ind << "cutlass::half_t *stensor_ptr =\n"
       << ind << "    (cutlass::half_t*)(smem_buffer + " << input_smem_offset
       << ");\n";
  int kRow = stensor_matrix_shape.x;
  int kColumn = stensor_matrix_shape.y;
  printf("stensor_matrix(%d %d) dtensor_matrix(%d %d)\n",
         kRow,
         kColumn,
         dtensor_matrix_shape.x,
         dtensor_matrix_shape.y);
  code << ind << "int base_offset = global_offset + tb_offset_row * "
       << dtensor_matrix_shape.y << " + tb_offset_column;\n";

  // FIXME: currently assume a row-major layout for both stensor and dtensor
  code << ind << "cutlass::half_t *dtensor_ptr = " << dtensor_name
       << " + base_offset;\n";
  // Currently assume that kColumn can divide blockDim.x
  assert(128 % kColumn == 0);
  int num_rows_per_iter = 128 / kColumn;
  code << ind << "int _col = threadIdx.x % " << kColumn << ";\n";
  code << ind << "for (int _row = threadIdx.x / " << num_rows_per_iter
       << "; _row < " << kRow << "; _row += " << num_rows_per_iter << ") {\n";
  code << ind << "  dtensor_ptr[_row * " << dtensor_matrix_shape.y << " + _col"
       << "] = stensor_ptr[_row * " << kColumn << " + _col];\n";
  code << ind << "} // end of for-loop\n";
  // We generate a nvshmem collective communication operator
  if (use_nvshmem) {
    switch (epilogue) {
      case mirage::type::TB_EPILOGUE_NONE: {
        // do nothing when epilogue is none
        break;
      }
      case mirage::type::TB_EPILOGUE_ALLREDUCE: {
        code << ind << "nvshmemx_int16_sum_reduce_block(\n"
             << ind << "    NVSHMEM_TEAM_WORLD,\n"
             << ind << "    dtensor_ptr,\n"
             << ind << "    dtensor_ptr,\n"
             << ind << "    " << kRow * kColumn << ");\n";
        break;
      }
      default: {
        assert(false && "Unsupported epilogue");
      }
    }
  }
  ending << code.str();
}

void CudaTranspiler::gen_cuda_code_matmul_op(std::string ind) {
  int m, n, k;
  int A_smem_offset, B_smem_offset, C_smem_offset;
  mirage::threadblock::deserialize_matmul_op_parameters(params.parameters,
                                                        param_idx,
                                                        m,
                                                        n,
                                                        k,
                                                        A_smem_offset,
                                                        B_smem_offset,
                                                        C_smem_offset);
  define_stensor_from_offset(main, A_smem_offset, "_A_ptr", ind);
  define_stensor_from_offset(main, B_smem_offset, "_B_ptr", ind);
  define_stensor_from_offset(main, C_smem_offset, "_C_ptr", ind);
  if (op < params.num_operators &&
      params.operator_types[op + 1] == mirage::type::TB_EXP_OP) {
    // fuse this matmul with the next exp
    int E_smem_offset, num_elements;
    mirage::threadblock::deserialize_elementunary_op_parameters(
        params.parameters, param_idx, E_smem_offset, num_elements);
    // assert inline exp
    assert(C_smem_offset == E_smem_offset);
    main << ind
         << "//"
            "mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_EXP> "
            "executor(\n"
         << ind << "//_A_ptr, _B_ptr, _C_ptr, " << m << ", " << n << ", " << k
         << ", threadIdx.x);\n";

    op += 1;
  } else {
    main << ind
         << "//"
            "mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_NONE>"
            " executor(\n"
         << ind << "//_A_ptr, _B_ptr, _C_ptr, " << m << ", " << n << ", " << k
         << ", threadIdx.x);\n";
  }
}

void CudaTranspiler::gen_cuda_code_exp_op(std::string ind) {
  int smem_offset, num_elements;
  mirage::threadblock::deserialize_elementunary_op_parameters(
      params.parameters, param_idx, smem_offset, num_elements);
  define_stensor_from_offset(main, smem_offset, "_ptr", ind);
}

void CudaTranspiler::gen_cuda_code_div_op(std::string ind) {
  int3 input1_shape, input2_shape;
  int input1_smem_offset, input2_smem_offset, output_smem_offset;
  mirage::threadblock::deserialize_elementbinary_op_parameters(
      params.parameters,
      param_idx,
      input1_shape,
      input2_shape,
      input1_smem_offset,
      input2_smem_offset,
      output_smem_offset);

  define_stensor_from_offset(main, input1_smem_offset, "_in1_ptr", ind);
  define_stensor_from_offset(main, input2_smem_offset, "_in2_ptr", ind);
  define_stensor_from_offset(main, output_smem_offset, "_out_ptr", ind);
  // FIXME: Currently we assume broadcast the inner-most dim
  int3 output_shape = {std::max(input1_shape.x, input2_shape.x),
                       std::max(input1_shape.y, input2_shape.y),
                       std::max(input1_shape.z, input2_shape.z)};
  int output_num_elements = output_shape.x * output_shape.y * output_shape.z;
  int input1_num_elements = input1_shape.x * input1_shape.y * input1_shape.z;
  int input2_num_elements = input2_shape.x * input2_shape.y * input2_shape.z;
  int factor1 = output_num_elements / input1_num_elements;
  int factor2 = output_num_elements / input2_num_elements;
  main << ind << "for (int i = 0; i < " << output_num_elements
       << "; i += blockDim.x) {\n"
       << ind << "  _out_ptr[i] = _in1_ptr[i / " << factor1 << "] / "
       << "_in2_ptr[i / " << factor2 << "];\n"
       << ind << "}\n";
}

void CudaTranspiler::gen_cuda_code_reduction_op(std::string ind) {
  int output_num_elements, reduction_degree, inner_range;
  int input_smem_offset, output_smem_offset;
  mirage::threadblock::deserialize_reduction_op_parameters(params.parameters,
                                                           param_idx,
                                                           output_num_elements,
                                                           reduction_degree,
                                                           inner_range,
                                                           input_smem_offset,
                                                           output_smem_offset);
}

void gen_cuda_code_launch_device_func(std::stringstream &ss,
                                      std::string func_name,
                                      std::string prefix_arg,
                                      std::vector<std::string> arg_names,
                                      std::string ind) {
  ss << ind << func_name << "(\n";
  ss << ind << "    " << prefix_arg << ",\n";
  for (size_t i = 0; i < arg_names.size(); i++) {
    if (i > 0) {
      ss << ",\n";
    }
    ss << ind << "    " << arg_names[i];
  }
  ss << ");\n";
}

std::string CudaTranspiler::generate_kernel_code(
    mirage::threadblock::NewKernelParams _params,
    mirage::threadblock::Graph const *_bgraph,
    std::string func_name,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    std::string ind) {
  params = _params;
  bgraph = _bgraph;
  using namespace mirage::threadblock;
  using namespace std;
  string input_loader_func_name =
      "smem_input_loader_func_" + func_name.substr(func_name.length() - 1, 1);
  input_loader_func.str("");
  {
    input_loader_func << "__device__ void " << input_loader_func_name << "(\n";
    input_loader_func << "    int i,\n";
    for (size_t i = 0; i < input_names.size(); i++) {
      if (i > 0) {
        input_loader_func << ",\n";
      }
      input_loader_func << "    cutlass::half_t *" << input_names[i];
    }
    input_loader_func << ") {\n";
    input_loader_func << "  extern __shared__ char smem_buffer[];\n";
  }
  output_saver_func.str("");
  header.str("");
  main.str("");
  ending.str("");
  input_loader_smem_offsets.clear();
  {
    header << "__global__ void " << func_name << "(\n";
    for (size_t i = 0; i < input_names.size(); i++) {
      if (i > 0) {
        header << ",\n";
      }
      header << "    cutlass::half_t *" << input_names[i];
    }
    for (size_t i = 0; i < output_names.size(); i++) {
      header << ",\n";
      header << "    cutlass::half_t *" << output_names[i];
    }
    header << ") {\n";
  }
  header << ind << "extern __shared__ char smem_buffer[];\n";
  gen_cuda_code_launch_device_func(
      main, input_loader_func_name, "0", input_names, ind);
  main << ind << "for (int i = 0; i < " << bgraph->forloop_range
       << "; i++) {\n";
  // increase the indent by 2 spaces since we have a for loop wrapper
  ind = ind + "  ";
  main << ind << "// launch cp.async operators\n";
  main << ind << "if (i + 1 < " << bgraph->forloop_range << ") {\n";
  gen_cuda_code_launch_device_func(
      main, input_loader_func_name, "i + 1", input_names, ind + "  ");
  main << ind << "  asm volatile(\"cp.async.wait_group 1;\\n\" ::);\n";
  main << ind << "} else {\n";
  main << ind << "  asm volatile(\"cp.async.wait_group 0;\\n\" ::);\n";
  main << ind << "}\n";
  param_idx = 0;
  int output_idx = 0;
  for (op = 0; op < params.num_operators; op++) {
    mirage::type::TBOperatorType op_type = params.operator_types[op];
    switch (op_type) {
      case mirage::type::TB_INPUT_OP: {
        input_loader_func
            << ind.substr(0, ind.length() - 2)
            << "// Load input tensor from device to shared memory\n";
        input_loader_func << ind.substr(0, ind.length() - 2) << "{\n";
        gen_cuda_code_input_loader(input_names[op], ind);
        input_loader_func << ind.substr(0, ind.length() - 2) << "}\n";
        break;
      }
      case mirage::type::TB_OUTPUT_OP: {
        ending << ind.substr(0, ind.length() - 2)
               << "// Save output tensor from shared to device memory\n";
        ending << ind.substr(0, ind.length() - 2) << "{\n";
        gen_cuda_code_output_saver(output_names[output_idx++], ind);
        ending << ind.substr(0, ind.length() - 2) << "}\n";
        break;
      }
      case mirage::type::TB_FORLOOP_ACCUM_OP: {
        gen_cuda_code_forloop_accum(ind);
        break;
      }
      case mirage::type::TB_MATMUL_OP: {
        main << ind << "// Perform thread-block matmul\n";
        main << ind << "{\n";
        gen_cuda_code_matmul_op(ind + "  ");
        main << ind << "}\n";
        break;
      }
      case mirage::type::TB_EXP_OP: {
        main << ind << "// Perform thread-block elementwise exp\n";
        main << ind << "{\n";
        gen_cuda_code_exp_op(ind + "  ");
        main << ind << "}\n";
        break;
      }
      case mirage::type::TB_DIV_OP: {
        main << ind << "// Perform thread-block elementwise div\n";
        main << ind << "{\n";
        gen_cuda_code_div_op(ind + "  ");
        main << ind << "}\n";
        break;
      }
      case mirage::type::TB_REDUCTION_0_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_1_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_2_TO_DIMX_OP: {
        main << ind << "// Perform thread-block elementwise reduction\n";
        main << ind << "{\n";
        gen_cuda_code_reduction_op(ind + "  ");
        main << ind << "}\n";
        break;
      }
      case mirage::type::TB_REDUCTION_0_OP:
      case mirage::type::TB_REDUCTION_1_OP:
      case mirage::type::TB_REDUCTION_2_OP: {
        main << ind << "// Perform thread-block elementwise reduction\n";
        main << ind << "{\n";
        gen_cuda_code_reduction_op(ind + "  ");
        main << ind << "}\n";
        break;
      }
      default: {
        assert(false && "Unsupported op_type");
      }
    }
  }
  // Recover the original indent
  ind = ind.substr(0, ind.length() - 2);
  // Add commit_group at the end of the input_loader
  input_loader_func << "  asm volatile(\"cp.async.commit_group;\\n\" ::);\n";
  input_loader_func << "} // end of " << input_loader_func_name << "\n\n";
  main << ind << "} // end of for-loop\n";
  // End of kernel
  ending << "}\n";
  assert(params.num_parameters == param_idx);
  return input_loader_func.str() + header.str() + main.str() + ending.str();
}

CudaTranspiler::CudaTranspiler(bool _nvshmem) : use_nvshmem(_nvshmem) {}

}; // namespace transpiler
}; // namespace mirage

namespace mirage {
namespace kernel {

void Graph::generate_cuda_program(char const *file_path) {
  using namespace std;
  using mirage::transpiler::CodeKeeper;
  using mirage::transpiler::fmt;

  vector<string> kernels;
  bool use_nvshmem = (gpu_dim.x > 1 || gpu_dim.y > 1 || gpu_dim.z > 1);
  mirage::transpiler::CudaTranspiler ct(use_nvshmem);

  CodeKeeper kernel_init; // e.g. cudaFuncSetAttribute
  CodeKeeper executor;
  executor.e("void mugraph_executor(char *gpu_base_ptr) {");
  for (KNOperator *const op : this->operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      assert(op->output_tensors.size() == 1);

    } else if (op->op_type == type::KNOperatorType::KN_ADD_OP ||
               op->op_type == type::KNOperatorType::KN_MUL_OP ||
               op->op_type == type::KNOperatorType::KN_DIV_OP) {
      assert(op->input_tensors.size() == 2);
      assert(op->output_tensors.size() == 1);
      string output_ptr = fmt("dtensor$", op->output_tensors[0].guid);
      string input1_ptr = fmt("dtensor$", op->input_tensors[0].guid);
      string input2_ptr = fmt("dtensor$", op->input_tensors[1].guid);
      size_t numel = op->output_tensors[0].num_elements();
      string kernel_name =
          op->op_type == type::KNOperatorType::KN_ADD_OP   ? "ElementAddKernel"
          : op->op_type == type::KNOperatorType::KN_MUL_OP ? "ElementMulKernel"
          : op->op_type == type::KNOperatorType::KN_DIV_OP
              ? "ElementDivKernel"
              : "UnknownElementwiseKernel";
      executor.e("  {");
      executor.e(
          "    cutlass::half_t *$ = (cutlass::half_t*)(gpu_base_ptr + $);",
          output_ptr,
          op->output_tensors[0].data_offset);
      executor.e(
          "    cutlass::half_t *$ = (cutlass::half_t*)(gpu_base_ptr + $);",
          input1_ptr,
          op->input_tensors[0].data_offset);
      executor.e(
          "    cutlass::half_t *$ = (cutlass::half_t*)(gpu_base_ptr + $);",
          input2_ptr,
          op->input_tensors[1].data_offset);
      executor.e("    $<cutlass::half_t>::run($, $, $, $);",
                 kernel_name,
                 output_ptr,
                 input1_ptr,
                 input2_ptr,
                 numel);
      executor.e("  }");

    } else if (op->op_type == type::KNOperatorType::KN_EXP_OP) {
      assert(op->input_tensors.size() == 1);
      assert(op->output_tensors.size() == 1);
      string output_ptr = fmt("dtensor$", op->output_tensors[0].guid);
      string input_ptr = fmt("dtensor$", op->input_tensors[0].guid);
      size_t numel = op->output_tensors[0].num_elements();
      executor.e("  {");
      executor.e(
          "    cutlass::half_t *$ = (cutlass::half_t*)(gpu_base_ptr + $);",
          output_ptr,
          op->output_tensors[0].data_offset);
      executor.e(
          "    cutlass::half_t *$ = (cutlass::half_t*)(gpu_base_ptr + $);",
          input_ptr,
          op->input_tensors[0].data_offset);
      executor.e("    ElementExpKernel<cutlass::half_t>::run($, $, $);",
                 output_ptr,
                 input_ptr,
                 numel);
      executor.e("  }");

    } else if (op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      KNCustomizedOp const *customized =
          static_cast<KNCustomizedOp const *>(op);
      mirage::threadblock::NewKernelParams params =
          customized->bgraph.get_new_kernel_params(false /*fingerprint*/);
      string kernel_name = "graphdef_kernel_" + to_string(kernels.size());
      vector<string> input_names;
      vector<string> output_names;
      for (auto const &t : op->input_tensors) {
        input_names.push_back("dtensor" + to_string(t.guid));
      }
      for (auto const &t : op->output_tensors) {
        output_names.push_back("dtensor" + to_string(t.guid));
      }
      string kernel_code = ct.generate_kernel_code(params,
                                                   &(customized->bgraph),
                                                   kernel_name,
                                                   input_names,
                                                   output_names,
                                                   "  " /*indent*/);
      kernels.push_back(kernel_code);
      int smem_size = customized->bgraph.get_smem_size_with_pipeline();
      if (smem_size > 48 * 1024) {
        kernel_init.e("  CHECK_CUDA(cudaFuncSetAttribute($, "
                      "cudaFuncAttributeMaxDynamicSharedMemorySize, $)",
                      kernel_name,
                      smem_size);
      }
      executor.e("  // launching kernel: $", kernel_name);
      executor.e("  {");
      for (auto const &t : op->input_tensors) {
        executor.e("    cutlass::half_t *dtensor$ = "
                   "(cutlass::half_t*)(gpu_base_ptr + $);",
                   t.guid,
                   t.data_offset);
      }
      for (auto const &t : op->output_tensors) {
        executor.e("    cutlass::half_t *dtensor$ = "
                   "(cutlass::half_t*)(gpu_base_ptr + $);",
                   t.guid,
                   t.data_offset);
      }
      executor.e("    dim3 grid_dim = {$, $, $}",
                 customized->bgraph.grid_dim.x,
                 customized->bgraph.grid_dim.y,
                 customized->bgraph.grid_dim.z);
      executor.e("    dim3 block_dim = {$, $, $}",
                 customized->bgraph.block_dim.x,
                 customized->bgraph.block_dim.y,
                 customized->bgraph.block_dim.z);
      executor.e("    $<<<grid_dim, block_dim, $>>>(", kernel_name, smem_size);
      string input_names_line = "";
      for (auto const &t : op->input_tensors) {
        if (!input_names_line.empty()) {
          input_names_line += ", ";
        }
        input_names_line += fmt("dtensor$", t.guid);
      }
      string output_names_line = "";
      for (auto const &t : op->output_tensors) {
        if (!output_names_line.empty() || !input_names_line.empty()) {
          output_names_line += ", ";
        }
        output_names_line += fmt("dtensor$", t.guid);
      }
      executor.e("      $", input_names_line);
      executor.e("      $", output_names_line);
      executor.e("    );");
      executor.e("  }");

    } else {
      assert(false && ("Cannot generate CUDA operator for this operator: " +
                       to_string(int(op->op_type)))
                          .c_str());
    }
  }
  executor.e("} // end of mugraph_executer");

  CodeKeeper main;
  main.e("int main() {");
  assert(gpu_dim.y == 1);
  assert(gpu_dim.z == 1);
  main.e("  char *gpu_base_ptrs[$];", mirage::config::MAX_NUM_GPUS);
  main.e("  for (int i = 0; i < $; i++) {", gpu_dim.x);
  main.e("    CHECK_CUDA(cudaSetDevice(i));");
  if (use_nvshmem) {
    main.e("    gpu_base_ptrs[i] = nvshmem_malloc($);",
           mirage::config::MAX_DMEM_DATA_SIZE);
  } else {
    main.e("    CHECK_CUDA(cudaMalloc(&gpu_base_ptrs[i], $));",
           mirage::config::MAX_DMEM_DATA_SIZE);
  }
  main.e("  }");

  main.e("  CHECK_CUDA(cudaDeviceSynchronize());");
  main.e("  cudaEvent_t events[2];");
  main.e("  CHECK_CUDA(cudaEventCreate(&events[0]));");
  main.e("  CHECK_CUDA(cudaEventCreate(&events[1]));");
  main.e("  for (int i = 0; i < 1024; i++) {");
  main.e("    mugraph_executor(gpu_base_ptrs[0]);");
  main.e("  }");
  main.e("  CHECK_CUDA(cudaEventRecord(events[0]));");
  main.e("  for (int i = 0; i < 1024; i++) {");
  main.e("    mugraph_executor(gpu_base_ptrs[0]);");
  main.e("  }");
  main.e("  CHECK_CUDA(cudaEventRecord(events[1]));");
  main.e("  CHECK_CUDA(cudaEventSynchronize(events[1]));");
  main.e("  float runtime_ms;");
  main.e("  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);");
  main.e("  printf(\"Mugraph runtime = %.8lfms\\n\", runtime_ms / 1024);");
  main.e("}");

  // Write profiling code for main program
  ofstream file(file_path);
  file << ct.generate_header_code("") << "\n";
  for (auto const &k : kernels) {
    file << k << "\n";
  }
  file << executor.to_string() << "\n";
  file << main.to_string();
  file.close();
}

}; // namespace kernel
}; // namespace mirage
