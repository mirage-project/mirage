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
#include "mirage/threadblock/serializer/concat_serializer.h"
#include "mirage/threadblock/serializer/element_binary_serializer.h"
#include "mirage/threadblock/serializer/element_unary_serializer.h"
#include "mirage/threadblock/serializer/input_loader_serializer.h"
#include "mirage/threadblock/serializer/matmul_serializer.h"
#include "mirage/threadblock/serializer/output_saver_serializer.h"
#include "mirage/threadblock/serializer/reduction_serializer.h"

#include <fstream>
#include <iostream>
#include <map>

namespace mirage {
namespace transpiler {

std::string gen_header_code(std::string indent) {
  std::stringstream ss;
  ss << indent << "#include<cuda.h>\n";
  return ss.str();
}

void gen_cuda_code_input_loader(mirage::threadblock::NewKernelParams params,
                                int &param_idx,
                                std::stringstream &main,
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
  main << ind << "int tb_offset_row = "
       << "blockIdx.x * " << input_matrix_row_offset_block_stride.x
       << "+ blockIdx.y * " << input_matrix_row_offset_block_stride.y
       << "+ blockIdx.z * " << input_matrix_row_offset_block_stride.z
       << "+ i * " << input_matrix_row_offset_forloop_stride << "\n";
  main << ind << "int tb_offset_column = "
       << "blockIdx.x * " << input_matrix_column_offset_block_stride.x
       << "+ blockIdx.y * " << input_matrix_column_offset_block_stride.y
       << "+ blockIdx.z * " << input_matrix_column_offset_block_stride.z
       << "+ i * " << input_matrix_column_offset_forloop_stride << "\n";
  main << ind << "int global_offset = "
       << "blockIdx.x * " << global_offset_block_stride.x << "+ blockIdx.y * "
       << global_offset_block_stride.y << "+ blockIdx.z * "
       << global_offset_block_stride.z << "+ i * "
       << global_offset_forloop_stride << "\n";
  main << ind << "cutlass::MatrixCoord matrix_offset"
       << " = {tb_offset_row, tb_offset_column};\n";
  main << ind << "cutlass::half_t *stensor_ptr =\n"
       << ind << "    (cutlass::half_t*)(smem_buffer[(i+1)%2] + "
       << input_smem_offset << ")\n";
  main << ind << "mirage::threadblock::GenericInputLoader loader(\n"
       << ind << "    dtensor_ptr,\n"
       << ind << "    stensor_ptr,\n"
       << ind << "    dtensor_matrix_shape,\n"
       << ind << "    stensor_matrix_shape,\n"
       << ind << "    dtensor_layout,\n"
       << ind << "    stensor_layout,\n"
       << ind << "    threadIdx.x,\n"
       << ind << "    blockDim.x,\n"
       << ind << "    matrix_offset,\n"
       << ind << "    global_offset);\n";
}

void gen_cuda_code_matmul_op(mirage::threadblock::NewKernelParams params,
                             int &param_idx,
                             int &op,
                             std::stringstream &main,
                             std::string ind) {
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
  main << ind << "cutlass::half_t *_A_ptr =\n"
       << ind << "    (cutlass::half_t*)(smem_buffer[i % 2] + " << A_smem_offset
       << ")\n";
  main << ind << "cutlass::half_t *_B_ptr =\n"
       << ind << "    (cutlass::half_t*)(smem_buffer[i % 2] + " << B_smem_offset
       << ")\n";
  main << ind << "cutlass::half_t *_C_ptr =\n"
       << ind << "    (cutlass::half_t*)(smem_buffer[i % 2] + " << C_smem_offset
       << ")\n";
  if (op < params.num_operators &&
      params.operator_types[op + 1] == mirage::type::TB_EXP_OP) {
    // fuse this matmul with the next exp
    int E_smem_offset, num_elements;
    mirage::threadblock::deserialize_elementunary_op_parameters(
        params.parameters, param_idx, E_smem_offset, num_elements);
    // assert inline exp
    assert(C_smem_offset == E_smem_offset);
    main << ind
         << "mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_EXP> "
            "executor(\n"
         << ind << "_A_ptr, _B_ptr, _C_ptr, " << m << ", " << n << ", " << k
         << ", threadIdx.x);\n";

    op += 1;
  } else {
    main << ind
         << "mirage::threadblock::GenericMatmulExecutor<mirage::type::ACT_NONE>"
            " executor(\n"
         << ind << "_A_ptr, _B_ptr, _C_ptr, " << m << ", " << n << ", " << k
         << ", threadIdx.x);\n";
  }
}

void gen_cuda_code_exp_op(mirage::threadblock::NewKernelParams params,
                          int &param_idx,
                          std::stringstream &main,
                          std::string ind) {
  int smem_offset, num_elements;
  mirage::threadblock::deserialize_elementunary_op_parameters(
      params.parameters, param_idx, smem_offset, num_elements);
  main << ind << "cutlass::half_t *_ptr = "
       << "(cutlass::half_t *)(smem_buffer[i % 2] + " << smem_offset << ")\n";
}

void gen_cuda_code_div_op(mirage::threadblock::NewKernelParams params,
                          int &param_idx,
                          std::stringstream &main,
                          std::string ind) {
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

  main << ind << "cutlass::half_t *_in1_ptr = "
       << "(cutlass::half_t *)(smem_buffer[i % 2] + " << input1_smem_offset
       << ")\n";
  main << ind << "cutlass::half_t *_in2_ptr = "
       << "(cutlass::half_t *)(smem_buffer[i % 2] + " << input2_smem_offset
       << ")\n";
  main << ind << "cutlass::half_t *_out_ptr = "
       << "(cutlass::half_t *)(smem_buffer[i % 2] + " << output_smem_offset
       << ")\n";
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
       << "}\n";
}

std::string
    CudaTranspiler::gen_kernel_code(mirage::threadblock::NewKernelParams params,
                                    int forloop_range,
                                    int reduction_dimx,
                                    std::string func_name,
                                    std::vector<std::string> input_names,
                                    std::vector<std::string> output_names,
                                    std::string ind) {
  using namespace mirage::threadblock;
  using namespace std;
  stringstream header;
  stringstream main;
  stringstream ending;
  {
    header << "__global__ void " << func_name << "(";
    for (size_t i = 0; i < input_names.size(); i++) {
      if (i > 0) {
        header << ", ";
      }
      header << "void * " << input_names[i];
    }
    for (size_t i = 0; i < output_names.size(); i++) {
      header << ", ";
      header << output_names[i];
    }
    header << ") {\n";
  }
  header << ind << "extern __shared__ char smem_buffer[];\n";
  main << ind << "for (int i = 0; i < " << forloop_range << "; i++) {\n";
  int param_idx = 0;
  int output_idx = 0;
  for (int op = 0; op < params.num_operators; op++) {
    mirage::type::TBOperatorType op_type = params.operator_types[op];
    switch (op_type) {
      case mirage::type::TB_INPUT_OP: {
        main << ind << "{\n";
        gen_cuda_code_input_loader(params, param_idx, main, ind + "  ");
        main << ind << "}\n";
        break;
      }
      case mirage::type::TB_OUTPUT_OP: {
        break;
      }
      case mirage::type::TB_MATMUL_OP: {
        main << ind << "{\n";
        gen_cuda_code_matmul_op(params, param_idx, op, main, ind + "  ");
        main << ind << "}\n";
        break;
      }
      case mirage::type::TB_EXP_OP: {
        main << ind << "{\n";
        gen_cuda_code_exp_op(params, param_idx, main, ind + "  ");
        main << ind << "}\n";
        break;
      }
      case mirage::type::TB_DIV_OP: {
        main << ind << "{\n";
        gen_cuda_code_div_op(params, param_idx, main, ind + "  ");
        main << ind << "}\n";
        break;
      }
      default: {
        assert(false && "Unsupported op_type");
      }
    }
  }
  assert(params.num_parameters == param_idx);
  return header.str() + main.str() + ending.str();
}

}; // namespace transpiler
}; // namespace mirage
