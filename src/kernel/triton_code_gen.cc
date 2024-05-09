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

#include "mirage/kernel/graph.h"
#include "mirage/kernel/customized.h"
#include "mirage/utils/hash_utils.h"
#include "mirage/threadblock/serializer/concat_serializer.h"
#include "mirage/threadblock/serializer/element_binary_serializer.h"
#include "mirage/threadblock/serializer/element_unary_serializer.h"
#include "mirage/threadblock/serializer/input_loader_serializer.h"
#include "mirage/threadblock/serializer/matmul_serializer.h"
#include "mirage/threadblock/serializer/output_saver_serializer.h"
#include "mirage/threadblock/serializer/reduction_serializer.h"

#include <iostream>
#include <fstream>
#include <map>

namespace mirage {
namespace kernel {

std::string dtensor_name(int i) {
  return "dtensor"+std::to_string(i);
}

std::string stensor_name(int i) {
  return "stensor"+std::to_string(i);
}

std::string stensor_ptr_name(int i) {
  return "stensor"+std::to_string(i)+"_ptr";
}

std::string tensor_dims(DTensor const &tensor) {
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < tensor.num_dims; i++) {
    ss << tensor.dim[i];
    if (i != tensor.num_dims-1) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

std::string block_offset_calculation(int off_x, int off_y, int off_z) {
  if (off_x == 0 && off_y == 0 && off_z == 0) {
    return "0";
  }
  std::stringstream ret;
  if (off_x > 0) {
    ret << "bidx * " << off_x;
  }
  if (off_y > 0) {
    if (off_x > 0) {
      ret << " + ";
    }
    ret << "bidy * " << off_y;
  }
  if (off_z > 0) {
    if (off_x > 0 || off_y > 0) {
      ret << " + ";
    }
    ret << "bidz * " << off_z;
  }
  return ret.str();
}

std::string generate_kernel_code(mirage::threadblock::NewKernelParams params,
                                 int forloop_range,
                                 int reduction_dimx,
                                 std::string func_name,
                                 std::vector<std::string> input_names,
                                 std::vector<std::string> output_names,
                                 std::map<int, mirage::threadblock::STensor> offset_to_stensor) {
  using namespace mirage::threadblock;
  using namespace std;
  stringstream header;
  stringstream main;
  stringstream ending;
  {
    header << "@triton.jit\n";
    header << "def " << func_name << "(";
    for (size_t i = 0; i < input_names.size(); i++) {
      if (i > 0) {
        header << ", ";
      }
      header << input_names[i];
    }
    for (size_t i = 0; i < output_names.size(); i++) {
      header << ", ";
      header << output_names[i];
    }
    header << "):\n";
  }
  header << "\tbidx = tl.program_id(0)\n";
  header << "\tbidy = tl.program_id(1)\n";
  header << "\tbidz = tl.program_id(2)\n";
  main << "\tfor i in range(" << forloop_range << "):\n";
  map<int, string> stensor_guid_to_name;
  int param_idx = 0;
  int output_idx = 0;
  for (int op = 0; op < params.num_operators; op++) {
    mirage::type::TBOperatorType op_type = params.operator_types[op];
    if (op_type == mirage::type::TB_INPUT_OP) {
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
      header << "\t" << stensor_ptr_name(input_smem_offset)
           << " = tl.make_block_ptr(\n"
           << "\t\tbase = " << input_names[op] << " + "
           << block_offset_calculation(global_offset_block_stride.x, global_offset_block_stride.y, global_offset_block_stride.z)
           << ",\n";
      header << "\t\tshape = ("
           << dtensor_matrix_shape.x << ", "
           << dtensor_matrix_shape.y << "),\n";
      header << "\t\tblock_shape = ("
           << stensor_matrix_shape.x << ", "
           << stensor_matrix_shape.y << "),\n";
      string tb_offset_row = block_offset_calculation(input_matrix_row_offset_block_stride.x,
                                                      input_matrix_row_offset_block_stride.y,
                                                      input_matrix_row_offset_block_stride.z);
      string tb_offset_col = block_offset_calculation(input_matrix_column_offset_block_stride.x,
                                                      input_matrix_column_offset_block_stride.y,
                                                      input_matrix_column_offset_block_stride.z);
      header << "\t\toffsets = ("
           << tb_offset_row << ", "
           << tb_offset_col << "),\n";
      // Assume row major layout for now
      header << "\t\tstrides = ("
           << stensor_matrix_shape.y << ", 1),\n";
      header << "\t\torder = (1, 0))\n";
      if ((input_matrix_row_offset_forloop_stride != 0) || (input_matrix_column_offset_forloop_stride != 0)) {
        main << "\t\t" << stensor_name(input_smem_offset)
             << " = tl.load(" << stensor_ptr_name(input_smem_offset) << ")\n";
        main << "\t\t" << stensor_name(input_smem_offset) << "_ptr"
             << " = tl.advance(" << stensor_ptr_name(input_smem_offset) << ", "
             << "(" << input_matrix_row_offset_forloop_stride << ", "
             << input_matrix_column_offset_forloop_stride << "))\n";
      } else {
        // No need to load it in the forloop
        header << "\t" << stensor_name(input_smem_offset)
               << " = tl.load(" << stensor_ptr_name(input_smem_offset) << ")\n";
      }
    } else if (op_type == mirage::type::TB_OUTPUT_OP) {
      int3 output_matrix_row_offset_block_stride;
      int3 output_matrix_column_offset_block_stride;
      int3 global_offset_block_stride;
      int2 dtensor_matrix_shape, stensor_matrix_shape;
      int input_smem_offset, accum_smem_offset;
      mirage::layout::DmemLayout dtensor_layout;
      mirage::layout::SmemLayout stensor_layout;
      mirage::threadblock::deserialize_output_saver_parameters(
          params.parameters,
          param_idx,
          output_matrix_row_offset_block_stride,
          output_matrix_column_offset_block_stride,
          global_offset_block_stride,
          dtensor_matrix_shape,
          stensor_matrix_shape,
          dtensor_layout,
          stensor_layout,
          input_smem_offset,
          accum_smem_offset);
      header << "\t" << stensor_ptr_name(accum_smem_offset)
           << " = tl.make_block_ptr(\n"
           << "\t\tbase = " << output_names[output_idx] << " + "
           << block_offset_calculation(global_offset_block_stride.x, global_offset_block_stride.y, global_offset_block_stride.z)
           << ",\n";
      header << "\t\tshape = ("
           << dtensor_matrix_shape.x << ", "
           << dtensor_matrix_shape.y << "),\n";
      header << "\t\tblock_shape = ("
           << stensor_matrix_shape.x << ", "
           << stensor_matrix_shape.y << "),\n";
      string tb_offset_row = block_offset_calculation(output_matrix_row_offset_block_stride.x,
                                                      output_matrix_row_offset_block_stride.y,
                                                      output_matrix_row_offset_block_stride.z);
      string tb_offset_col = block_offset_calculation(output_matrix_column_offset_block_stride.x,
                                                      output_matrix_column_offset_block_stride.y,
                                                      output_matrix_column_offset_block_stride.z);
      header << "\t\toffsets = ("
           << tb_offset_row << ", "
           << tb_offset_col << "),\n";
      // Assume row major layout for now
      header << "\t\tstrides = ("
           << stensor_matrix_shape.y << ", 1),\n";
      header << "\t\torder = (1, 0))\n";
      header << "\t" << stensor_name(accum_smem_offset)
             << " = tl.zeros(["
             << stensor_matrix_shape.x << ", "
             << stensor_matrix_shape.y << "]"
             << ", dtype = tl.float16)\n";
      main << "\t\t" << stensor_name(accum_smem_offset)
           << " = " << stensor_name(accum_smem_offset) << " + "
           << stensor_name(input_smem_offset) << "\n";
      ending << "\ttl.store(" << stensor_ptr_name(accum_smem_offset) << ", "
             << stensor_name(accum_smem_offset) <<")\n" ;
      output_idx ++;
    } else if (op_type == mirage::type::TB_MATMUL_OP) {
      int m, n, k;
      int A_smem_offset, B_smem_offset, C_smem_offset;
      mirage::threadblock::deserialize_matmul_op_parameters(
          params.parameters,
          param_idx,
          m,
          n,
          k,
          A_smem_offset,
          B_smem_offset,
          C_smem_offset);
      main << "\t\t" << stensor_name(C_smem_offset) << " = tl.dot("
           << stensor_name(A_smem_offset) << ", "
           << stensor_name(B_smem_offset) << ", out_dtype=tl.float16)\n";
    } else if (op_type == mirage::type::TB_EXP_OP) {
      int smem_offset, num_elements;
      mirage::threadblock::deserialize_elementunary_op_parameters(
          params.parameters, param_idx, smem_offset, num_elements);
      main << "\t\t" << stensor_name(smem_offset) << " = tl.math.exp("
           << stensor_name(smem_offset) << ".to(tl.float32)).to(tl.float16)\n";
    } else if (op_type == mirage::type::TB_DIV_OP) {
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
      main << "\t\t" << stensor_name(output_smem_offset) << " = tl.fdiv("
           << stensor_name(input1_smem_offset) << ".to(tl.float32)" << ", "
           << stensor_name(input2_smem_offset) << ".to(tl.float32)).to(tl.float16)\n";
    } else if ((op_type >= mirage::type::TB_REDUCTION_FIRST_OP_ID) &&
               (op_type <= mirage::type::TB_REDUCTION_LAST_OP_ID)) {
      int output_num_elements, reduction_degree, inner_range;
      int input_smem_offset, output_smem_offset;
      mirage::threadblock::deserialize_reduction_op_parameters(
          params.parameters,
          param_idx,
          output_num_elements,
          reduction_degree,
          inner_range,
          input_smem_offset,
          output_smem_offset);
      int reduction_dim = -1;
      if (op_type >= mirage::type::TB_REDUCTION_0_TO_DIMX_OP &&
          op_type <= mirage::type::TB_REDUCTION_2_TO_DIMX_OP) {
        reduction_dim = op_type - mirage::type::TB_REDUCTION_0_TO_DIMX_OP;
        assert(offset_to_stensor.find(input_smem_offset) != offset_to_stensor.end());
        STensor stensor = offset_to_stensor[input_smem_offset];
        // Assert that only the last two dimensions can be non-zero
        for (int i = 0; i < stensor.num_dims - 2; i++) {
          assert(stensor.dim[i] == 1);
        }
        if (reduction_dim == stensor.num_dims - 1) {
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.reshape("
               << stensor_name(input_smem_offset) << ", ("
               << stensor.dim[stensor.num_dims-2] << ", "
               << stensor.dim[stensor.num_dims-1] / reduction_dimx << ", "
               << reduction_dimx << "))\n";
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.sum("
               << stensor_name(output_smem_offset) << ", axis=1)\n";
        } else if (reduction_dim == stensor.num_dims - 2) {
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.reshape("
               << stensor_name(input_smem_offset) << ", ("
               << stensor.dim[stensor.num_dims-2] / reduction_dimx << ", "
               << reduction_dimx << ", "
               << stensor.dim[stensor.num_dims-1] << "))\n";
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.sum("
               << stensor_name(output_smem_offset) << ", axis=0)\n";
        } else {
          assert(false && "Unsupported reduction dim");
        }
      } else if (op_type >= mirage::type::TB_REDUCTION_0_OP &&
                 op_type <= mirage::type::TB_REDUCTION_2_OP) {
        reduction_dim = op_type - mirage::type::TB_REDUCTION_0_OP;
        assert(offset_to_stensor.find(input_smem_offset) != offset_to_stensor.end());
        STensor stensor = offset_to_stensor[input_smem_offset];
        // Assert that only the last two dimensions can be non-zero
        for (int i = 0; i < stensor.num_dims - 2; i++) {
          assert(stensor.dim[i] == 1);
        }
        if (reduction_dim == stensor.num_dims - 1) {
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.reshape("
               << stensor_name(input_smem_offset) << ", ("
               << stensor.dim[stensor.num_dims-2] << ", "
               << stensor.dim[stensor.num_dims-1] << ", "
               << 1 << "))\n";
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.sum("
               << stensor_name(output_smem_offset) << ", axis=1)\n";
        } else if (reduction_dim == stensor.num_dims - 2) {
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.reshape("
               << stensor_name(input_smem_offset) << ", ("
               << stensor.dim[stensor.num_dims-2] << ", "
               << 1 << ", "
               << stensor.dim[stensor.num_dims-1] << "))\n";
          main << "\t\t" << stensor_name(output_smem_offset) << " = tl.sum("
               << stensor_name(output_smem_offset) << ", axis=0)\n";
        } else {
          assert(false && "Unsupported reduction dim");
        }
#ifdef DEADCODE
        // The tensor in triton is 2D, so we need to adjust the reduction dimension
        main << "\t\t" << stensor_name(output_smem_offset) << " = tl.sum("
             << stensor_name(input_smem_offset) << ", axis="
             << reduction_dim - stensor.num_dims + 2 << ", keep_dims=True)\n";
#endif
      } else {
        assert(false);
      }
    }
  }
  assert(params.num_parameters == param_idx);
  assert(output_names.size() == (size_t)output_idx);
  return header.str() + main.str() + ending.str();
}

void Graph::generate_triton_program(char const *file_path) {
  using namespace std;
  stringstream header;
  vector<std::string> kernels;
  stringstream launcher;
  stringstream main_program;
  main_program << "def main():\n";
  header << "import triton\nimport torch\nimport triton.language as tl\n";
  launcher << "def kernel_launcher(";
  for (KNOperator *const op : this->operators) {
    for (const auto & output : op->output_tensors) {
      launcher << dtensor_name(output.guid) << ", ";
    }
  }
  launcher << "):\n";
  for (KNOperator *const op : this->operators) {
    for (const auto & output : op->output_tensors) {
        main_program << "\t"
            << dtensor_name(output.guid)
            << " = torch.randn("
            << tensor_dims(output)
            << ", dtype=torch.float16, device=\"cuda\", requires_grad=False)\n";
    }
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        assert(op->output_tensors.size() == 1);
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        const KNCustomizedOp *customized = static_cast<const KNCustomizedOp*>(op);
        vector<string> input_names;
        vector<string> output_names;
        for (const auto& t : op->input_tensors) {
          input_names.push_back(dtensor_name(t.guid));
        }
        for (const auto& t : op->output_tensors) {
          output_names.push_back(dtensor_name(t.guid));
        }
        std::map<int, mirage::threadblock::STensor> offset_to_stensor;
        for (const auto& tbo : customized->bgraph.operators) {
          for (const auto & t : tbo->output_tensors) {
            offset_to_stensor[t.smem_offset] = t;
          }
        }
        mirage::threadblock::NewKernelParams params = customized->bgraph.get_new_kernel_params(false/*fingerprint*/);
        string kernel_code = generate_kernel_code(
            params,
            customized->bgraph.forloop_range,
            customized->bgraph.reduction_dimx,
            "graphdef_kernel_" + to_string(kernels.size()),
            input_names,
            output_names,
            offset_to_stensor);
        launcher << "\tgrid = (" << customized->bgraph.grid_dim.x << ", "
                 << customized->bgraph.grid_dim.y << ", "
                 << customized->bgraph.grid_dim.z << ")\n";
        launcher << "\tgraphdef_kernel_" << kernels.size()
                 << "[grid](\n\t\t";
        for (size_t i = 0; i < input_names.size(); i++) {
          if (i > 0)
            launcher << ", \n\t\t";
          launcher << input_names[i];
        }
        for (size_t i = 0; i < output_names.size(); i++) {
          launcher << ", \n\t\t";
          launcher << output_names[i];
        }
        launcher << ")\n";
        kernels.push_back(kernel_code);
        break;
      }
      default: {
        //assert(false && "Cannot tritonize this operator");
      }
    }
  }
  // Write profiling code for main_program
  main_program << "\tfn = lambda: kernel_launcher(";
  for (KNOperator *const op : this->operators) {
    for (const auto & output : op->output_tensors) {
      main_program << dtensor_name(output.guid) << ", ";
    }
  }
  main_program << ")\n";
  main_program << "\tquantiles = [0.5, 0.1, 0.9]\n";
  main_program << "\tms, mmin, mmax = triton.testing.do_bench(fn, warmup=1000, rep=1000, quantiles=quantiles)\n";
  main_program << "\tprint(ms, mmin, mmax)\n";

  std::ofstream file(file_path);
  file << header.str() << "\n";
  for (const auto & k : kernels) {
    file << k << "\n";
  }
  file << launcher.str() << "\n" << main_program.str() << "\n";
  file << "if __name__ == \"__main__\":\n\tmain()\n";
  file.close();
}

} // namespace kernel
} // namespace mirage
