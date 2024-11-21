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

#include "mirage/triton_transpiler/transpile.h"
#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/forloop_accum.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace triton_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

using mirage::transpiler::CodeKeeper;
using mirage::transpiler::Combine;
using mirage::transpiler::fmt;
using mirage::transpiler::map;
using std::string;


string operator_type_to_triton(type::TBOperatorType type) {
  switch (type) {
    case type::TB_EXP_OP:
      return "tl.exp";  
    case type::TB_SILU_OP:  
      return "tl.sigmoid";
    case type::TB_SQRT_OP:
      return "tl.sqrt";
    case type::TB_DIV_OP:
      return "tl.fdiv"; //TODO: AttributeError: module 'triton.language' has no attribute 'div_rn'
    default:
      assert(false && "Unsupported operator type");
  }
}

TritonCustomOPTranspileResult
TritonTranspiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  int cur_kernel_idx = kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_kernel_idx);

  // Generate kernel function
  CodeKeeper code;
  code.e("@triton.jit");
  code.e("def $($, $):",
          func_name,
          map<kn::DTensor, string>(op->output_tensors,
             [](kn::DTensor const &dtensor) -> string {
               return fmt("dtensor$: tl.tensor", dtensor.guid);
             }),
          map<kn::DTensor, string>(op->input_tensors,
             [](kn::DTensor const &dtensor) -> string {
               return fmt("dtensor$: tl.tensor", dtensor.guid);
             }));
  code.inc_indent();

  // Initialize accumulation tensors
  for (tb::TBOperator *tb_op : g.operators) {
    if (tb_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      tb::STensor const &stensor = tb_op->output_tensors.at(0);
      // STensorMeta meta = stensor_metas.at(stensor.guid);
      
      // Create accumulator with appropriate shape
      std::string shape = "";
      for (int i = 0; i < stensor.num_dims; i++) {
        shape += fmt("$,", stensor.dim[i]);
      }
      code.e("$ = tl.zeros(($), dtype=tl.float32)",
             fmt("stensor$", stensor.guid),
             shape);
    }
  }
  // Generate forloop if needed
  if (g.forloop_range > 1) {
    code.e("for i in range($):", g.forloop_range);
    code.inc_indent();
  }

  // Generate code for operators before accum
  for (tb::TBOperator *tb_op : g.operators) {
    bool after_accum = false;
    for (tb::STensor const &input : tb_op->input_tensors) {
      if (input.after_accum) {
        after_accum = true;
      }
    }
    if (after_accum) {
      continue;
    }
    switch (tb_op->op_type) {
      case type::TB_INPUT_OP: {
        tb::TBInputOp const *input_op = 
            static_cast<tb::TBInputOp const *>(tb_op);
        kn::DTensor const &dtensor = input_op->dtensor;
        tb::STensor const &stensor = input_op->output_tensors.at(0);

        int3 imap = input_op->input_map;
        int forloop_dim = input_op->forloop_dim;
        std::string ptr_expr;

        if (stensor.num_dims == 1) {
            // 1D case
            if (forloop_dim != -1) {
                ptr_expr = fmt("i * $ + tl.arange(0, $)", 
                    stensor.dim[forloop_dim], 
                    stensor.dim[forloop_dim]);
            } else if (imap.x != -1) {
                ptr_expr = fmt("tl.program_id(0) * $ + tl.arange(0, $)",
                    stensor.dim[imap.x],
                    stensor.dim[imap.x]);
            }
        } 
        else if (stensor.num_dims == 2) {
            // 2D case - M x N matrix
            int M = stensor.dim[0];
            int N = stensor.dim[1];
            
            std::string m_expr, n_expr;
            
            // M dimension (rows)
            if (imap.x == 0) {
                m_expr = fmt("tl.program_id(0) * $ + tl.arange(0, $)", M, M);
            } else if (forloop_dim == 0) {
                m_expr = fmt("i * $ + tl.arange(0, $)", M, M);
            } else {
                m_expr = fmt("tl.arange(0, $)", M);
            }
            
            // N dimension (cols)
            if (imap.x == 1) {
                n_expr = fmt("tl.program_id(0) * $ + tl.arange(0, $)", N, N);
            } else if (forloop_dim == 1) {
                n_expr = fmt("i * $ + tl.arange(0, $)", N, N);
            } else {
                n_expr = fmt("tl.arange(0, $)", N);
            }
            
            ptr_expr = fmt("($)[:, None] * $ + ($)[None, :]", 
                m_expr, N, n_expr);
        }

        // Generate load instruction
        code.e("$ = tl.load($ + $)",
              fmt("stensor$", stensor.guid),
              fmt("dtensor$", dtensor.guid),
              ptr_expr);
        break;
      }

      case type::TB_OUTPUT_OP: {
        assert(false && "Cannot have output op before forloop_accum");
        break;
      }

      case type::TB_MATMUL_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        code.e("$ = tl.dot($, $)",
               fmt("stensor$", output.guid),
               fmt("stensor$", input0.guid),
               fmt("stensor$", input1.guid));
        break;
      }

      case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        code.e("$ += $", 
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid));
        break;
      }

      case type::TB_EXP_OP:
      case type::TB_SILU_OP:
      case type::TB_SQRT_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input.num_dims == output.num_dims);
        
        string op_str = operator_type_to_triton(tb_op->op_type);

        code.e("$ = $($)",
              fmt("stensor$", output.guid),
              op_str,
              fmt("stensor$", input.guid));
        break;
      }
      case type::TB_SQUARE_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        code.e("$ = $ * $",
              fmt("stensor$", output.guid),
              fmt("stensor$", input.guid),
              fmt("stensor$", input.guid));
        break;
      }
      case type::TB_MUL_SCALAR_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        tb::TBElementUnaryOp *unary = 
            static_cast<tb::TBElementUnaryOp *>(tb_op);
        code.e("$ = $ * $",
              fmt("stensor$", output.guid),
              fmt("stensor$", input.guid),
              unary->scalar);
        break;
      }

      case type::TB_ADD_OP:
      case type::TB_MUL_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        string op_str;
        switch (tb_op->op_type) {
          case type::TB_ADD_OP:
            op_str = "+";break;
          case type::TB_MUL_OP:
            op_str = "*";break;
        }
        code.e("$ = $ $ $",
              fmt("stensor$", output.guid),
              fmt("stensor$", input0.guid),
              op_str,
              fmt("stensor$", input1.guid));
      }

      case type::TB_DIV_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        code.e("$ = tl.fdiv($, $)",  
              fmt("stensor$", output.guid),
              fmt("stensor$", input0.guid),
              fmt("stensor$", input1.guid));
        break;
      }

      case type::TB_REDUCTION_0_OP:
      case type::TB_REDUCTION_1_OP:
      case type::TB_REDUCTION_2_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        int reduc_dim = tb_op->op_type - type::TB_REDUCTION_0_OP;
        code.e("$ = tl.sum($, axis=$)",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid),
               reduc_dim);
        // keep dims
        string keep_dims = "";
        for (int i = 0; i < output.num_dims; i++) {
          if (i == reduc_dim) {
            keep_dims += "None";
          } else {
            keep_dims += ":";
          }
          if (i < output.num_dims - 1) {
            keep_dims += ",";
          }
        }
        code.e("$ = $[$]", fmt("stensor$", output.guid), fmt("stensor$", output.guid), keep_dims);
        break;
      }

      default:
        assert(false && "Unsupported threadblock operator");
    }
  }
  if (g.forloop_range > 1) {
    code.dec_indent();
  }

  // Perform post processing after forloop accum
  for (tb::TBOperator *tb_op : g.operators) {
      bool after_accum = false;
      for (tb::STensor const &input : tb_op->input_tensors) {
          if (input.after_accum) {
              after_accum = true;
          }
      }
      if (!after_accum) {
          continue;
      }

      switch (tb_op->op_type) {
          case type::TB_INPUT_OP: {
              assert(false && "Cannot have input op after forloop_accum");
              break;
          }
          
          case type::TB_OUTPUT_OP: {
              tb::TBOutputOp const *output_op = 
                  static_cast<tb::TBOutputOp const *>(tb_op);
              kn::DTensor const &dtensor = output_op->dtensor;
              tb::STensor const &stensor = output_op->input_tensors.at(0);

              int3 omap = output_op->output_map;
              std::string ptr_expr;

              if (stensor.num_dims == 1) {
                  // 1D case
                  if (omap.x != -1) {
                      ptr_expr = fmt("tl.program_id(0) * $ + tl.arange(0, $)",
                          stensor.dim[omap.x],
                          stensor.dim[omap.x]);
                  }
              } 
              else if (stensor.num_dims == 2) {
                  // 2D case - M x N matrix
                  int M = stensor.dim[0];
                  int N = stensor.dim[1];
                  
                  std::string m_expr, n_expr;
                  
                  // M dimension (rows)
                  if (omap.x == 0) {
                      m_expr = fmt("tl.program_id(0) * $ + tl.arange(0, $)", M, M);
                  } else {
                      m_expr = fmt("tl.arange(0, $)", M);
                  }
                  
                  // N dimension (cols)
                  if (omap.x == 1) {
                      n_expr = fmt("tl.program_id(0) * $ + tl.arange(0, $)", N, N);
                  } else if (omap.y == 1) {
                      n_expr = fmt("tl.program_id(1) * $ + tl.arange(0, $)", N, N);
                  } else {
                      n_expr = fmt("tl.arange(0, $)", N);
                  }
                  
                  // Combine expressions with proper broadcasting
                  ptr_expr = fmt("($)[:, None] * $ + ($)[None, :]", 
                      m_expr, N, n_expr);
              }

              // Handle transpose if needed
              // bool need_transpose = (meta.partition_dim == stensor.num_dims - 1);
              // if (need_transpose) {
              //     code.e("$_trans = tl.trans($)",
              //           fmt("stensor$", stensor.guid),
              //           fmt("stensor$", stensor.guid));
              // }

              // Generate store instruction
              code.e("tl.store($ + $, $)",
                    fmt("dtensor$", dtensor.guid),
                    ptr_expr,
                    fmt("stensor$", stensor.guid));

              break;
          }

          case type::TB_EXP_OP:
          case type::TB_SILU_OP:
          case type::TB_SQRT_OP: {
              tb::STensor const &input = tb_op->input_tensors.at(0);
              tb::STensor const &output = tb_op->output_tensors.at(0);
              
              string op_str = operator_type_to_triton(tb_op->op_type);
              code.e("$ = $($)",
                    fmt("stensor$", output.guid),
                    op_str,
                    fmt("stensor$", input.guid));
              break;
          }
          case type::TB_SQUARE_OP: {
              tb::STensor const &input = tb_op->input_tensors.at(0);
              tb::STensor const &output = tb_op->output_tensors.at(0);
              code.e("$ = $ * $",
                    fmt("stensor$", output.guid),
                    fmt("stensor$", input.guid),
                    fmt("stensor$", input.guid));
              break;
          }
          case type::TB_ADD_OP:
          case type::TB_MUL_OP: {
              tb::STensor const &input0 = tb_op->input_tensors.at(0);
              tb::STensor const &input1 = tb_op->input_tensors.at(1);
              tb::STensor const &output = tb_op->output_tensors.at(0);
              
              string op_symbol = "";
              switch(tb_op->op_type) {
                  case type::TB_ADD_OP: op_symbol = "+"; break;
                  case type::TB_MUL_OP: op_symbol = "*"; break;
                  default: assert(false);
              }
              
              code.e("$ = $ $ $",
                    fmt("stensor$", output.guid),
                    fmt("stensor$", input0.guid),
                    op_symbol,
                    fmt("stensor$", input1.guid));
              break;
          }
          case type::TB_DIV_OP: {
              tb::STensor const &input0 = tb_op->input_tensors.at(0);
              tb::STensor const &input1 = tb_op->input_tensors.at(1);
              tb::STensor const &output = tb_op->output_tensors.at(0);
              code.e("$ = tl.fdiv($, $)",  //TODO: AttributeError: module 'triton.language' has no attribute 'div_rn'
                    fmt("stensor$", output.guid),
                    fmt("stensor$", input0.guid),
                    fmt("stensor$", input1.guid));
              break;
          }

          case type::TB_REDUCTION_0_OP:
          case type::TB_REDUCTION_1_OP:
          case type::TB_REDUCTION_2_OP: {
              tb::STensor const &input = tb_op->input_tensors.at(0);
              tb::STensor const &output = tb_op->output_tensors.at(0);
              int reduc_dim = tb_op->op_type - type::TB_REDUCTION_0_OP;
              
              code.e("$ = tl.sum($, axis=$)",
                    fmt("stensor$", output.guid),
                    fmt("stensor$", input.guid),
                    reduc_dim);
              // keep dims
              string keep_dims = "";
              for (int i = 0; i < output.num_dims; i++) {
                if (i == reduc_dim) {
                  keep_dims += "None";
                } else {
                  keep_dims += ":";
                }
                if (i < output.num_dims - 1) {
                  keep_dims += ",";
                }
              }
              code.e("$ = $[$]", fmt("stensor$", output.guid), fmt("stensor$", output.guid), keep_dims);
              break;
          }
          default: {
              assert(false && fmt("Unsupported op_type:$", tb_op->op_type).c_str());
          }
      }
  }

  return TritonCustomOPTranspileResult{func_name, code.to_string()};
}

} // namespace triton_transpiler
} // namespace mirage