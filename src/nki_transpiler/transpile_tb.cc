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

#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/forloop_accum.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/threadblock/graph.h"
#include "mirage/nki_transpiler/transpile.h"
#include "mirage/transpiler/utils.h"

#include <algorithm>

namespace mirage {
namespace nki_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

using std::string;
using mirage::transpiler::fmt;
using mirage::transpiler::CodeKeeper;
using mirage::transpiler::map;

NKICustomOPTranspileResult
    NKITranspiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  static int nki_custom_kernel_idx_counter = 0;
  int cur_custom_kernel_idx = nki_custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  // Generate code prologue
  CodeKeeper code;
  code.e("@nki_jit");
  code.e("def $($, $):",
      func_name,
      map<kn::DTensor, string>(op->output_tensors,
                               [](kn::DTensor const &dtensor) -> string {
                                 return fmt("dtensor$", dtensor.guid);
                               }),
      map<kn::DTensor, string>(op->input_tensors,
                               [](kn::DTensor const &dtensor) -> string {
                                 return fmt("dtensor$", dtensor.guid);
                               }));
  code.inc_indent();
  if (g.forloop_range > 1) {
    code.e("for l in range($):", g.forloop_range);
    code.inc_indent();
  }
  // Generate code for operators before accum
  for (tb::TBOperator * tb_op : g.operators) {
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
        tb::TBInputOp const *input_op = static_cast<tb::TBInputOp const *>(tb_op);
        kn::DTensor const &dtensor = input_op->dtensor;
        tb::STensor const &stensor = input_op->output_tensors.at(0);
        int3 imap = input_op->input_map;
        std::string range;
        for (int i = 0; i < stensor.num_dims; i++) {
          for (int dim = 0; dim < 3; dim ++) {
            int div_dim = dim == 0 ? imap.x : dim == 1 ? imap.y : imap.z;
            if (div_dim >= 0) {

            }
          }
        }
        // Generate code for TB Input
        code.e("$ = nl.load($)",
            fmt("stensor$", stensor.guid));
        break;
      }
      default: {
        assert(false);
      }
    }
  }
  if (g.forloop_range > 1) {
    code.dec_indent();
  }
}

} // namespace nki_transpiler
} // namespace mirage
