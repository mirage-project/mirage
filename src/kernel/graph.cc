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
#include "mirage/utils/hash_utils.h"

#include <iostream>

namespace mirage {
namespace kernel {

Graph::Graph() {}

size_t Graph::pair_hash::operator()(std::pair<int, int> const &p) const {
  size_t h1 = std::hash<int>{}(p.first);
  size_t h2 = std::hash<int>{}(p.second);
  hash_combine(h1, h2);
  return h1;
}

void to_json(json &j, Graph const &g) {
  for (KNOperator *const op : g.operators) {
    j.push_back(json(*op));
  }
}

void from_json(json const &j, Graph &g) {
  std::unordered_map<size_t, size_t> guid_mapping; // from deseralized guid to json guid

  auto get_tensor_from_guid = [&](size_t guid) {
    for (auto const &op : g.operators) {
      for (DTensor const &dtensor : op->output_tensors) {
        if (guid_mapping.at(dtensor.guid) == guid) {
          return dtensor;
        }
      }
    }
    assert(false);
  };

  for (json const &jop : j) {
    type::KNOperatorType op_type;
    jop.at("op_type").get_to(op_type);
    switch (op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        int num_dim, dim[MAX_TENSOR_DIMS];
        type::DataType data_type;
        layout::DmemLayout layout;
        size_t guidO;
        jop.at("output_tensors")[0].at("num_dims").get_to(num_dim);
        jop.at("output_tensors")[0].at("dim").get_to(dim);
        jop.at("output_tensors")[0].at("data_type").get_to(data_type);
        jop.at("output_tensors")[0].at("layout").get_to(layout);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        std::vector<int> dims = to_vector(num_dim, dim);
        DTensor const &output = g.new_input(dims, data_type, layout);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_MATMUL_OP: {
        size_t guidA, guidB, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guidA);
        jop.at("input_tensors")[1].at("guid").get_to(guidB);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.matmul(get_tensor_from_guid(guidA), get_tensor_from_guid(guidB));
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_EXP_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output = g.exp(get_tensor_from_guid(guid));
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_DIV_OP: {
        size_t guidA, guidB, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guidA);
        jop.at("input_tensors")[1].at("guid").get_to(guidB);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.div(get_tensor_from_guid(guidA), get_tensor_from_guid(guidB));
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_ADD_OP: {
        size_t guidA, guidB, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guidA);
        jop.at("input_tensors")[1].at("guid").get_to(guidB);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output =
            g.add(get_tensor_from_guid(guidA), get_tensor_from_guid(guidB));
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_REDUCTION_0_OP:
      case type::KNOperatorType::KN_REDUCTION_1_OP:
      case type::KNOperatorType::KN_REDUCTION_2_OP: {
        size_t guid, guidO;
        jop.at("input_tensors")[0].at("guid").get_to(guid);
        jop.at("output_tensors")[0].at("guid").get_to(guidO);
        DTensor const &output = g.reduction(
            get_tensor_from_guid(guid),
            op_type - type::KNOperatorType::KN_REDUCTION_0_OP);
        guid_mapping[output.guid] = guidO;
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        std::vector<DTensor> inputs;
        for (auto const &jinput : jop.at("input_tensors")) {
          size_t guid;
          jinput.at("guid").get_to(guid);
          inputs.push_back(get_tensor_from_guid(guid));
        }
        threadblock::ExecutionPlan plan;
        jop.at("plan").get_to(plan);
        std::vector<DTensor> outputs = g.customized(inputs, plan);
        for (size_t i = 0; i < outputs.size(); ++i) {
          size_t guidO;
          jop.at("output_tensors")[i].at("guid").get_to(guidO);
          guid_mapping[outputs[i].guid] = guidO;
        }

        // Synchronize layouts with bgraph
        KNCustomizedOp *op = dynamic_cast<KNCustomizedOp *>(g.operators.back());
        assert(op->bgraph.operators.size() == jop.at("bgraph").at("operators").size());
        for (size_t i = 0; i < op->bgraph.operators.size(); ++i) {
          threadblock::TBOperator *bop = op->bgraph.operators[i];
          json jbop = jop.at("bgraph").at("operators")[i];
          assert(bop->input_tensors.size() == jbop.at("input_tensors").size());
          assert(bop->output_tensors.size() == jbop.at("output_tensors").size());
          for (size_t j = 0; j < bop->input_tensors.size(); ++j) {
            jbop.at("input_tensors")[j].at("layout").get_to(bop->input_tensors[j].layout);
          }
          for (size_t j = 0; j < bop->output_tensors.size(); ++j) {
            jbop.at("output_tensors")[j].at("layout").get_to(bop->output_tensors[j].layout);
          }
        }
        break;
      }
      default:
        assert(false && "Cannot deserialize this operator");
    }
  }
}

} // namespace kernel
} // namespace mirage
