#include "mirage/search/search_context.h"

namespace mirage {
namespace search {

SearchContext::SearchContext()
    : kn_graph(nullptr), tb_graph(nullptr), level(SearchLevel::LV_KERNEL) {}

SearchContext::~SearchContext() {}

SearchContext SearchContext::copy() const {
  SearchContext c;
  from_json(json(*this), c);
  return c;
}

void to_json(json &j, SearchContext const &c) {
  j["kn_graph"] = json(*c.kn_graph);
  std::vector<std::pair<size_t, size_t>> inputs;
  if (c.tb_graph) {
    threadblock::ExecutionPlan plan = c.tb_graph->get_plan();
    j["tb_plan"] = json(plan);
    for (auto const &op : c.tb_graph->operators) {
      if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
        for (size_t i = 0; i < c.kn_graph->operators.size(); ++i) {
          for (size_t j = 0;
               j < c.kn_graph->operators[i]->output_tensors.size();
               ++j) {
            if (c.kn_graph->operators[i]->output_tensors[j].guid ==
                static_cast<threadblock::TBInputOp *>(op)->dtensor.guid) {
              inputs.push_back({i, j});
              break;
            }
          }
        }
      }
    }
    assert(plan.input_map.size() == inputs.size());
  }
  j["inputs"] = inputs;
  j["level"] = c.level;
}

void from_json(json const &j, SearchContext &c) {
  c.kn_graph = std::make_shared<kernel::Graph>();
  from_json(j["kn_graph"], *c.kn_graph);
  std::vector<std::pair<size_t, size_t>> inputs;
  from_json(j["inputs"], inputs);
  if (inputs.size()) {
    std::vector<kernel::DTensor> input_tensors;
    threadblock::ExecutionPlan plan;
    from_json(j["tb_plan"], plan);
    for (auto const &id : inputs) {
      input_tensors.push_back(
          c.kn_graph->operators[id.first]->output_tensors[id.second]);
    }
    c.tb_graph = std::make_shared<threadblock::Graph>(input_tensors, plan);
  }
  c.level = j["level"];
}

}
}