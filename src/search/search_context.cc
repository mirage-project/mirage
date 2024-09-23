#include "mirage/search/search_context.h"

namespace mirage {
namespace search {

void from_json(json const &j, SearchContext &c) {
  c.kn_graph = std::make_shared<kernel::Graph>();
  from_json(j.at("kn_graph"), *c.kn_graph);
  if (j.contains("tb_graph")) {
    auto get_index = [&](int guid) {
      for (size_t i = 0; i < j.at("kn_graph").size(); ++i) {
        auto op = j.at("kn_graph")[i];
        for (size_t k = 0; k < op.at("output_tensors").size(); ++k) {
          if (op.at("output_tensors")[k].at("guid") == guid) {
            return std::make_pair(i, k);
          }
        }
      }
    };

    c.tb_graph = std::make_shared<threadblock::Graph>();
    from_json(j.at("tb_graph"), *c.tb_graph);

    for (size_t i = 0; i < c.tb_graph->operators.size(); ++i) {
      auto op = c.tb_graph->operators[i];
      if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
        auto index = get_index(
            j.at("tb_graph").at("operators")[i].at("dtensor").at("guid"));
        static_cast<threadblock::TBInputOp *>(op)->dtensor =
            c.kn_graph->operators[index.first]->output_tensors[index.second];
      }
    }
  }
  from_json(j.at("level"), c.level);
}

void to_json(json &j, SearchContext const &c) {
  j["kn_graph"] = *c.kn_graph;
  if (c.tb_graph) {
    j["tb_graph"] = *c.tb_graph;
  }
  j["level"] = c.level;
}

SerializedSearchContext::SerializedSearchContext(SearchContext const &c) {
  to_json(data, c);
}

SearchContext SerializedSearchContext::deserialize() const {
  SearchContext c;
  from_json(data, c);
  return c;
}

} // namespace search
} // namespace mirage
