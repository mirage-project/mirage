#include "mirage/search/search_context.h"

namespace mirage {
namespace search {

void from_json(json const &j, SearchContext &c) {
  c.kn_graph = std::make_shared<kernel::Graph>();
  from_json(j.at("kn_graph"), *c.kn_graph);
  if (j.contains("tb_graph")) {
    c.tb_graph = std::make_shared<threadblock::Graph>();
    from_json(j.at("tb_graph"), *c.tb_graph);
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
