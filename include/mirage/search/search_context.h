#pragma once

#include <memory>

#include <atomic>

#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace search {

enum class SearchLevel {
  LV_KERNEL,
  LV_THREADBLOCK,
};

struct SearchContext {
  std::shared_ptr<kernel::Graph> kn_graph;
  std::shared_ptr<threadblock::Graph> tb_graph;
  SearchLevel level;

  SearchContext copy() const;

  SearchContext();
  ~SearchContext();
};

void to_json(json &j, SearchContext const &);
void from_json(json const &j, SearchContext &);

} // namespace search
} // namespace mirage
