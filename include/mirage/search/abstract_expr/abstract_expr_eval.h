#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace search {

void abstract_expr_eval(
    threadblock::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns);

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns);

} // namespace search
} // namespace mirage