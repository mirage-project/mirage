#pragma once

#include "mirage/search/abstract_expr/abstract_expr_for_ops.h"

namespace mirage {
namespace search {

void abstract_expr_eval(
    threadblock::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &exprs);

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &exprs);

void abstract_expr_eval(SymbolicKNGraph const &kn_graph,
                        std::vector<std::shared_ptr<AbstractExpr>> &exprs);

void abstract_expr_eval(
    SymbolicTBGraph const &tb_graph,
    std::vector<std::shared_ptr<AbstractExpr>> const &input_exprs,
    std::vector<std::shared_ptr<AbstractExpr>> &exprs,
    std::vector<std::shared_ptr<AbstractExpr>> &output_exprs);

} // namespace search
} // namespace mirage