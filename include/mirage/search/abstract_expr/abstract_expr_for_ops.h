#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace search {

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::KNOperatorType op,
    std::vector<kernel::DTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds);
std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::TBOperatorType op,
    std::vector<threadblock::STensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds);

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::KNOperatorType op,
    std::vector<SymbolicDTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds,
    SymbolicKNGraph const &g);

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::TBOperatorType op,
    std::vector<SymbolicSTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds,
    SymbolicTBGraph const &g);

} // namespace search
} // namespace mirage