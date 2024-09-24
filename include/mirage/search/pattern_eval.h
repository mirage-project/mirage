#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/search/algebraic_pattern.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace search {

void pattern_eval(
    threadblock::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> &patterns);

void pattern_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> &patterns);

} // namespace search
} // namespace mirage