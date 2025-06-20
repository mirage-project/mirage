#pragma once

#include "z3++.h"

namespace mirage {

z3::expr_vector to_expr_vector(std::vector<z3::expr> const &_vec);

/*
 * Adapted from
 * https://github.com/Z3Prover/z3/blob/e3566288a48e9e88346b9e8eac44472d1c34aff6/examples/c%2B%2B/example.cpp#L807C1-L834C2
 */
template <typename F>
void visit(std::vector<bool> &visited, z3::expr const &e, F const &func) {
  if (visited.size() <= e.id()) {
    visited.resize(e.id() + 1, false);
  }
  if (visited[e.id()]) {
    return;
  }
  visited[e.id()] = true;

  func(e);

  if (e.is_app()) {
    unsigned num = e.num_args();
    for (unsigned i = 0; i < num; i++) {
      visit(visited, e.arg(i), func);
    }
  } else if (e.is_quantifier()) {
    visit(visited, e.body(), func);
  } else {
    assert(e.is_var());
  }
}

z3::expr_vector get_free_vars(std::vector<z3::expr> const &e,
                              std::vector<z3::sort> const &s);

} // namespace mirage
