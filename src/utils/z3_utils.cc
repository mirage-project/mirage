#include "mirage/utils/z3_utils.h"

#include <iostream>

namespace mirage {

z3::expr_vector to_expr_vector(std::vector<z3::expr> const &_vec) {
  assert(!_vec.empty());
  z3::expr_vector vec(_vec[0].ctx());
  for (auto const &e : _vec) {
    vec.push_back(e);
  }
  return vec;
}

z3::expr_vector get_free_vars(std::vector<z3::expr> const &e,
                              std::vector<z3::sort> const &s) {
  assert(!e.empty());
  z3::expr_vector vars(e[0].ctx());
  std::vector<bool> visited;
  for (auto const &expr : e) {
    visit(visited, expr, [&](z3::expr const &e) {
      if (e.is_const() && e.kind() == Z3_ast_kind::Z3_APP_AST && !e.is_true()) {
        for (auto const &sort : s) {
          if (Z3_is_eq_sort(e.ctx(), e.get_sort(), sort)) {
            vars.push_back(e);
            break;
          }
        }
      }
    });
  }
  return vars;
}

} // namespace mirage
