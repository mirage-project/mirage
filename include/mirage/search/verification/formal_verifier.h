#pragma once

#include "mirage/search/verification/verifier.h"

#include <mutex>
#include <z3++.h>

namespace mirage {
namespace search {

class FormalVerifier : public Verifier {
public:
  FormalVerifier(kernel::Graph const &input_graph);
  virtual OutputMatch verify(kernel::Graph const &graph) override;

private:
  std::vector<z3::expr> input_exprs;
  z3::context _ctx;
  std::unordered_set<std::string> all_dims;

  static std::mutex formal_verifier_mutex;
};

std::vector<z3::expr>
    get_concrete_exprs(kernel::Graph const &graph,
                       z3::context &ctx,
                       bool with_output_ops,
                       std::unordered_set<std::string> &all_dims);
bool is_equivalent(z3::expr const &lhs,
                   z3::expr const &rhs,
                   z3::context &ctx,
                   std::unordered_set<std::string> const &all_dims);

} // namespace search
} // namespace mirage
