#pragma once

#include "mirage/search/verification/verifier.h"

#include <z3++.h>

namespace mirage {
namespace search {

class FormalVerifier : public Verifier {
public:
  FormalVerifier(kernel::Graph const &input_graph);
  virtual OutputMatch verify(kernel::Graph const &graph) override;

private:
  std::vector<z3::expr> input_exprs;
  z3::context ctx;
};

std::vector<z3::expr> get_concrete_exprs(kernel::Graph const &graph, z3::context &ctx);
bool is_equivalent(z3::expr const &lhs, z3::expr const &rhs, z3::context &ctx);

}
}
