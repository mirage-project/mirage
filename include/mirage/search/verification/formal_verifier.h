#pragma once

#include "mirage/search/verification/verifier.h"

#include <mutex>

extern "C" {
bool check_equiv(char const *lhs, char const *rhs);
}

namespace mirage {
namespace search {

class FormalVerifier : public Verifier {
public:
  FormalVerifier(kernel::Graph const &input_graph);
  virtual OutputMatch verify(kernel::Graph const &graph) override;

private:
  std::vector<std::string> input_exprs;
  std::unordered_set<std::string> all_dims;

  static std::mutex formal_verifier_mutex;
};

std::vector<std::string>
    get_concrete_exprs(kernel::Graph const &graph,
                       bool with_output_ops,
                       std::unordered_set<std::string> &all_dims);

} // namespace search
} // namespace mirage
