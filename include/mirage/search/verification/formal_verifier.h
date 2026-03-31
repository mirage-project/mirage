#pragma once

#include "mirage/search/verification/verifier.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"

#include <mutex>

extern "C" {
bool check_equiv(char const *lhs, char const *rhs, bool is_symbolic);
}

namespace mirage {
namespace search {

class FormalVerifier : public Verifier {
public:
  FormalVerifier(kernel::Graph const &input_graph);
  virtual OutputMatch verify(kernel::Graph const &graph) override;
  OutputMatch verify_symbolic_graph(SymbolicKNGraph const &graph);
  std::vector<std::pair<SymbolicKNGraph, OutputMatch>>
      verify_symbolic_graph_with_unknown_maps(SymbolicKNGraph const &graph);

private:
  std::vector<std::string> input_exprs;
  std::vector<std::vector<int>> shapes_std;

  static std::mutex formal_verifier_mutex;
};

std::vector<std::string>
    get_concrete_exprs(kernel::Graph const &graph,
                       bool with_output_ops);

std::vector<std::string>
    get_concrete_exprs(SymbolicKNGraph const &graph,
                       bool with_output_ops);

} // namespace search
} // namespace mirage
