#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"

namespace mirage {
namespace search {

class AutoTuner {
public:
  AutoTuner() = default;

  // Tune a single TB graph: grid search with shape-aware candidates,
  // parallel compilation, sequential profiling.
  DimVarAssignment tune_tb(SymbolicTBGraph const &symbolic_tb_graph);

  // Tune a single KN graph: calls tune_tb for each customized op.
  DimVarAssignment tune_kn(SymbolicKNGraph const &symbolic_kn_graph);

  // Tune multiple KN graphs in parallel (one thread per graph),
  // then compile and profile to pick the best.
  // This is the main entry point for SSO auto-tuning.
  kernel::Graph *tune(std::vector<SymbolicKNGraph> const &symbolic_kn_graphs);
};

} // namespace search
} // namespace mirage
