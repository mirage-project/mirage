#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"

namespace mirage {
namespace search {

struct AutoTunerConfig {
};

class AutoTuner {
public:
  AutoTuner(AutoTunerConfig const &config);
  DimVarAssignment tune(SymbolicTBGraph const &symbolic_tb_graph);
  DimVarAssignment tune(SymbolicKNGraph const &symbolic_kn_graph);
  kernel::Graph *tune(std::vector<SymbolicKNGraph> const &symbolic_kn_graphs);

  /** Tune multiple graphs in parallel: one thread per graph, each calls tune()
   *  on a single SymbolicKNGraph. GPU profiling is serialized (critical section)
   *  to avoid resource conflict; CUDA compilation can run in parallel.
   *  Returns the first graph tuned with its assignment. */
  kernel::Graph *tune_multi_threaded(std::vector<SymbolicKNGraph> const &symbolic_kn_graphs);

private:
  AutoTunerConfig config;
};

} // namespace search
} // namespace mirage
