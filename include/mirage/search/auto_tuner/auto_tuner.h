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

private:
  AutoTunerConfig config;
};

} // namespace search
} // namespace mirage
