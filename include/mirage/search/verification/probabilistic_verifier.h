#pragma once

#include "mirage/search/verification/verifier.h"

#include <mutex>

namespace mirage {
namespace search {

class ProbabilisticVerifier : public Verifier {
public:
  ProbabilisticVerifier(kernel::Graph const &input_graph);
  virtual OutputMatch verify(kernel::Graph const &graph) override;

  static std::mutex fp_mutex;

private:
  std::vector<cpu::CTensor> input_graph_fingerprints;
};

} // namespace search
} // namespace mirage