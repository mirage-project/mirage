#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/search/verification/output_match.h"

namespace mirage {
namespace search {

class Verifier {
public:
  Verifier() = default;
  virtual OutputMatch verify(kernel::Graph const &graph) = 0;
  virtual ~Verifier() = default;
};

} // namespace search
} // namespace mirage
