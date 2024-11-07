#pragma once

#include "mirage/search/verification/verifier.h"

namespace mirage {
namespace search {

class ProbabilisticVerifier : public Verifier {
public:
    ProbabilisticVerifier(kernel::Graph const &input_graph);
    virtual bool verify(kernel::Graph const &graph) override;
};

}
}