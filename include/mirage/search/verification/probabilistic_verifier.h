#pragma once

#include "mirage/search/verification/verifier.h"

#include <mutex>

namespace mirage {
namespace search {

class ProbabilisticVerifier : public Verifier {
public:
    ProbabilisticVerifier(kernel::Graph const &input_graph);
    virtual OutputMatch verify(kernel::Graph const &graph) override;

private:
    std::vector<cpu::CTensor> input_graph_fingerprints;
    std::mutex fp_mutex;
};

}
}