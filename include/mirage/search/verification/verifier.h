#pragma once

#include "mirage/kernel/graph.h"

namespace mirage {
namespace search {

class Verifier {
public:
    Verifier() = default;
    virtual bool verify(kernel::Graph const &graph, std::vector<int> const& match) = 0;
    virtual ~Verifier() = default;
};

}
}

