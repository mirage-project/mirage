#include "mirage/search/symbolic_graph/symbolic_tensor.h"

namespace mirage {
namespace search {

SymbolicDTensor::SymbolicDTensor(std::vector<SymbolicTensorDim> dim_templates) : dims(dim_templates) {}

SymbolicSTensor::SymbolicSTensor(std::vector<SymbolicTensorDim> dim_templates, bool after_accum) : dims(dim_templates), after_accum(after_accum) {}

}
}
