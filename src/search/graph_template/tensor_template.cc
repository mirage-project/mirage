#include "mirage/search/graph_template/tensor_template.h"

namespace mirage {
namespace search {

DTensorTemplate::DTensorTemplate(std::vector<TensorDimTemplate> dim_templates) : dims(dim_templates) {}

STensorTemplate::STensorTemplate(std::vector<TensorDimTemplate> dim_templates, bool after_accum) : dims(dim_templates), after_accum(after_accum) {}

}
}
