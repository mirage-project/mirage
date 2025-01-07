#include "mirage/search/graph_template/tensor_dim_template.h"

#include <vector>

namespace mirage {
namespace search {

class DTensorTemplate {
public:
  DTensorTemplate(std::vector<TensorDimTemplate> dim_templates);

  std::vector<TensorDimTemplate> dims;
};

class STensorTemplate {
public:
  STensorTemplate(std::vector<TensorDimTemplate> dim_templates, bool after_accum);

  std::vector<TensorDimTemplate> dims;
  bool after_accum;
};

}
}
