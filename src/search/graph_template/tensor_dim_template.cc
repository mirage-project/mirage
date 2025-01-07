#include "mirage/search/graph_template/tensor_dim_template.h"

namespace mirage {
namespace search {

TensorDimTemplate::TensorDimTemplate(std::shared_ptr<TensorDimExpr> dim_expr) : dim_expr(dim_expr) {}

}
}
