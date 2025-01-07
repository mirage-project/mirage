#include "mirage/search/graph_template/tensor_template.h"
#include "mirage/search/graph_template/op_template.h"
#include "mirage/search/graph_template/tensor_dim_constraint.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"

#include <vector_types.h>

namespace mirage {
namespace search {

class TBGraphTemplate {
public:
  TBGraphTemplate();

  mirage::threadblock::Graph to_threadblock_graph();
  bool add_operator(type::TBOperatorType op_type, std::vector<int> input_indices);

  dim3 grid_dim, block_dim;
  int forloop_range;
  int reduction_dimx;
  std::vector<TBOpTemplate> operators;
  std::vector<STensorTemplate> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  std::vector<TensorDimConstraint> conds;
};

class KNGraphTemplate {
public:
  KNGraphTemplate() = default;

  mirage::kernel::Graph to_kernel_graph();
  bool add_operator(type::KNOperatorType op_type, std::vector<int> input_indices);
  bool add_customized_operator(TBGraphTemplate tb_graph, std::vector<int> input_indices); 

  std::vector<KNOpTemplate> operators;
  std::vector<DTensorTemplate> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  std::vector<TensorDimConstraint> conds;
};

}
}