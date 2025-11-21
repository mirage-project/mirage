#include "mirage/search/symbolic_graph/op_args.h"

namespace mirage {
namespace search {

KNInputOpArgs::KNInputOpArgs(std::vector<int> input_dims, std::vector<size_t> input_strides, mirage::type::DataType data_type, mirage::layout::DmemLayout layout, int3 input_map)
    : input_dims(input_dims), input_strides(input_strides), data_type(data_type), layout(layout), input_map(input_map) {}

KNOutputOpArgs::KNOutputOpArgs(std::vector<size_t> output_strides,
                               int3 output_map)
    : output_strides(output_strides), output_map(output_map) {}

KNReductionOpArgs::KNReductionOpArgs(int reduction_dim_idx,
                                     int reduction_dim_size)
    : reduction_dim_idx(reduction_dim_idx),
      reduction_dim_size(reduction_dim_size) {}

KNRMSNormOpArgs::KNRMSNormOpArgs(int normalized_size)
    : normalized_size(normalized_size) {}

KNCustomizedOpArgs::KNCustomizedOpArgs(SymbolicTBGraph tb_graph_template)
    : tb_graph_template(tb_graph_template) {}

TBInputOpArgs::TBInputOpArgs(SymbolicDTensor dtensor, std::vector<int> const &input_map, int forloop_dim)
    : dtensor(dtensor), input_map(input_map), forloop_dim(forloop_dim) {}

TBOutputOpArgs::TBOutputOpArgs(SymbolicDTensor dtensor,
                              std::vector<int> const &output_map,
                               mirage::type::TBEpilogueType epilogue)
    : dtensor(dtensor), output_map(output_map),
      epilogue(epilogue) {}

TBConcatOpArgs::TBConcatOpArgs(int concat_dim) : concat_dim(concat_dim) {}

TBElementUnaryOpArgs::TBElementUnaryOpArgs(mirage::type::TBOperatorType op_type,
                                           float scalar)
    : op_type(op_type), scalar(scalar) {}

TBElementBinaryOpArgs::TBElementBinaryOpArgs(
    mirage::type::TBOperatorType op_type)
    : op_type(op_type) {}

TBReductionOpArgs::TBReductionOpArgs(int reduce_dim, SymbolicTensorDim reduce_degree)
    : reduce_dim(reduce_dim), reduce_degree(reduce_degree) {}

EmptyOpArgs::operator json() const {
  return json{{}};
}

KNInputOpArgs::operator json() const {
  return json{{"input_dims", input_dims}}; // TODO: remove, just for debugging
  // return json{{"input_dims", input_dims}, {"input_strides", input_strides}, {"data_type", data_type}, {"layout", layout}, {"input_map", input_map}};
}

KNOutputOpArgs::operator json() const {
  return json{{"output_strides", output_strides}, {"output_map", output_map}};
}

KNReductionOpArgs::operator json() const {
  return json{{"reduction_dim_idx", reduction_dim_idx},
              {"reduction_dim_size", reduction_dim_size}};
}

KNRMSNormOpArgs::operator json() const {
  return json{{"normalized_size", normalized_size}};
}

KNCustomizedOpArgs::operator json() const {
  return json{{"tb_graph_template", tb_graph_template}};
}

TBInputOpArgs::operator json() const {
  return json{{"dtensor", dtensor}, {"input_map", input_map}, {"forloop_dim", forloop_dim}};
}

TBOutputOpArgs::operator json() const {
  return json{{"dtensor", dtensor},
              {"output_map", output_map},
              {"epilogue", epilogue}};
}

TBConcatOpArgs::operator json() const {
  return json{{"concat_dim", concat_dim}};
}

TBElementUnaryOpArgs::operator json() const {
  return json{{"op_type", op_type}, {"scalar", scalar}};
}

TBElementBinaryOpArgs::operator json() const {
  return json{{"op_type", op_type}};
}

TBReductionOpArgs::operator json() const {
  return json{{"reduce_dim", reduce_dim}, {"reduce_degree", *reduce_degree}};
}

} // namespace search
} // namespace mirage