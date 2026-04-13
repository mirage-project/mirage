#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/symbolic_map.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/json_utils.h"

namespace mirage {
namespace search {

KNInputOpArgs::KNInputOpArgs(std::vector<int> input_dims,
                             std::vector<size_t> input_strides,
                             mirage::type::DataType data_type,
                             mirage::layout::DmemLayout layout,
                             int3 input_map)
    : input_dims(input_dims), input_strides(input_strides),
      data_type(data_type), layout(layout), input_map(input_map) {}

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

TBInputOpArgs::TBInputOpArgs(SymbolicDTensor dtensor,
                             SymbolicMap const &input_map)
    : dtensor(dtensor), input_map(input_map) {}

TBOutputOpArgs::TBOutputOpArgs(SymbolicDTensor dtensor,
                               SymbolicMap const &output_map,
                               mirage::type::TBEpilogueType epilogue)
    : dtensor(dtensor), output_map(output_map), epilogue(epilogue) {}

TBConcatOpArgs::TBConcatOpArgs(int concat_dim) : concat_dim(concat_dim) {}

TBElementUnaryOpArgs::TBElementUnaryOpArgs(mirage::type::TBOperatorType op_type,
                                           float scalar)
    : op_type(op_type), scalar(scalar) {}

TBElementBinaryOpArgs::TBElementBinaryOpArgs(
    mirage::type::TBOperatorType op_type)
    : op_type(op_type) {}

TBReductionOpArgs::TBReductionOpArgs(int reduce_dim,
                                     SymbolicTensorDim reduce_degree)
    : reduce_dim(reduce_dim), reduce_degree(reduce_degree) {}

EmptyOpArgs::operator json() const {
  return json{{}};
}

KNInputOpArgs::operator json() const {
  return json{{"input_dims", input_dims}}; // TODO: remove, just for debugging
  // return json{{"input_dims", input_dims}, {"input_strides", input_strides},
  // {"data_type", data_type}, {"layout", layout}, {"input_map", input_map}};
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
  return json{{"dtensor", dtensor}, {"input_map", (json)input_map}};
}

TBOutputOpArgs::operator json() const {
  return json{{"dtensor", dtensor},
              {"output_map", (json)output_map},
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

std::shared_ptr<OpArgs const>
    op_args_from_json_tb(json const &j, type::TBOperatorType op_type) {
  switch (op_type) {
    case type::TBOperatorType::TB_INPUT_OP: {
      SymbolicDTensor dtensor(std::vector<SymbolicTensorDim>{});
      from_json(j.at("dtensor"), dtensor);
      auto const &imap_json = j.at("input_map");
      SymbolicMap input_map(0, 0, std::vector<int>{});
      if (imap_json.is_object() && imap_json.contains("num_grid_dims")) {
        // New SymbolicMap format
        mirage::search::from_json(imap_json, input_map);
      } else {
        // Legacy format: input_map is array of ints, forloop_dim is separate
        // int
        std::vector<int> legacy_map = imap_json.get<std::vector<int>>();
        int forloop_dim = j.at("forloop_dim").get<int>();
        input_map = SymbolicMap(
            legacy_map.size(), dtensor.dims.size(), legacy_map, forloop_dim);
      }
      return std::make_shared<TBInputOpArgs const>(dtensor, input_map);
    }
    case type::TBOperatorType::TB_OUTPUT_OP: {
      SymbolicDTensor dtensor(std::vector<SymbolicTensorDim>{});
      from_json(j.at("dtensor"), dtensor);
      auto const &omap_json = j.at("output_map");
      SymbolicMap output_map(0, 0, std::vector<int>{});
      if (omap_json.is_object() && omap_json.contains("num_grid_dims")) {
        mirage::search::from_json(omap_json, output_map);
      } else {
        std::vector<int> legacy_map = omap_json.get<std::vector<int>>();
        output_map =
            SymbolicMap(legacy_map.size(), dtensor.dims.size(), legacy_map);
      }
      type::TBEpilogueType epilogue =
          j.at("epilogue").get<type::TBEpilogueType>();
      return std::make_shared<TBOutputOpArgs const>(
          dtensor, output_map, epilogue);
    }
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      int concat_dim = j.at("concat_dim").get<int>();
      return std::make_shared<TBConcatOpArgs const>(concat_dim);
    }
    case type::TBOperatorType::TB_EXP_OP:
    case type::TBOperatorType::TB_SQUARE_OP:
    case type::TBOperatorType::TB_SQRT_OP:
    case type::TBOperatorType::TB_SILU_OP:
    case type::TBOperatorType::TB_GELU_OP:
    case type::TBOperatorType::TB_RELU_OP:
    case type::TBOperatorType::TB_MUL_SCALAR_OP:
    case type::TBOperatorType::TB_CLAMP_OP: {
      if (j.contains("op_type") && j.contains("scalar")) {
        type::TBOperatorType op_type_val =
            j.at("op_type").get<type::TBOperatorType>();
        float scalar = j.at("scalar").get<float>();
        return std::make_shared<TBElementUnaryOpArgs const>(op_type_val,
                                                            scalar);
      }
      return std::make_shared<EmptyOpArgs>();
    }
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_MUL_OP:
    case type::TBOperatorType::TB_DIV_OP:
    case type::TBOperatorType::TB_SUB_OP:
    case type::TBOperatorType::TB_POW_OP: {
      if (j.contains("op_type")) {
        type::TBOperatorType op_type_val =
            j.at("op_type").get<type::TBOperatorType>();
        return std::make_shared<TBElementBinaryOpArgs const>(op_type_val);
      }
      return std::make_shared<EmptyOpArgs>();
    }
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP:
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP: {
      int reduce_dim = j.at("reduce_dim").get<int>();
      SymbolicTensorDim reduce_degree;
      from_json(j.at("reduce_degree"), reduce_degree);
      return std::make_shared<TBReductionOpArgs const>(reduce_dim,
                                                       reduce_degree);
    }
    default:
      return std::make_shared<EmptyOpArgs>();
  }
}

std::shared_ptr<OpArgs const>
    op_args_from_json_kn(json const &j, type::KNOperatorType op_type) {
  switch (op_type) {
    case type::KNOperatorType::KN_INPUT_OP: {
      std::vector<int> input_dims = j.at("input_dims").get<std::vector<int>>();
      std::vector<size_t> input_strides;
      if (j.contains("input_strides")) {
        input_strides = j.at("input_strides").get<std::vector<size_t>>();
      } else {
        for (size_t i = 0; i < input_dims.size(); ++i) {
          size_t stride = 1;
          for (size_t k = i + 1; k < input_dims.size(); ++k) {
            stride *= input_dims[k];
          }
          input_strides.push_back(stride);
        }
      }
      type::DataType data_type = j.contains("data_type")
                                     ? j.at("data_type").get<type::DataType>()
                                     : type::DT_FLOAT16;
      layout::DmemLayout layout = j.contains("layout")
                                      ? j.at("layout").get<layout::DmemLayout>()
                                      : layout::DmemRowMajor;
      int3 input_map = j.contains("input_map") ? j.at("input_map").get<int3>()
                                               : int3{-1, -1, -1};
      return std::make_shared<KNInputOpArgs const>(
          input_dims, input_strides, data_type, layout, input_map);
    }
    case type::KNOperatorType::KN_OUTPUT_OP: {
      std::vector<size_t> output_strides =
          j.at("output_strides").get<std::vector<size_t>>();
      int3 output_map = j.at("output_map").get<int3>();
      return std::make_shared<KNOutputOpArgs const>(output_strides, output_map);
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP: {
      int reduction_dim_idx = j.at("reduction_dim_idx").get<int>();
      int reduction_dim_size = j.at("reduction_dim_size").get<int>();
      return std::make_shared<KNReductionOpArgs const>(reduction_dim_idx,
                                                       reduction_dim_size);
    }
    case type::KNOperatorType::KN_RMS_NORM_OP: {
      int normalized_size = j.at("normalized_size").get<int>();
      return std::make_shared<KNRMSNormOpArgs const>(normalized_size);
    }
    case type::KNOperatorType::KN_CUSTOMIZED_OP: {
      json const &jg = j.at("tb_graph_template");
      SymbolicTBGraph tb_graph(0, jg.at("grid_dim").size());
      from_json(jg, tb_graph);
      return std::make_shared<KNCustomizedOpArgs const>(tb_graph);
    }
    default:
      return std::make_shared<EmptyOpArgs>();
  }
}

} // namespace search
} // namespace mirage