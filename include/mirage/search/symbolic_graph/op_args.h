#pragma once

#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/type.h"

#include <vector>
#include <vector_types.h>

namespace mirage {
namespace search {

class OpArgs {
public:
  OpArgs() = default;
  virtual ~OpArgs() = default;
};

class EmptyOpArgs : public OpArgs {
public:
  EmptyOpArgs() = default;
};

class KNInputOpArgs : public OpArgs {
public:
  KNInputOpArgs(std::vector<size_t> input_strides, int3 input_map);
  std::vector<size_t> input_strides;
  int3 input_map;
};

class KNOutputOpArgs : public OpArgs {
public:
  KNOutputOpArgs(std::vector<size_t> output_strides, int3 output_map);
  std::vector<size_t> output_strides;
  int3 output_map;
};

class KNReductionOpArgs : public OpArgs {
public:
  KNReductionOpArgs(int reduction_dim_idx, int reduction_dim_size);
  int reduction_dim_idx, reduction_dim_size;
};

class KNRMSNormOpArgs : public OpArgs {
public:
  KNRMSNormOpArgs(int normalized_size);
  int normalized_size;
};

class SymbolicTBGraph;

class KNCustomizedOpArgs : public OpArgs {
public:
  KNCustomizedOpArgs(SymbolicTBGraph tb_graph_template);
  SymbolicTBGraph tb_graph_template;
};

class TBInputOpArgs : public OpArgs {
public:
  TBInputOpArgs(SymbolicDTensor dtensor, int3 input_map, int forloop_dim);
  SymbolicDTensor dtensor;
  int3 input_map;
  int forloop_dim;
};

class TBOutputOpArgs : public OpArgs {
public:
  TBOutputOpArgs(SymbolicDTensor dtensor, int3 output_map, int forloop_dim, mirage::type::TBEpilogueType epilogue);
  SymbolicDTensor dtensor;
  int3 output_map;
  int forloop_dim;
  mirage::type::TBEpilogueType epilogue;
};

class TBConcatOpArgs : public OpArgs {
public:
  TBConcatOpArgs(int concat_dim);
  int concat_dim;
};

class TBElementUnaryOpArgs : public OpArgs {
public:
  TBElementUnaryOpArgs(mirage::type::TBOperatorType op_type, float scalar);
  mirage::type::TBOperatorType op_type;
  float scalar;
};

class TBElementBinaryOpArgs : public OpArgs {
  TBElementBinaryOpArgs(mirage::type::TBOperatorType op_type);
  mirage::type::TBOperatorType op_type;
};

class TBReductionOpArgs : public OpArgs {
public:
  TBReductionOpArgs(int reduce_dim, int reduce_size);
  int reduce_dim, reduce_size;
};

}
}