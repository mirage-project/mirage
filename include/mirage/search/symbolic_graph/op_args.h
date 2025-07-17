#pragma once

#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/search/symbolic_graph/symbolic_map.h"
#include "mirage/type.h"

#include <vector>
#include <vector_types.h>

namespace mirage {
namespace search {

class OpArgs {
public:
  OpArgs() = default;
  virtual ~OpArgs() = default;

  virtual operator json() const = 0;
};

class EmptyOpArgs : public OpArgs {
public:
  EmptyOpArgs() = default;

  operator json() const override;
};

class KNInputOpArgs : public OpArgs {
public:
  KNInputOpArgs(std::vector<size_t> input_strides, int3 input_map);
  std::vector<size_t> input_strides;
  int3 input_map;

  operator json() const override;
};

class KNOutputOpArgs : public OpArgs {
public:
  KNOutputOpArgs(std::vector<size_t> output_strides, int3 output_map);
  std::vector<size_t> output_strides;
  int3 output_map;

  operator json() const override;
};

class KNReductionOpArgs : public OpArgs {
public:
  KNReductionOpArgs(int reduction_dim_idx, int reduction_dim_size);
  int reduction_dim_idx, reduction_dim_size;

  operator json() const override;
};

class KNRMSNormOpArgs : public OpArgs {
public:
  KNRMSNormOpArgs(int normalized_size);
  int normalized_size;

  operator json() const override;
};

class SymbolicTBGraph;

class KNCustomizedOpArgs : public OpArgs {
public:
  KNCustomizedOpArgs(SymbolicTBGraph tb_graph_template);
  SymbolicTBGraph tb_graph_template;

  operator json() const override;
};

class TBInputOpArgs : public OpArgs {
public:
  TBInputOpArgs(SymbolicDTensor dtensor, SymbolicMap const &imap);
  SymbolicDTensor dtensor;
  SymbolicMap imap;

  operator json() const override;
};

class TBOutputOpArgs : public OpArgs {
public:
  TBOutputOpArgs(SymbolicDTensor dtensor,
                 int3 output_map,
                 int forloop_dim,
                 mirage::type::TBEpilogueType epilogue);
  SymbolicDTensor dtensor;
  int3 output_map;
  int forloop_dim;
  mirage::type::TBEpilogueType epilogue;

  operator json() const override;
};

class TBConcatOpArgs : public OpArgs {
public:
  TBConcatOpArgs(int concat_dim);
  int concat_dim;

  operator json() const override;
};

class TBElementUnaryOpArgs : public OpArgs {
public:
  TBElementUnaryOpArgs(mirage::type::TBOperatorType op_type, float scalar);
  mirage::type::TBOperatorType op_type;
  float scalar;

  operator json() const override;
};

class TBElementBinaryOpArgs : public OpArgs {
  TBElementBinaryOpArgs(mirage::type::TBOperatorType op_type);
  mirage::type::TBOperatorType op_type;

  operator json() const override;
};

class TBReductionOpArgs : public OpArgs {
public:
  TBReductionOpArgs(int reduce_dim, int reduce_size);
  int reduce_dim, reduce_size;

  operator json() const override;
};

} // namespace search
} // namespace mirage