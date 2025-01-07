#include "mirage/search/graph_template/tensor_template.h"

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
  std::vector<size_t> input_strides;
  int3 input_map;
};

class KNOutputOpArgs : public OpArgs {
public:
  std::vector<size_t> output_strides;
  int3 output_map;
};

class KNReductionOpArgs : public OpArgs {
public:
  int reduction_dim_idx, reduction_dim_size;
};

class KNRMSNormOpArgs : public OpArgs {
public:
  int normalized_size;
};

class TBGraphTemplate;

class KNCustomizedOpArgs : public OpArgs {
public:
  TBGraphTemplate tb_graph_template;
};

class TBInputOpArgs : public OpArgs {
public:
  DTensorTemplate dtensor;
  int3 input_map;
  int forloop_dim;
};

class TBOutputOpArgs : public OpArgs {
public:
  DTensorTemplate dtensor;
  int3 output_map;
  int forloop_dim;
  mirage::type::TBEpilogueType epilogue;
};

class TBConcatOpArgs : public OpArgs {
public:
  int concat_dim;
};

class TBElementUnaryOpArgs : public OpArgs {
public:
  float scalar;
};

class TBReductionOpArgs : public OpArgs {
public:
  int reduce_dim, reduce_size;
};

}
}