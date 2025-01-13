#pragma once

#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/type.h"

namespace mirage {
namespace search {

class OpArgs;

class SymbolicKNOp {
public:
  SymbolicKNOp(type::KNOperatorType op_type, std::shared_ptr<OpArgs> args);
  SymbolicKNOp(type::KNOperatorType op_type);

  type::KNOperatorType op_type;
  std::shared_ptr<OpArgs> args;
};

class SymbolicTBOp {
public:
  SymbolicTBOp(type::TBOperatorType op_type, std::shared_ptr<OpArgs> args);
  SymbolicTBOp(type::TBOperatorType op_type);

  type::TBOperatorType op_type;
  std::shared_ptr<OpArgs> args;
};

}
}