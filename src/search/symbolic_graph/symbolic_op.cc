#include "mirage/search/symbolic_graph/symbolic_op.h"
#include "mirage/search/symbolic_graph/op_args.h"

namespace mirage {
namespace search {

SymbolicKNOp::SymbolicKNOp(type::KNOperatorType op_type,
                           std::shared_ptr<OpArgs const> args)
    : op_type(op_type), args(args) {}

SymbolicKNOp::SymbolicKNOp(type::KNOperatorType op_type)
    : op_type(op_type), args(std::make_shared<EmptyOpArgs>()) {}

SymbolicTBOp::SymbolicTBOp(type::TBOperatorType op_type,
                           std::shared_ptr<OpArgs const> args)
    : op_type(op_type), args(args) {}

SymbolicTBOp::SymbolicTBOp(type::TBOperatorType op_type)
    : op_type(op_type), args(std::make_shared<EmptyOpArgs>()) {}

SymbolicKNOp::operator json() const {
  return json{
      {"op_type", op_type},
      {"args", *args},
  };
}

SymbolicTBOp::operator json() const {
  return json{
      {"op_type", op_type},
      {"args", *args},
  };
}

} // namespace search
} // namespace mirage
