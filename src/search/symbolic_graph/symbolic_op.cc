#include "mirage/search/symbolic_graph/symbolic_op.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/utils/json_utils.h"

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

void from_json(json const &j, SymbolicTBOp &op) {
  op.op_type = j.at("op_type").get<type::TBOperatorType>();
  op.args = op_args_from_json_tb(j.at("args"), op.op_type);
}

void from_json(json const &j, SymbolicKNOp &op) {
  op.op_type = j.at("op_type").get<type::KNOperatorType>();
  op.args = op_args_from_json_kn(j.at("args"), op.op_type);
}

} // namespace search
} // namespace mirage
