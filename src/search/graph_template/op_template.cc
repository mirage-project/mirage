#include "mirage/search/graph_template/op_template.h"
#include "mirage/search/graph_template/op_args.h"

namespace mirage {
namespace search {

KNOpTemplate::KNOpTemplate(type::KNOperatorType op_type, std::shared_ptr<OpArgs> args) : op_type(op_type), args(args) {}

KNOpTemplate::KNOpTemplate(type::KNOperatorType op_type) : op_type(op_type), args(std::make_shared<EmptyOpArgs>()) {}

TBOpTemplate::TBOpTemplate(type::TBOperatorType op_type, std::shared_ptr<OpArgs> args) : op_type(op_type), args(args) {}

TBOpTemplate::TBOpTemplate(type::TBOperatorType op_type) : op_type(op_type), args(std::make_shared<EmptyOpArgs>()) {}

}
}
