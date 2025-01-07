#include "mirage/search/graph_template/op_template.h"

namespace mirage {
namespace search {

KNOpTemplate::KNOpTemplate(type::KNOperatorType op_type, std::shared_ptr<OpArgs> args) : op_type(op_type), args(args) {}
TBOpTemplate::TBOpTemplate(type::TBOperatorType op_type, std::shared_ptr<OpArgs> args) : op_type(op_type), args(args) {}

}
}
