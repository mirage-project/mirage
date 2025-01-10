#pragma once

#include "mirage/search/graph_template/tensor_template.h"
#include "mirage/type.h"

namespace mirage {
namespace search {

class OpArgs;

class KNOpTemplate {
public:
  KNOpTemplate(type::KNOperatorType op_type, std::shared_ptr<OpArgs> args);
  KNOpTemplate(type::KNOperatorType op_type);

  type::KNOperatorType op_type;
  std::shared_ptr<OpArgs> args;
};

class TBOpTemplate {
public:
  TBOpTemplate(type::TBOperatorType op_type, std::shared_ptr<OpArgs> args);
  TBOpTemplate(type::TBOperatorType op_type);

  type::TBOperatorType op_type;
  std::shared_ptr<OpArgs> args;
};

}
}