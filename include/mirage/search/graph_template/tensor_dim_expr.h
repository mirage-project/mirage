#include <memory>
#include <string>

namespace mirage {
namespace search {

class TensorDimExpr {
public:
  TensorDimExpr() = default;
  virtual ~TensorDimExpr() = default;
};

class TensorDimVar : public TensorDimExpr {
public:
  TensorDimVar(int index);
  int index;
};

class TensorDimConst : public TensorDimExpr {
public:
  TensorDimConst(int value);
  int value;
};

class TensorDimAdd : public TensorDimExpr {
public:
  TensorDimAdd(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;
};

class TensorDimDiv : public TensorDimExpr {
public:
  TensorDimDiv(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;
};

}
}
