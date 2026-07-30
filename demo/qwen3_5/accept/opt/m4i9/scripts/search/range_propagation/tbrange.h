#pragma once

#include "mirage/threadblock/smem_tensor.h"
#include "range.h"

namespace mirage {
namespace search {

class TBRange {
public:
  TBRange();
  TBRange(Range tensor_range, Range block_range, Range forloop_range);

  bool is_empty() const;
  bool is_valid() const;

  bool is_all(threadblock::STensor const &tensor,
              int dim,
              int forloop_dim = -1,
              int forlooop_range = 1) const;

  bool is_subrange(TBRange const &range) const;

  TBRange truncate(threadblock::STensor const &tensor) const;
  TBRange extend_dim(int dim) const;
  TBRange extend_forloop_dim() const;
  TBRange offset(std::vector<int> const &offset) const;
  TBRange transpose(int dim1, int dim2) const;

public:
  Range tensor_range;
  Range block_range;
  Range forloop_range;
};

std::ostream &operator<<(std::ostream &os, TBRange const &range);

} // namespace search
} // namespace mirage