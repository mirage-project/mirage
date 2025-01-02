#include "mirage/search/range_propagation/tbrange.h"

namespace mirage {
namespace search {

TBRange::TBRange() {}

TBRange::TBRange(Range tensor_range, Range block_range, Range forloop_range)
    : tensor_range(tensor_range), block_range(block_range),
      forloop_range(forloop_range) {}

bool TBRange::is_empty() const {
  return tensor_range.is_empty() || block_range.is_empty() ||
         forloop_range.is_empty();
}

bool TBRange::is_valid() const {
  return tensor_range.is_valid() && block_range.is_valid() &&
         forloop_range.is_valid();
}

bool TBRange::is_all(threadblock::STensor const &tensor,
                     int dim,
                     int forloop_dim,
                     int forlooop_range) const {
  if (forloop_dim == dim) {
    return tensor_range.is_all(tensor, dim) &&
           forloop_range.is_all(0, forlooop_range, 0);
  } else {
    return tensor_range.is_all(tensor, dim);
  }
}

TBRange TBRange::truncate(threadblock::STensor const &tensor) const {
  return TBRange(tensor_range.truncate(tensor), block_range, forloop_range);
}

TBRange TBRange::extend_dim(int dim) const {
  return TBRange(tensor_range.extend_dim(dim), block_range, forloop_range);
}

TBRange TBRange::extend_forloop_dim() const {
  return TBRange(tensor_range, block_range, forloop_range.extend_dim(0));
}

TBRange TBRange::offset(std::vector<int> const &offset) const {
  return TBRange(tensor_range.offset(offset), block_range, forloop_range);
}

TBRange TBRange::transpose(int dim1, int dim2) const {
  return TBRange(
      tensor_range.transpose(dim1, dim2), block_range, forloop_range);
}

bool TBRange::is_subrange(TBRange const &range) const {
  return tensor_range.is_subrange(range.tensor_range) &&
         block_range.is_subrange(range.block_range) &&
         forloop_range.is_subrange(range.forloop_range);
}

std::ostream &operator<<(std::ostream &os, TBRange const &range) {
  os << range.tensor_range << "|" << range.block_range << "|"
     << range.forloop_range;
  return os;
}

} // namespace search
} // namespace mirage