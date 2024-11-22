#include "mirage/search/range_propagation/range.h"

#include <algorithm>

namespace mirage {
namespace search {

Range::Range(bool valid) : valid(valid) {}

Range::Range(std::vector<int> lower, std::vector<int> upper, bool valid)
    : lower(lower), upper(upper), valid(valid) {
  assert(lower.size() == upper.size());
}

bool Range::is_subrange(Range const &range) const {
  if (!is_valid() || !range.is_valid()) {
    return false;
  }
  if (is_empty()) {
    return true;
  }
  if (range.is_empty()) {
    return false;
  }
  assert(lower.size() == range.lower.size());
  for (size_t i = 0; i < lower.size(); ++i) {
    if (lower[i] < range.lower[i] || upper[i] > range.upper[i]) {
      return false;
    }
  }
  return true;
}

bool Range::is_empty() const {
  if (lower.size() == 0) {
    return true;
  }
  for (size_t i = 0; i < lower.size(); ++i) {
    if (lower[i] >= upper[i]) {
      return true;
    }
  }
  return false;
}

bool Range::is_valid() const {
  return valid;
}

Range Range::point_range(std::vector<int> const &point) {
  std::vector<int> upper;
  for (size_t i = 0; i < point.size(); ++i) {
    upper.push_back(point[i] + 1);
  }
  return Range(point, upper);
}

Range Range::all_range(int num_dims) {
  std::vector<int> lower(num_dims, 0);
  std::vector<int> upper(num_dims, INF);
  return Range(lower, upper);
}

Range Range::extend_dim(int dim) const {
  if (is_empty()) {
    return Range();
  }
  std::vector<int> lower_ = lower;
  std::vector<int> upper_ = upper;
  lower_[dim] = 0;
  upper_[dim] = INF;
  return Range(lower_, upper_, valid);
}

Range Range::offset(std::vector<int> const &offset) const {
  std::vector<int> lower_ = lower;
  std::vector<int> upper_ = upper;
  for (size_t i = 0; i < lower.size(); ++i) {
    lower_[i] += offset[i];
    upper_[i] += offset[i];
  }
  return Range(lower_, upper_, valid);
}

Range Range::transpose(int dim1, int dim2) const {
  if (is_empty()) {
    return Range();
  }
  assert(dim1 < (int)lower.size());
  assert(dim2 < (int)lower.size());
  std::vector<int> lower_ = lower;
  std::vector<int> upper_ = upper;
  std::swap(lower_[dim1], lower_[dim2]);
  std::swap(upper_[dim1], upper_[dim2]);
  return Range(lower_, upper_, valid);
}

Range Range::empty_range() {
  return Range();
}

Range Range::invalid_range() {
  return Range(false);
}

std::ostream &operator<<(std::ostream &os, Range const &range) {
  os << "[";
  for (size_t i = 0; i < range.lower.size(); ++i) {
    os << range.lower[i] << "-" << range.upper[i];
    if (i + 1 < range.lower.size()) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

} // namespace search
} // namespace mirage