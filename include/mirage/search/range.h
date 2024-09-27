#pragma once

#include <cassert>
#include <iostream>
#include <vector>

namespace mirage {
namespace search {

class Range {
public:
  Range(bool valid = true);
  Range(std::vector<int> lower, std::vector<int> upper, bool valid = true);

  bool is_subrange(Range const &range) const;
  bool is_empty() const;
  bool is_valid() const;

  bool is_all(int l, int r, int dim) const {
    assert(dim >= 0);
    assert(dim < (int)lower.size());
    return lower[dim] <= l && upper[dim] >= r;
  }

  template <typename Tensor>
  bool is_all(Tensor const &tensor, int dim) const {
    assert(dim >= 0);
    assert(dim < tensor.num_dims);
    return is_all(0, tensor.dim[dim], dim);
  }

  template <typename Tensor>
  Range truncate(Tensor const &tensor) const {
    std::vector<int> new_lower, new_upper;
    for (size_t i = 0; i < lower.size(); ++i) {
      new_lower.push_back(std::max(lower[i], 0));
      new_upper.push_back(std::min(upper[i], tensor.dim[i]));
    }
    return Range(new_lower, new_upper);
  }

  Range extend_dim(int dim) const;
  Range offset(std::vector<int> const &offset) const;
  Range transpose(int dim1, int dim2) const;

  static Range point_range(std::vector<int> const &point);
  static Range all_range(int num_dims);
  static Range empty_range();
  static Range invalid_range();

  static int constexpr INF = 1e9;

public:
  std::vector<int> lower, upper;
  bool valid;
};

std::ostream &operator<<(std::ostream &os, Range const &range);

using KNRange = Range;

} // namespace search
} // namespace mirage