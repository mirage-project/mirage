#pragma once

#include "mirage/utils/containers.h"
#include "propagation_path.h"
#include "range.h"

namespace mirage {
namespace search {

template <typename R, typename T>
class RangeSet {
public:
  RangeSet() = default;
  RangeSet(std::vector<R> const &ranges,
           std::vector<PropagationPath<T>> const &paths)
      : ranges(ranges), paths(paths) {}

  bool is_empty() const {
    for (auto const &r : ranges) {
      if (!r.is_empty()) {
        return false;
      }
    }
    return true;
  }

  bool is_valid() const {
    for (auto const &r : ranges) {
      if (!r.is_valid()) {
        return false;
      }
    }
    return true;
  }

  void append(R const &range, PropagationPath<T> const &path) {
    ranges.push_back(range);
    paths.push_back(path);
  }

  bool extend_path(T const &n) {
    for (auto &path : paths) {
      if (path.is_visited(n)) {
        return false;
      }
    }
    for (auto &path : paths) {
      path.visit(n);
    }
    return true;
  }

  template <typename Tensor>
  RangeSet<R, T> truncate(Tensor const &tensor) const {
    std::vector<R> new_ranges;
    for (auto const &range : ranges) {
      new_ranges.push_back(range.truncate(tensor));
    }
    return RangeSet<R, T>(new_ranges, paths);
  }

  R get_only() const {
    assert(ranges.size() == 1);
    return ranges[0];
  }

  RangeSet<R, T> extend_dim(int dim) const {
    std::vector<R> new_ranges;
    for (auto const &r : ranges) {
      new_ranges.push_back(r.extend_dim(dim));
    }
    return RangeSet<R, T>(new_ranges, paths);
  }

  RangeSet<R, T> combine(RangeSet const &ranges2) const {
    return RangeSet<R, T>(vector_concat(ranges, ranges2.ranges),
                          vector_concat(paths, ranges2.paths))
        .simplify();
  }

  RangeSet<R, T> offset(std::vector<int> const &off) const {
    return RangeSet<R, T>(
        vector_map(ranges, [&](R const &r) { return r.offset(off); }), paths);
  }

  RangeSet<R, T> transpose(int dim1, int dim2) const {
    return RangeSet<R, T>(
        vector_map(ranges, [&](R const &r) { return r.transpose(dim1, dim2); }),
        paths);
  }

  RangeSet<R, T> simplify() const {
    std::vector<R> new_ranges;
    std::vector<PropagationPath<T>> new_paths;
    for (size_t i = 0; i < ranges.size(); ++i) {
      bool is_subrange = false;
      for (size_t j = 0; j < ranges.size(); ++j) {
        if (i == j) {
          continue;
        }
        if (ranges[i].is_subrange(ranges[j])) {
          if ((i > j) || !ranges[j].is_subrange(ranges[i])) {
            is_subrange = true;
            break;
          }
        }
      }
      if (!is_subrange) {
        new_ranges.push_back(ranges[i]);
        new_paths.push_back(paths[i]);
      }
    }
    return RangeSet(new_ranges, new_paths);
  }

public:
  std::vector<R> ranges;
  std::vector<PropagationPath<T>> paths;
};

template <typename R, typename T>
std::ostream &operator<<(std::ostream &os, RangeSet<R, T> const &ranges) {
  for (size_t i = 0; i < ranges.ranges.size(); ++i) {
    os << ranges.ranges[i] << "|" << ranges.paths[i];
  }
  return os;
}

} // namespace search
} // namespace mirage