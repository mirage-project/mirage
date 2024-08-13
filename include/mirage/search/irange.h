#pragma once

#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/operator.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace search {

class Range {
public:
  Range(std::vector<int> lower, std::vector<int> upper);

  bool is_subrange(Range const &range) const;
  bool is_empty_range() const;

  std::vector<int> lower, upper;

  static int const INF = 1e9;

  static Range point_range(std::vector<int> const &point);
  static Range all_range(int num_dims);
  static Range expand_dim(Range const &range, int dim);
  static Range intersect(Range const &range1, Range const &range2);
  static Range offset(Range const &range1, std::vector<int> const &offset);
};

class RangeSet {
public:
  RangeSet() = default;
  RangeSet(std::vector<Range> const &ranges);

  bool is_subrange(Range const &range) const;
  void append(Range const &range);

  std::vector<Range> ranges;

  static RangeSet expand_dim(RangeSet const &ranges, int dim);
  static RangeSet intersect(RangeSet const &ranges1, RangeSet const &ranges2);
  static RangeSet combine(RangeSet const &ranges1, RangeSet const &ranges2);
  static RangeSet offset(RangeSet const &ranges,
                         std::vector<int> const &offset);
};

class IRange {
public:
  IRange(std::vector<int> const &point);
  IRange(RangeSet const &ranges, Range const &block_range);

  static IRange forward_propagate(IRange range,
                                  kernel::KNOperator const &op,
                                  size_t opd_idx);
  static IRange forward_propagate(IRange range,
                                  threadblock::TBOperator const &op,
                                  size_t opd_idx);
  static IRange backward_propagate(IRange range,
                                   kernel::KNOperator const &op,
                                   size_t opd_idx);
  static IRange backward_propagate(IRange range,
                                   threadblock::TBOperator const &op,
                                   size_t opd_idx);
  static IRange multiplicative_interact(IRange range,
                                        kernel::KNOperator const &op,
                                        size_t opd_idx_from,
                                        size_t opd_idx_to);

private:
  RangeSet ranges;
  Range block_range;
};

} // namespace search
} // namespace mirage
