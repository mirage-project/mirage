#include "mirage/search/irange.h"

namespace mirage {
namespace search {

Range::Range(std::vector<int> lower, std::vector<int> upper)
    : lower(lower), upper(upper) {}

bool Range::is_subrange(Range const &range) const {
  for (size_t i = 0; i < lower.size(); ++i) {
    if (lower[i] < range.lower[i] || upper[i] > range.upper[i]) {
      return false;
    }
  }
  return true;
}

bool Range::is_empty_range() const {
  for (int i = 0; i < lower.size(); ++i) {
    if (lower[i] >= upper[i]) {
      return true;
    }
  }
  return false;
}

Range Range::point_range(std::vector<int> const &point) {
  std::vector<int> upper;
  for (int i = 0; i < point.size(); ++i) {
    upper.push_back(point[i] + 1);
  }
  return Range(point, upper);
}

Range Range::all_range(int num_dims) {
  std::vector<int> lower(num_dims, 0);
  std::vector<int> upper(num_dims, INF);
  return Range(lower, upper);
}

Range Range::expand_dim(Range const &range, int dim) {
  std::vector<int> lower = range.lower;
  std::vector<int> upper = range.upper;
  lower[dim] = 0;
  upper[dim] = INF;
  return Range(lower, upper);
}

Range Range::intersect(Range const &range1, Range const &range2) {
  std::vector<int> lower;
  std::vector<int> upper;
  for (size_t i = 0; i < range1.lower.size(); ++i) {
    lower.push_back(std::max(range1.lower[i], range2.lower[i]));
    upper.push_back(std::min(range1.upper[i], range2.upper[i]));
  }
  return Range(lower, upper);
}

Range Range::offset(Range const &range, std::vector<int> const &offset) {
  std::vector<int> lower = range.lower;
  std::vector<int> upper = range.upper;
  for (size_t i = 0; i < lower.size(); ++i) {
    lower[i] += offset[i];
    upper[i] += offset[i];
  }
  return Range(lower, upper);
}

RangeSet::RangeSet(std::vector<Range> const &ranges) : ranges(ranges) {}

bool RangeSet::is_subrange(Range const &range) const {
  for (auto const &r : ranges) {
    if (r.is_subrange(range)) {
      return true;
    }
  }
  return false;
}

void RangeSet::append(Range const &range) {
  ranges.push_back(range);
}

RangeSet RangeSet::expand_dim(RangeSet const &ranges, int dim) {
  std::vector<Range> new_ranges;
  for (auto const &r : ranges.ranges) {
    new_ranges.push_back(Range::expand_dim(r, dim));
  }
  return RangeSet(new_ranges);
}

RangeSet RangeSet::intersect(RangeSet const &ranges1, RangeSet const &ranges2) {
  std::vector<Range> new_ranges;
  for (auto const &r1 : ranges1.ranges) {
    for (auto const &r2 : ranges2.ranges) {
      Range inter = Range::intersect(r1, r2);
      if (!inter.is_empty_range()) {
        new_ranges.push_back(inter);
      }
    }
  }
  return RangeSet(new_ranges);
}

RangeSet RangeSet::combine(RangeSet const &ranges1, RangeSet const &ranges2) {
  std::vector<Range> new_ranges = ranges1.ranges;
  for (auto const &r : ranges2.ranges) {
    new_ranges.push_back(r);
  }
  return RangeSet(new_ranges);
}

RangeSet RangeSet::offset(RangeSet const &ranges,
                          std::vector<int> const &offset) {
  std::vector<Range> new_ranges;
  for (auto const &r : ranges.ranges) {
    new_ranges.push_back(Range::offset(r, offset));
  }
  return RangeSet(new_ranges);
}

IRange::IRange(std::vector<int> const &point)
    : ranges({Range::point_range(point)}), block_range({}, {}) {}

IRange::IRange(RangeSet const &ranges, Range const &block_range)
    : ranges(ranges), block_range(block_range) {}

IRange IRange::forward_propagate(IRange range,
                                 kernel::KNOperator const &op,
                                 size_t opd_idx) {
  switch (op.op_type) {
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_MUL_OP:
      return range;
    case type::KNOperatorType::KN_DIV_OP: {
      if (opd_idx == 0) {
        return range;
      } else {
        assert(opd_idx == 1);
        return IRange(RangeSet::expand_dim(range.ranges,
                                           op.output_tensors[0].num_dims - 1),
                      range.block_range);
      }
    }
    case type::KNOperatorType::KN_ALLREDUCE_OP:
      assert(false && "TBD");
    case type::KNOperatorType::KN_CUSTOMIZED_OP:
      assert(false && "TBD");
    case type::KNOperatorType::KN_MATMUL_OP: {
      if (opd_idx == 0) {
        return IRange(RangeSet::expand_dim(range.ranges, 0), range.block_range);
      } else {
        assert(opd_idx == 1);
        return IRange(RangeSet::expand_dim(range.ranges, 1), range.block_range);
      }
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP: {
      int dim = op.op_type - type::KNOperatorType::KN_REDUCTION_0_OP;
      return IRange(RangeSet::expand_dim(range.ranges, dim), range.block_range);
    }
    default:
      assert(false && "Invalid operator type");
  }
}

IRange IRange::forward_propagate(IRange range,
                                 threadblock::TBOperator const &op,
                                 size_t opd_idx) {
  switch (op.op_type) {
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_EXP_OP:
    case type::TBOperatorType::TB_MUL_OP:
      return range;
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      if (opd_idx == 0) {
        return range;
      }
      int dim = op.op_type - type::TBOperatorType::TB_CONCAT_0_OP;
      std::vector<int> offset(op.output_tensors[0].num_dims, 0);
      offset[dim] = op.input_tensors[0].dim[dim];
      return IRange(RangeSet::offset(range.ranges, offset), range.block_range);
    }
    case type::TBOperatorType::TB_DIV_OP: {
			if (opd_idx == 0) {
				return range;
			} else {
				assert(opd_idx == 1);
				return IRange(RangeSet::expand_dim(range.ranges,
																					 op.output_tensors[0].num_dims - 1),
											range.block_range);
			}
		}
		case type::TB_FORLOOP_ACCUM_OP:
			assert(false && "TBD");
		case type::TB_INPUT_OP:
			assert(false && "TBD");
		case type::TB_OUTPUT_OP:
			assert(false && "TBD");
		case type::TB_REDUCTION_0_OP:
		case type::TB_REDUCTION_1_OP:
		case type::TB_REDUCTION_2_OP: {
			int dim = op.op_type - type::TBOperatorType::TB_REDUCTION_0_OP;
			return IRange(RangeSet::expand_dim(range.ranges, dim), range.block_range);
		}
		case type::TB_REDUCTION_0_TO_DIMX_OP:
		case type::TB_REDUCTION_1_TO_DIMX_OP:
		case type::TB_REDUCTION_2_TO_DIMX_OP: {
			int reduction_dimx = op.bgraph->reduction_dimx;
			int dim = op.op_type - type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP;
			RangeSet ranges;
			for (auto const &r : range.ranges.ranges) {
				if (r.upper[dim] - r.lower[dim] >= reduction_dimx) {
					ranges.append(Range::expand_dim(r, dim));
				} else {
					if (r.lower[dim] % reduction_dimx != 0) {
						std::vector<int> lower = r.lower;
						std::vector<int> upper = r.upper;
						lower[dim] = lower[dim] % reduction_dimx;
						upper[dim] = reduction_dimx;
						ranges.append(Range(lower, upper));
					}
					if (r.upper[dim] % reduction_dimx != 0) {
						std::vector<int> lower = r.lower;
						std::vector<int> upper = r.upper;
						lower[dim] = 0;
						upper[dim] = upper[dim] % reduction_dimx;
						ranges.append(Range(lower, upper));
					}
				}
			}
			return IRange(ranges, range.block_range);
		}
    default:
      assert(false && "Invalid operator type");
  }
}



} // namespace search
} // namespace mirage
