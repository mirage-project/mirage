#include "mirage/search/range_propagation/irange.h"
#include "mirage/kernel/rms_norm.h"
#include "mirage/utils/containers.h"

#include <iostream>

namespace mirage {
namespace search {

IKNRange::IKNRange(RangeSet<KNRange, size_t> const &range_set)
    : range_set(range_set) {}

void IKNRange::combine(IKNRange const &range, bool simplify) {
  range_set = range_set.combine(range.range_set);
  if (simplify) {
    this->simplify();
  }
}

bool IKNRange::is_subrange(IKNRange const &knrange) const {
  if (knrange.is_empty()) {
    return this->is_empty();
  }
  Range range = knrange.range_set.get_only();
  return is_subrange(range);
}

bool IKNRange::is_subrange(Range const &range) const {
  for (auto const &r : range_set.ranges) {
    if (!r.is_subrange(range)) {
      return false;
    }
  }
  return true;
}

void IKNRange::simplify() {
  range_set.simplify();
}

bool IKNRange::is_empty() const {
  return range_set.is_empty();
}

bool IKNRange::is_valid() const {
  return range_set.is_valid();
}

IKNRange IKNRange::point_range(std::vector<int> const &point) {
  std::vector<KNRange> ranges{KNRange::point_range(point)};
  std::vector<PropagationPath<size_t>> paths{PropagationPath<size_t>()};
  return IKNRange(RangeSet(ranges, paths));
}

TBRange propagate_from_dtensor_to_stensor(KNRange const &range,
                                          kernel::DTensor const &dtensor,
                                          threadblock::STensor const &stensor,
                                          dim3 grid_dim,
                                          int3 dim_map,
                                          int forloop_dim,
                                          int forloop_range) {
  if (!range.is_valid()) {
    return TBRange(
        Range::invalid_range(), Range::invalid_range(), Range::invalid_range());
  }
  if (range.is_empty()) {
    return TBRange();
  }

  std::vector<int> lower(range.lower), upper(range.upper);
  std::vector<int> blower(3, 0), bupper(3, 0);
  int flower = 0, fupper = forloop_range;

  bool forloop_processed = false;
  for (int d = 0; d < 3; ++d) {
    int dim_idx = -1, dim_deg = 1;
    if (d == 0) {
      dim_idx = dim_map.x;
      dim_deg = grid_dim.x;
    } else if (d == 1) {
      dim_idx = dim_map.y;
      dim_deg = grid_dim.y;
    } else if (d == 2) {
      dim_idx = dim_map.z;
      dim_deg = grid_dim.z;
    }
    if (dim_idx != -1) {
      // NOTE(@Mengdi Wu): We ignore the case where the range contains partial
      // of more than one STensors. In this case, a subset of the range is
      // returned.
      int dim_size = dtensor.dim[dim_idx] / dim_deg;
      assert(dim_idx >= 0);
      assert(dim_idx < (int)range.lower.size());
      if (range.upper[dim_idx] - range.lower[dim_idx] >= dim_size) {
        upper[dim_idx] = stensor.dim[dim_idx];
        blower[d] = range.lower[dim_idx] / dim_size;
        bupper[d] = range.upper[dim_idx] / dim_size;
        if (forloop_dim == dim_idx) {
          fupper = forloop_range;
          forloop_processed = true;
        }
      } else if (range.upper[dim_idx] - range.lower[dim_idx] >=
                 stensor.dim[dim_idx]) {
        upper[dim_idx] = stensor.dim[dim_idx];
        blower[d] = range.lower[dim_idx] / dim_size;
        bupper[d] = blower[d] + 1;
        if (forloop_dim == dim_idx) {
          flower = (range.lower[dim_idx] % stensor.dim[dim_idx]) /
                   stensor.dim[dim_idx];
          fupper = (range.upper[dim_idx] % stensor.dim[dim_idx]) /
                   stensor.dim[dim_idx];
          forloop_processed = true;
        }
      } else {
        lower[dim_idx] = range.lower[dim_idx] % stensor.dim[dim_idx];
        upper[dim_idx] = range.upper[dim_idx] % stensor.dim[dim_idx];
        blower[d] = range.lower[dim_idx] / stensor.dim[dim_idx];
        bupper[d] = blower[d] + 1;
        if (forloop_dim == dim_idx) {
          flower = (range.lower[dim_idx] % stensor.dim[dim_idx]) /
                   stensor.dim[dim_idx];
          fupper = flower + 1;
          forloop_processed = true;
        }
      }
    } else {
      bupper[d] = dim_deg;
    }
  }

  if (!forloop_processed && forloop_dim != -1) {
    if (range.upper[forloop_dim] - range.lower[forloop_dim] >=
        stensor.dim[forloop_dim]) {
      upper[forloop_dim] = stensor.dim[forloop_dim];
      flower = range.lower[forloop_dim] / stensor.dim[forloop_dim];
      fupper = range.upper[forloop_dim] / stensor.dim[forloop_dim];
    } else {
      lower[forloop_dim] = range.lower[forloop_dim] % stensor.dim[forloop_dim];
      upper[forloop_dim] = range.upper[forloop_dim] % stensor.dim[forloop_dim];
      flower = range.lower[forloop_dim] / stensor.dim[forloop_dim];
      fupper = flower + 1;
    }
  }

  return TBRange(
      Range(lower, upper), Range(blower, bupper), Range({flower}, {fupper}));
}
ITBRange propagate_from_dtensor_to_stensor(IKNRange const &range,
                                           kernel::DTensor const &dtensor,
                                           threadblock::STensor const &stensor,
                                           dim3 grid_dim,
                                           int3 dim_map,
                                           int forloop_dim,
                                           int forloop_range) {
  return ITBRange(RangeSet(
      vector_map(range.range_set.ranges,
                 [&](KNRange const &r) {
                   return propagate_from_dtensor_to_stensor(r,
                                                            dtensor,
                                                            stensor,
                                                            grid_dim,
                                                            dim_map,
                                                            forloop_dim,
                                                            forloop_range);
                 }),
      range.range_set.paths));
}

KNRange propagate_from_stensor_to_dtensor(TBRange const &range,
                                          kernel::DTensor const &dtensor,
                                          threadblock::STensor const &stensor,
                                          dim3 grid_dim,
                                          int3 dim_map,
                                          int forloop_dim,
                                          int forloop_range) {
  if (!range.is_valid()) {
    return KNRange::invalid_range();
  }
  if (range.is_empty()) {
    return KNRange();
  }

  std::vector<int> lower(range.tensor_range.lower),
      upper(range.tensor_range.upper);

  bool forloop_processed = false;
  for (int d = 0; d < 3; ++d) {
    int dim_idx = -1, dim_deg = 1;
    if (d == 0) {
      dim_idx = dim_map.x;
      dim_deg = grid_dim.x;
    }
    if (d == 1) {
      dim_idx = dim_map.y;
      dim_deg = grid_dim.y;
    }
    if (d == 2) {
      dim_idx = dim_map.z;
      dim_deg = grid_dim.z;
    }
    if (dim_idx != -1) {
      bool is_all = range.is_all(stensor, dim_idx);
      if (forloop_dim == dim_idx) {
        is_all = is_all && range.forloop_range.is_all(0, forloop_range, 0);
      }
      if (is_all) {
        lower[dim_idx] =
            range.block_range.lower[d] * dtensor.dim[dim_idx] / dim_deg;
        upper[dim_idx] =
            range.block_range.upper[d] * dtensor.dim[dim_idx] / dim_deg;
      } else if (range.block_range.upper[d] - range.block_range.lower[d] == 1) {
        if (forloop_dim == dim_idx) {
          if (range.is_all(stensor, dim_idx)) {
            lower[dim_idx] =
                range.block_range.lower[d] * dtensor.dim[dim_idx] / dim_deg +
                range.forloop_range.lower[0] * stensor.dim[dim_idx];
            upper[dim_idx] =
                range.block_range.lower[d] * dtensor.dim[dim_idx] / dim_deg +
                range.forloop_range.upper[0] * stensor.dim[dim_idx];
          } else if (range.forloop_range.upper[0] -
                         range.forloop_range.lower[0] ==
                     1) {
            lower[dim_idx] =
                range.block_range.lower[d] * dtensor.dim[dim_idx] / dim_deg +
                range.forloop_range.lower[0] * stensor.dim[dim_idx] +
                range.tensor_range.lower[dim_idx];
            upper[dim_idx] =
                range.block_range.lower[d] * dtensor.dim[dim_idx] / dim_deg +
                range.forloop_range.lower[0] * stensor.dim[dim_idx] +
                range.tensor_range.upper[dim_idx];
          } else {
            return KNRange::invalid_range();
          }
        } else {
          lower[dim_idx] =
              range.block_range.lower[d] * dtensor.dim[dim_idx] / dim_deg +
              range.tensor_range.lower[dim_idx];
          upper[dim_idx] =
              range.block_range.lower[d] * dtensor.dim[dim_idx] / dim_deg +
              range.tensor_range.upper[dim_idx];
        }
      } else {
        return KNRange::invalid_range();
      }
    }
  }

  if (!forloop_processed && forloop_dim != -1) {
    if (range.is_all(stensor, forloop_dim)) {
      lower[forloop_dim] =
          range.forloop_range.lower[0] * stensor.dim[forloop_dim];
      upper[forloop_dim] =
          range.forloop_range.upper[0] * stensor.dim[forloop_dim];
    } else if (range.forloop_range.upper[0] - range.forloop_range.lower[0] ==
               1) {
      lower[forloop_dim] =
          range.forloop_range.lower[0] * stensor.dim[forloop_dim] +
          range.tensor_range.lower[forloop_dim];
      upper[forloop_dim] =
          range.forloop_range.lower[0] * stensor.dim[forloop_dim] +
          range.tensor_range.upper[forloop_dim];
    } else {
      return KNRange::invalid_range();
    }
  }

  return KNRange(lower, upper);
}

IKNRange propagate_from_stensor_to_dtensor(ITBRange const &range,
                                           kernel::DTensor const &dtensor,
                                           threadblock::STensor const &stensor,
                                           dim3 grid_dim,
                                           int3 dim_map,
                                           int forloop_dim,
                                           int forloop_range) {
  return IKNRange(RangeSet(
      vector_map(range.range_set.ranges,
                 [&](TBRange const &r) {
                   return propagate_from_stensor_to_dtensor(r,
                                                            dtensor,
                                                            stensor,
                                                            grid_dim,
                                                            dim_map,
                                                            forloop_dim,
                                                            forloop_range);
                 }),
      range.range_set.paths));
}

IKNRange forward_propagate(IKNRange const &range,
                           kernel::KNOperator const &op,
                           size_t opd_idx) {
  IKNRange ret;
  switch (op.op_type) {
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_SQUARE_OP:
    case type::KNOperatorType::KN_SQRT_OP:
    case type::KNOperatorType::KN_SILU_OP:
    case type::KNOperatorType::KN_GELU_OP:
    case type::KNOperatorType::KN_RELU_OP:
    case type::KNOperatorType::KN_CLAMP_OP: {
      ret = EXP_AS_IDENTITY ? range : IKNRange();
      break;
    }
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP: {
      ret = range;
      break;
    }
    case type::KNOperatorType::KN_DIV_OP:
    case type::KNOperatorType::KN_POW_OP: {
      if (opd_idx == 0) {
        ret = range;
      } else {
        assert(opd_idx == 1);
        ret = IKNRange(
            range.range_set.extend_dim(op.output_tensors[0].num_dims - 1)
                .truncate(op.output_tensors[0]));
      }
      break;
    }
    case type::KNOperatorType::KN_ALLREDUCE_OP:
      assert(false && "TBD");
    case type::KNOperatorType::KN_MATMUL_OP: {
      int dim_to_extend = opd_idx == 0 ? op.output_tensors[0].num_dims - 1
                                       : op.output_tensors[0].num_dims - 2;
      ret = IKNRange(range.range_set.extend_dim(dim_to_extend)
                         .truncate(op.output_tensors[0]));
      break;
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP: {
      int dim = op.op_type - type::KNOperatorType::KN_REDUCTION_0_OP;
      ret = IKNRange(
          range.range_set.extend_dim(dim).truncate(op.output_tensors[0]));
      break;
    }
    case type::KNOperatorType::KN_RMS_NORM_OP: {
      int num_norm_dims = 0, s = 1;
      ret = range;
      while (s < static_cast<kernel::KNRMSNormOp const &>(op).normalized_size) {
        num_norm_dims++;
        s *= op.output_tensors[0]
                 .dim[op.output_tensors[0].num_dims - num_norm_dims];
        ret = IKNRange(ret.range_set.extend_dim(op.output_tensors[0].num_dims -
                                                num_norm_dims));
      }
      ret = IKNRange(ret.range_set.truncate(op.output_tensors[0]));
      break;
    }
    default:
      assert(false && "Invalid operator type");
  }
  if (ret.range_set.extend_path(op.output_tensors[0].guid)) {
    return ret;
  } else {
    return IKNRange();
  }
}

IKNRange backward_propagate(IKNRange const &knrange,
                            kernel::KNOperator const &op,
                            size_t opd_idx) {
  IKNRange ret;
  switch (op.op_type) {
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_SQUARE_OP:
    case type::KNOperatorType::KN_SQRT_OP:
    case type::KNOperatorType::KN_SILU_OP:
    case type::KNOperatorType::KN_GELU_OP:
    case type::KNOperatorType::KN_RELU_OP:
    case type::KNOperatorType::KN_CLAMP_OP: {
      ret = EXP_AS_IDENTITY ? knrange : IKNRange();
      break;
    }
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP: {
      ret = knrange;
      break;
    }
    case type::KNOperatorType::KN_ALLREDUCE_OP:
      assert(false && "TBD");
    case type::KNOperatorType::KN_DIV_OP:
    case type::KNOperatorType::KN_POW_OP: {
      ret = IKNRange(
          knrange.range_set.extend_dim(op.input_tensors[opd_idx].num_dims - 1)
              .truncate(op.input_tensors[opd_idx]));
      break;
    }
    case type::KNOperatorType::KN_MATMUL_OP: {
      int dim_to_extend = opd_idx == 0 ? op.input_tensors[0].num_dims - 1
                                       : op.input_tensors[1].num_dims - 2;
      ret = IKNRange(knrange.range_set.extend_dim(dim_to_extend)
                         .truncate(op.input_tensors[opd_idx]));
      break;
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP: {
      int dim = op.op_type - type::KNOperatorType::KN_REDUCTION_0_OP;
      ret = IKNRange(knrange.range_set.extend_dim(dim).truncate(
          op.input_tensors[opd_idx]));
      break;
    }
    case type::KNOperatorType::KN_RMS_NORM_OP: {
      int num_norm_dims = 0, s = 1;
      ret = knrange;
      while (s < static_cast<kernel::KNRMSNormOp const &>(op).normalized_size) {
        num_norm_dims++;
        s *= op.output_tensors[0]
                 .dim[op.output_tensors[0].num_dims - num_norm_dims];
        ret = IKNRange(ret.range_set.extend_dim(op.output_tensors[0].num_dims -
                                                num_norm_dims));
      }
      ret = IKNRange(ret.range_set.truncate(op.output_tensors[0]));
      break;
    }

    default:
      assert(false && "Invalid operator type");
  }
  if (ret.range_set.extend_path(op.input_tensors[opd_idx].guid)) {
    return ret;
  } else {
    return IKNRange();
  }
}

IKNRange multiplicative_interact(IKNRange const &knrange,
                                 kernel::KNOperator const &op,
                                 size_t opd_idx_from,
                                 size_t opd_idx_to) {
  IKNRange ret;
  switch (op.op_type) {
    case type::KNOperatorType::KN_MUL_OP: {
      if (opd_idx_from != opd_idx_to) {
        ret = knrange;
      }
      break;
    }
    case type::KNOperatorType::KN_MATMUL_OP: {
      if (opd_idx_from != opd_idx_to) {
        int num_dims = op.input_tensors[opd_idx_to].num_dims;
        int dim_to_extend = opd_idx_to == 0 ? num_dims - 2 : num_dims - 1;
        ret = IKNRange(knrange.range_set.transpose(num_dims - 2, num_dims - 1)
                           .extend_dim(dim_to_extend)
                           .truncate(op.input_tensors[opd_idx_to]));
      }
      break;
    }
    case type::KNOperatorType::KN_CUSTOMIZED_OP: {
      assert(false && "Invalid operator type");
    }
    default:
      return IKNRange();
  }
  if (ret.range_set.extend_path(op.input_tensors[opd_idx_to].guid)) {
    return ret;
  } else {
    return IKNRange();
  }
}

std::vector<IKNRange>
    forward_propagate(std::vector<IKNRange> const &input_ranges,
                      kernel::KNOperator const &op) {
  if (op.op_type == type::KNOperatorType::KN_OUTPUT_OP) {
    return {};
  }
  if (op.op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
    assert(op.input_tensors.size() == input_ranges.size());
    std::unordered_map<size_t, IKNRange> forward_ranges;
    std::unordered_map<size_t, ITBRange> tb_forward_ranges;

    for (size_t i = 0; i < op.input_tensors.size(); ++i) {
      forward_ranges[op.input_tensors[i].guid] = input_ranges[i];
    }
    range_propagate_forward(
        forward_ranges,
        tb_forward_ranges,
        static_cast<kernel::KNCustomizedOp const &>(op).bgraph);
    std::vector<IKNRange> output_ranges;
    for (auto const &output_tensor : op.output_tensors) {
      output_ranges.push_back(forward_ranges[output_tensor.guid]);
    }
    return output_ranges;
  } else {
    IKNRange output_range;
    for (size_t i = 0; i < input_ranges.size(); ++i) {
      IKNRange output_range_i = forward_propagate(input_ranges[i], op, i);
      output_range.combine(forward_propagate(input_ranges[i], op, i));
    }
    return {output_range};
  }
}

std::vector<IKNRange>
    backward_propagate(std::vector<IKNRange> const &output_ranges,
                       kernel::KNOperator const &op) {
  if (op.op_type == type::KNOperatorType::KN_OUTPUT_OP) {
    return {};
  }
  if (op.op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
    assert(op.output_tensors.size() == output_ranges.size());
    std::unordered_map<size_t, IKNRange> backward_ranges;
    std::unordered_map<size_t, ITBRange> tb_backward_ranges;

    for (size_t i = 0; i < op.output_tensors.size(); ++i) {
      backward_ranges[op.output_tensors[i].guid] = output_ranges[i];
    }
    range_propagate_backward(
        backward_ranges,
        tb_backward_ranges,
        static_cast<kernel::KNCustomizedOp const &>(op).bgraph);
    std::vector<IKNRange> input_ranges;
    for (auto const &input_tensor : op.input_tensors) {
      input_ranges.push_back(backward_ranges[input_tensor.guid]);
    }
    return input_ranges;
  } else {
    assert(output_ranges.size() == 1);
    std::vector<IKNRange> input_ranges;
    for (size_t i = 0; i < op.input_tensors.size(); ++i) {
      input_ranges.push_back(backward_propagate(output_ranges[0], op, i));
    }
    return input_ranges;
  }
}

std::vector<IKNRange>
    multiplicative_interact(std::vector<IKNRange> const &input_ranges,
                            kernel::KNOperator const &op) {
  std::vector<IKNRange> interacted_ranges(input_ranges.size());
  if (op.op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
    assert(op.input_tensors.size() == input_ranges.size());
    std::unordered_map<size_t, IKNRange> forward_ranges;
    for (size_t i = 0; i < op.input_tensors.size(); ++i) {
      forward_ranges[op.input_tensors[i].guid] = input_ranges[i];
    }
    std::unordered_map<size_t, IKNRange> backward_ranges;
    interact_range_propagate(
        forward_ranges,
        backward_ranges,
        static_cast<kernel::KNCustomizedOp const &>(op).bgraph);
    for (size_t i = 0; i < op.input_tensors.size(); ++i) {
      interacted_ranges[i] = backward_ranges[op.input_tensors[i].guid];
    }
  } else {
    for (size_t from = 0; from < op.input_tensors.size(); ++from) {
      for (size_t to = 0; to < op.input_tensors.size(); ++to) {
        interacted_ranges[to].combine(
            multiplicative_interact(input_ranges[from], op, from, to));
      }
    }
  }
  return interacted_ranges;
}

std::ostream &operator<<(std::ostream &os, IKNRange const &knrange) {
  os << "IKNRange(" << knrange.range_set << ")";
  return os;
}

ITBRange::ITBRange(RangeSet<TBRange, size_t> const &range_set)
    : range_set(range_set) {}

void ITBRange::combine(ITBRange const &tbrange, bool simplify) {
  range_set = range_set.combine(tbrange.range_set);
  if (simplify) {
    this->simplify();
  }
}

ITBRange ITBRange::extend_forloop_dim() const {
  std::vector<TBRange> new_ranges;
  for (auto const &range : range_set.ranges) {
    new_ranges.push_back(range.extend_forloop_dim());
  }
  return ITBRange(RangeSet(new_ranges, range_set.paths));
}

void ITBRange::simplify() {
  range_set.simplify();
}

bool ITBRange::is_empty() const {
  return range_set.is_empty();
}

bool ITBRange::is_valid() const {
  return range_set.is_valid();
}

ITBRange forward_propagate(ITBRange const &tbrange,
                           threadblock::TBOperator const &op,
                           size_t opd_idx) {
  ITBRange ret;
  switch (op.op_type) {
    case type::TBOperatorType::TB_EXP_OP:
    case type::TBOperatorType::TB_SQUARE_OP:
    case type::TBOperatorType::TB_SQRT_OP:
    case type::TBOperatorType::TB_SILU_OP:
    case type::TBOperatorType::TB_GELU_OP:
    case type::TBOperatorType::TB_RELU_OP:
    case type::TBOperatorType::TB_CLAMP_OP: {
      ret = EXP_AS_IDENTITY ? tbrange : ITBRange();
      break;
    }
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_MUL_OP: {
      ret = tbrange;
      break;
    }
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      if (opd_idx == 0) {
        ret = tbrange;
      } else {
        int dim = op.op_type - type::TBOperatorType::TB_CONCAT_0_OP;
        std::vector<int> offset(op.output_tensors[0].num_dims, 0);
        offset[dim] = op.input_tensors[0].dim[dim];
        ret = ITBRange(
            tbrange.range_set.offset(offset).truncate(op.output_tensors[0]));
      }
      break;
    }
    case type::TBOperatorType::TB_DIV_OP:
    case type::TBOperatorType::TB_POW_OP: {
      if (opd_idx == 0) {
        ret = tbrange;
      } else {
        assert(opd_idx == 1);
        ret = ITBRange(
            tbrange.range_set.extend_dim(op.output_tensors[0].num_dims - 1)
                .truncate(op.output_tensors[0]));
      }
    }
    case type::TB_MATMUL_OP: {
      int dim_to_extend = opd_idx == 0 ? op.output_tensors[0].num_dims - 1
                                       : op.output_tensors[0].num_dims - 2;
      ret = ITBRange(tbrange.range_set.extend_dim(dim_to_extend)
                         .truncate(op.output_tensors[0]));
      break;
    }
    case type::TB_RMS_NORM_OP: {
      ret = ITBRange(
          tbrange.range_set.extend_dim(op.output_tensors[0].num_dims - 1)
              .truncate(op.output_tensors[0]));
      break;
    }
    case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
      ret = tbrange.extend_forloop_dim();
      break;
    }
    case type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
    case type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
    case type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP:
    case type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      ret = ITBRange(
          tbrange.range_set.extend_dim(op.output_tensors[0].num_dims - 1)
              .truncate(op.output_tensors[0]));
      ret = ret.extend_forloop_dim();
      break;
    }
    case type::TB_INPUT_OP:
    case type::TB_OUTPUT_OP: {
      assert(false && "Invalid Operator Type (should be handled elsewhere)");
    }
    case type::TB_REDUCTION_0_OP:
    case type::TB_REDUCTION_1_OP:
    case type::TB_REDUCTION_2_OP:
    case type::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TB_REDUCTION_2_TO_DIMX_OP: {
      assert(false && "Invalid Operator Type");
      break;
    }
    default:
      assert(false && "Invalid operator type");
  }
  if (ret.range_set.extend_path(op.output_tensors[0].guid)) {
    return ret;
  } else {
    return ITBRange();
  }
}

ITBRange backward_propagate(ITBRange const &tbrange,
                            threadblock::TBOperator const &op,
                            size_t opd_idx) {
  ITBRange ret;
  switch (op.op_type) {
    case type::TBOperatorType::TB_EXP_OP:
    case type::TBOperatorType::TB_SQUARE_OP:
    case type::TBOperatorType::TB_SQRT_OP:
    case type::TBOperatorType::TB_SILU_OP:
    case type::TBOperatorType::TB_GELU_OP:
    case type::TBOperatorType::TB_RELU_OP:
    case type::TBOperatorType::TB_CLAMP_OP: {
      ret = EXP_AS_IDENTITY ? tbrange : ITBRange();
      break;
    }
    case type::TBOperatorType::TB_RMS_NORM_OP: {
      ret = ITBRange(
          tbrange.range_set.extend_dim(op.input_tensors[opd_idx].num_dims - 1)
              .truncate(op.input_tensors[opd_idx]));
      break;
    }
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_MUL_OP: {
      ret = tbrange;
      break;
    }
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      if (opd_idx == 0) {
        ret = tbrange;
      } else {
        int dim = op.op_type - type::TBOperatorType::TB_CONCAT_0_OP;
        std::vector<int> offset(op.input_tensors[0].num_dims, 0);
        offset[dim] = -op.input_tensors[0].dim[dim];
        ret = ITBRange(tbrange.range_set.offset(offset).truncate(
            op.input_tensors[opd_idx]));
      }
      break;
    }
    case type::TBOperatorType::TB_DIV_OP:
    case type::TBOperatorType::TB_POW_OP: {
      ret = ITBRange(
          tbrange.range_set.extend_dim(op.input_tensors[opd_idx].num_dims - 1)
              .truncate(op.input_tensors[opd_idx]));
      break;
    }
    case type::TB_MATMUL_OP: {
      int dim_to_extend = opd_idx == 0 ? op.input_tensors[0].num_dims - 1
                                       : op.input_tensors[1].num_dims - 2;
      ret = ITBRange(tbrange.range_set.extend_dim(dim_to_extend)
                         .truncate(op.input_tensors[opd_idx]));
      break;
    }
    case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
      ret = tbrange.extend_forloop_dim();
      break;
    }
    case type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
    case type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP:
    case type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
    case type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      ret = ITBRange(
          tbrange.range_set.extend_dim(op.input_tensors[opd_idx].num_dims - 1)
              .truncate(op.input_tensors[opd_idx]));
      ret = ret.extend_forloop_dim();
      break;
    }
    case type::TB_INPUT_OP:
    case type::TB_OUTPUT_OP: {
      assert(false && "Invalid Operator Type");
    }
    case type::TB_REDUCTION_0_OP:
    case type::TB_REDUCTION_1_OP:
    case type::TB_REDUCTION_2_OP:
    case type::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TB_REDUCTION_2_TO_DIMX_OP: {
      assert(false && "Invalid Operator Type");
    }
    default:
      assert(false && "Invalid operator type");
  }
  if (ret.range_set.extend_path(op.input_tensors[opd_idx].guid)) {
    return ret;
  } else {
    return ITBRange();
  }
}

ITBRange multiplicative_interact(ITBRange const &range,
                                 threadblock::TBOperator const &op,
                                 size_t opd_idx_from,
                                 size_t opd_idx_to) {
  ITBRange ret;
  switch (op.op_type) {
    case type::TBOperatorType::TB_MUL_OP: {
      if (opd_idx_from != opd_idx_to) {
        ret = range;
      }
      break;
    }
    case type::TBOperatorType::TB_MATMUL_OP: {
      if (opd_idx_from != opd_idx_to) {
        int num_dims = op.input_tensors[opd_idx_to].num_dims;
        int dim_to_extend = opd_idx_to == 0 ? num_dims - 2 : num_dims - 1;
        ret = ITBRange(range.range_set.transpose(num_dims - 2, num_dims - 1)
                           .extend_dim(dim_to_extend)
                           .truncate(op.input_tensors[opd_idx_to]));
      }
      break;
    }
    default:
      return ITBRange();
  }
  if (ret.range_set.extend_path(op.input_tensors[opd_idx_to].guid)) {
    return ret;
  } else {
    return ITBRange();
  }
}

std::vector<ITBRange>
    forward_propagate(std::vector<ITBRange> const &input_ranges,
                      threadblock::TBOperator const &op) {
  ITBRange output_range;
  for (size_t i = 0; i < input_ranges.size(); ++i) {
    output_range.combine(forward_propagate(input_ranges[i], op, i));
  }
  return {output_range};
}

std::vector<ITBRange>
    backward_propagate(std::vector<ITBRange> const &output_ranges,
                       threadblock::TBOperator const &op) {
  assert(output_ranges.size() == 1);
  std::vector<ITBRange> input_ranges;
  for (size_t i = 0; i < op.input_tensors.size(); ++i) {
    input_ranges.push_back(backward_propagate(output_ranges[0], op, i));
  }
  return input_ranges;
}

std::vector<ITBRange>
    multiplicative_interact(std::vector<ITBRange> const &input_ranges,
                            threadblock::TBOperator const &op) {
  std::vector<ITBRange> interacted_ranges(input_ranges.size());
  for (size_t from = 0; from < op.input_tensors.size(); ++from) {
    for (size_t to = 0; to < op.input_tensors.size(); ++to) {
      interacted_ranges[to].combine(
          multiplicative_interact(input_ranges[from], op, from, to));
    }
  }
  return interacted_ranges;
}

std::ostream &operator<<(std::ostream &os, ITBRange const &tbrange) {
  os << "ITBRange(" << tbrange.range_set << ")";
  return os;
}

void range_propagate_forward(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    kernel::Graph const &graph) {
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      continue;
    }
    std::vector<IKNRange> input_ranges;
    for (auto const &input_tensor : op->input_tensors) {
      input_ranges.push_back(forward_ranges[input_tensor.guid]);
    }
    std::vector<IKNRange> output_ranges = forward_propagate(input_ranges, *op);
    assert(output_ranges.size() == op->output_tensors.size());
    for (size_t i = 0; i < op->output_tensors.size(); ++i) {
      forward_ranges[op->output_tensors[i].guid].combine(output_ranges[i]);
    }
  }
}

void range_propagate_backward(
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    kernel::Graph const &graph) {
  for (auto const &op : reversed(graph.operators)) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      continue;
    }
    std::vector<IKNRange> output_ranges;
    for (auto const &output_tensor : op->output_tensors) {
      output_ranges.push_back(backward_ranges[output_tensor.guid]);
    }
    std::vector<IKNRange> input_ranges = backward_propagate(output_ranges, *op);
    assert(input_ranges.size() == op->input_tensors.size());
    for (size_t j = 0; j < op->input_tensors.size(); ++j) {
      backward_ranges[op->input_tensors[j].guid].combine(input_ranges[j]);
    }
  }
}

void interact_range_propagate(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    kernel::Graph const &graph) {
  for (auto const &op : graph.operators) {
    std::vector<IKNRange> input_ranges;
    for (auto const &input_tensor : op->input_tensors) {
      input_ranges.push_back(forward_ranges[input_tensor.guid]);
    }
    std::vector<IKNRange> interacted_ranges =
        multiplicative_interact(input_ranges, *op);
    assert(interacted_ranges.size() == op->input_tensors.size());
    for (size_t i = 0; i < interacted_ranges.size(); ++i) {
      backward_ranges[op->input_tensors[i].guid].combine(interacted_ranges[i]);
    }
  }
}

void range_propagate_forward(
    std::unordered_map<size_t, ITBRange> &forward_ranges,
    threadblock::Graph const &graph) {
  for (auto const &op : graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      continue;
    }
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      continue;
    }
    std::vector<ITBRange> input_ranges;
    for (auto const &input_tensor : op->input_tensors) {
      input_ranges.push_back(forward_ranges[input_tensor.guid]);
    }
    std::vector<ITBRange> output_ranges = forward_propagate(input_ranges, *op);
    assert(output_ranges.size() == op->output_tensors.size());
    for (size_t i = 0; i < op->output_tensors.size(); ++i) {
      forward_ranges[op->output_tensors[i].guid].combine(output_ranges[i]);
    }
  }
}

void range_propagate_backward(
    std::unordered_map<size_t, ITBRange> &backward_ranges,
    threadblock::Graph const &graph) {
  for (auto const &op : reversed(graph.operators)) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      continue;
    }
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      continue;
    }
    std::vector<ITBRange> output_ranges;
    for (auto const &output_tensor : op->output_tensors) {
      output_ranges.push_back(backward_ranges[output_tensor.guid]);
    }
    std::vector<ITBRange> input_ranges = backward_propagate(output_ranges, *op);
    assert(input_ranges.size() == op->input_tensors.size());
    for (size_t i = 0; i < op->input_tensors.size(); ++i) {
      backward_ranges[op->input_tensors[i].guid].combine(input_ranges[i]);
    }
  }
}

void interact_range_propagate(
    std::unordered_map<size_t, ITBRange> &forward_ranges,
    std::unordered_map<size_t, ITBRange> &backward_ranges,
    threadblock::Graph const &graph) {
  for (auto const &op : graph.operators) {
    std::vector<ITBRange> input_ranges;
    for (auto const &input_tensor : op->input_tensors) {
      input_ranges.push_back(forward_ranges[input_tensor.guid]);
    }
    std::vector<ITBRange> interacted_ranges =
        multiplicative_interact(input_ranges, *op);
    assert(interacted_ranges.size() == op->input_tensors.size());
    for (size_t i = 0; i < interacted_ranges.size(); ++i) {
      backward_ranges[op->input_tensors[i].guid].combine(interacted_ranges[i]);
    }
  }
}

void range_propagate_forward(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    std::unordered_map<size_t, ITBRange> &tb_forward_ranges,
    threadblock::Graph const &graph) {
  for (auto const &op : graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      auto const &input_op = static_cast<threadblock::TBInputOp const *>(op);
      tb_forward_ranges[input_op->output_tensors[0].guid].combine(
          propagate_from_dtensor_to_stensor(
              forward_ranges[input_op->dtensor.guid],
              input_op->dtensor,
              input_op->output_tensors[0],
              graph.grid_dim,
              input_op->input_map,
              input_op->forloop_dim,
              graph.forloop_range));
    }
  }
  range_propagate_forward(tb_forward_ranges, graph);
  for (auto const &op : graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      auto const &output_op = static_cast<threadblock::TBOutputOp const *>(op);
      forward_ranges[output_op->dtensor.guid].combine(
          propagate_from_stensor_to_dtensor(
              tb_forward_ranges[output_op->input_tensors[0].guid],
              output_op->dtensor,
              output_op->input_tensors[0],
              graph.grid_dim,
              output_op->output_map,
              output_op->forloop_dim,
              graph.forloop_range));
    }
  }
}

void range_propagate_backward(
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    std::unordered_map<size_t, ITBRange> &tb_backward_ranges,
    threadblock::Graph const &graph) {
  for (auto const &op : graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      auto const &output_op = static_cast<threadblock::TBOutputOp const *>(op);
      tb_backward_ranges[output_op->input_tensors[0].guid].combine(
          propagate_from_dtensor_to_stensor(
              backward_ranges[output_op->dtensor.guid],
              output_op->dtensor,
              output_op->input_tensors[0],
              graph.grid_dim,
              output_op->output_map,
              output_op->forloop_dim,
              graph.forloop_range));
    }
  }
  range_propagate_backward(tb_backward_ranges, graph);
  for (auto const &op : graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      auto const &input_op = static_cast<threadblock::TBInputOp const *>(op);
      backward_ranges[input_op->dtensor.guid].combine(
          propagate_from_stensor_to_dtensor(
              tb_backward_ranges[input_op->output_tensors[0].guid],
              input_op->dtensor,
              input_op->output_tensors[0],
              graph.grid_dim,
              input_op->input_map,
              input_op->forloop_dim,
              graph.forloop_range));
    }
  }
}

void interact_range_propagate(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    threadblock::Graph const &graph) {
  std::unordered_map<size_t, ITBRange> tb_forward_ranges, tb_backward_ranges;
  range_propagate_forward(forward_ranges, tb_forward_ranges, graph);
  interact_range_propagate(tb_forward_ranges, tb_backward_ranges, graph);
  range_propagate_backward(backward_ranges, tb_backward_ranges, graph);
}

std::unordered_map<size_t, IKNRange>
    range_propagate(std::unordered_map<size_t, IKNRange> &forward_ranges,
                    kernel::Graph const &graph,
                    std::shared_ptr<threadblock::Graph const> tb_graph) {
  range_propagate_forward(forward_ranges, graph);
  std::unordered_map<size_t, IKNRange> backward_ranges;
  if (tb_graph) {
    interact_range_propagate(forward_ranges, backward_ranges, *tb_graph);
  }
  interact_range_propagate(forward_ranges, backward_ranges, graph);
  range_propagate_backward(backward_ranges, graph);
  return backward_ranges;
}

bool check_range(std::pair<size_t, IKNRange> const &init_range,
                 std::vector<IKNRange> const &target_ranges,
                 kernel::Graph const &graph,
                 std::shared_ptr<threadblock::Graph const> tb_graph) {
  return true;
  std::vector<IKNRange> interact_ranges =
      get_interact_ranges(init_range, graph, tb_graph);
  assert(interact_ranges.size() == target_ranges.size());
  for (size_t i = 0; i < interact_ranges.size(); ++i) {
    if (!interact_ranges[i].is_subrange(target_ranges[i])) {
      return false;
    }
  }
  return true;
}

bool check_range(std::vector<std::pair<size_t, IKNRange>> const &init_ranges,
                 std::vector<std::vector<IKNRange>> const &target_ranges,
                 kernel::Graph const &graph,
                 std::shared_ptr<threadblock::Graph const> tb_graph) {
  return true;
  for (size_t i = 0; i < init_ranges.size(); ++i) {
    if (!check_range(init_ranges[i], target_ranges[i], graph, tb_graph)) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<size_t, IKNRange>>
    get_init_ranges(kernel::Graph const &graph) {

  auto get_points = [](kernel::DTensor const &dtensor) {
    std::vector<std::vector<int>> points;
    points.push_back(std::vector<int>(dtensor.num_dims, 0));
    {
      std::vector<int> random_point;
      for (int i = 0; i < dtensor.num_dims; ++i) {
        random_point.push_back(rand() % dtensor.dim[i]);
      }
      // points.push_back(random_point);
    }
    return points;
  };

  std::vector<std::pair<size_t, IKNRange>> init_ranges;
  for (size_t i = 0; i < graph.operators.size(); ++i) {
    if (graph.operators[i]->op_type == type::KNOperatorType::KN_INPUT_OP) {
      for (auto const &point :
           get_points(graph.operators[i]->output_tensors[0])) {
        init_ranges.push_back(std::make_pair(i, IKNRange::point_range(point)));
      }
    }
  }
  for (auto &[idx, range] : init_ranges) {
    range.range_set =
        range.range_set.truncate(graph.operators[idx]->output_tensors[0]);
  }
  return init_ranges;
}

std::vector<IKNRange>
    get_interact_ranges(std::pair<size_t, IKNRange> const &init_range,
                        kernel::Graph const &graph,
                        std::shared_ptr<threadblock::Graph const> tb_graph) {
  std::unordered_map<size_t, IKNRange> forward_ranges;
  forward_ranges[graph.operators[init_range.first]->output_tensors[0].guid] =
      init_range.second;

  std::unordered_map<size_t, IKNRange> backward_ranges =
      range_propagate(forward_ranges, graph, tb_graph);

  std::vector<IKNRange> interact_ranges;
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      interact_ranges.push_back(backward_ranges[op->output_tensors[0].guid]);
    }
  }
  return interact_ranges;
}

std::vector<std::vector<IKNRange>> get_interact_ranges(
    std::vector<std::pair<size_t, IKNRange>> const &init_ranges,
    kernel::Graph const &graph,
    std::shared_ptr<threadblock::Graph const> tb_graph) {
  return vector_map(init_ranges,
                    [&](std::pair<size_t, IKNRange> const &init_range) {
                      return get_interact_ranges(init_range, graph, tb_graph);
                    });
}

} // namespace search
} // namespace mirage
