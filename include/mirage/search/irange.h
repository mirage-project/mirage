#pragma once

#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "range.h"
#include "range_set.h"
#include "tbrange.h"

namespace mirage {
namespace search {

bool constexpr EXP_AS_IDENTITY = true;
bool constexpr EXP_AS_BARRIER = !EXP_AS_IDENTITY;

class IKNRange {
public:
  IKNRange() = default;
  IKNRange(RangeSet<KNRange, size_t> const &range_set);

  void combine(IKNRange const &knrange, bool simplify = true);
  bool is_subrange(IKNRange const &range) const;
  bool is_subrange(Range const &range) const;
  bool is_empty() const;
  bool is_valid() const;
  void simplify();

  static IKNRange point_range(std::vector<int> const &point);

public:
  RangeSet<KNRange, size_t> range_set;
};

IKNRange forward_propagate(IKNRange const &knrange,
                           kernel::KNOperator const &op,
                           size_t opd_idx);

IKNRange backward_propagate(IKNRange const &knrange,
                            kernel::KNOperator const &op,
                            size_t opd_idx);

IKNRange multiplicative_interact(IKNRange const &knrange,
                                 kernel::KNOperator const &op,
                                 size_t opd_idx_from,
                                 size_t opd_idx_to);

std::vector<IKNRange>
    forward_propagate(std::vector<IKNRange> const &input_ranges,
                      kernel::KNOperator const &op);

std::vector<IKNRange>
    backward_propagate(std::vector<IKNRange> const &output_ranges,
                       kernel::KNOperator const &op);

std::vector<IKNRange>
    multiplicative_interact(std::vector<IKNRange> const &input_ranges,
                            kernel::KNOperator const &op);

std::ostream &operator<<(std::ostream &os, IKNRange const &knrange);

class ITBRange {
public:
  ITBRange() = default;
  ITBRange(RangeSet<TBRange, size_t> const &range_set);

  void combine(ITBRange const &tbrange, bool simplify = true);
  ITBRange extend_forloop_dim() const;
  void simplify();
  bool is_empty() const;
  bool is_valid() const;

public:
  RangeSet<TBRange, size_t> range_set;
};

ITBRange forward_propagate(ITBRange const &tbrange,
                           threadblock::TBOperator const &op,
                           size_t opd_idx);

ITBRange backward_propagate(ITBRange const &tbrange,
                            threadblock::TBOperator const &op,
                            size_t opd_idx);

ITBRange multiplicative_interact(ITBRange const &tbrange,
                                 threadblock::TBOperator const &op,
                                 size_t opd_idx_from,
                                 size_t opd_idx_to);

std::vector<ITBRange>
    forward_propagate(std::vector<ITBRange> const &input_ranges,
                      threadblock::TBOperator const &op);

std::vector<ITBRange>
    backward_propagate(std::vector<ITBRange> const &output_ranges,
                       threadblock::TBOperator const &op);

std::vector<ITBRange>
    multiplicative_interact(std::vector<ITBRange> const &input_ranges,
                            threadblock::TBOperator const &op);

std::ostream &operator<<(std::ostream &os, ITBRange const &range);

void range_propagate_forward(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    kernel::Graph const &graph);

void range_propagate_backward(
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    kernel::Graph const &graph);

void interact_range_propagate(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    kernel::Graph const &graph);

void range_propagate_forward(
    std::unordered_map<size_t, ITBRange> &forward_ranges,
    threadblock::Graph const &graph);

void range_propagate_backward(
    std::unordered_map<size_t, ITBRange> &backward_ranges,
    threadblock::Graph const &graph);

void interact_range_propagate(
    std::unordered_map<size_t, ITBRange> &forward_ranges,
    std::unordered_map<size_t, ITBRange> &backward_ranges,
    threadblock::Graph const &graph);

void range_propagate_forward(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    std::unordered_map<size_t, ITBRange> &tb_forward_ranges,
    threadblock::Graph const &graph);

void range_propagate_backward(
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    std::unordered_map<size_t, ITBRange> &tb_backward_ranges,
    threadblock::Graph const &graph);

void interact_range_propagate(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    std::unordered_map<size_t, IKNRange> &backward_ranges,
    threadblock::Graph const &graph);

std::unordered_map<size_t, IKNRange> range_propagate(
    std::unordered_map<size_t, IKNRange> &forward_ranges,
    kernel::Graph const &graph,
    std::shared_ptr<threadblock::Graph const> tb_graph = nullptr);

bool check_range(std::pair<size_t, IKNRange> const &init_range,
                 std::vector<IKNRange> const &target_ranges,
                 kernel::Graph const &graph,
                 std::shared_ptr<threadblock::Graph const> tb_graph = nullptr);

bool check_range(std::vector<std::pair<size_t, IKNRange>> const &init_ranges,
                 std::vector<std::vector<IKNRange>> const &target_ranges,
                 kernel::Graph const &graph,
                 std::shared_ptr<threadblock::Graph const> tb_graph = nullptr);

std::vector<std::pair<size_t, IKNRange>>
    get_init_ranges(kernel::Graph const &graph);

std::vector<IKNRange> get_interact_ranges(
    std::pair<size_t, IKNRange> const &init_range,
    kernel::Graph const &graph,
    std::shared_ptr<threadblock::Graph const> tb_graph = nullptr);

std::vector<std::vector<IKNRange>> get_interact_ranges(
    std::vector<std::pair<size_t, IKNRange>> const &init_ranges,
    kernel::Graph const &graph,
    std::shared_ptr<threadblock::Graph const> tb_graph = nullptr);

} // namespace search
} // namespace mirage
