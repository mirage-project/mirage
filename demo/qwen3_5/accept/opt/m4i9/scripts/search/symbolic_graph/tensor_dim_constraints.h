#pragma once

#include "mirage/search/symbolic_graph/tensor_dim_constraint.h"

#include <vector>

namespace mirage {
namespace search {

// TODO: move to utils
template <typename T>
class RevertableSet {
  std::unordered_set<T> _set;
  std::vector<std::vector<T>> _history;

public:
  RevertableSet() = default;
  template <typename Iter>
  bool insert(Iter begin, Iter end) {
    std::vector<T> elems_inserted;
    for (auto it = begin; it != end; ++it) {
      if (_set.insert(*it).second) {
        elems_inserted.push_back(*it);
      }
    }
    _history.push_back(elems_inserted);
    return elems_inserted.size() > 0;
  }
  bool insert(std::initializer_list<T> const &elems) {
    return this->insert(elems.begin(), elems.end());
  }
  bool insert(T const &elem) {
    return this->insert({elem});
  }
  bool revert() {
    if (_history.size() == 0) {
      return false;
    }
    for (T const &elem : _history.back()) {
      _set.erase(elem);
    }
    _history.pop_back();
    return true;
  }

  using const_iterator = typename std::unordered_set<T>::const_iterator;
  const_iterator begin() const {
    return _set.begin();
  }
  const_iterator end() const {
    return _set.end();
  }
};

class TensorDimConstraints {
public:
  TensorDimConstraints() = default;
  bool revert();
  bool add_constraint(TensorDimConstraint const &constraint);
  template <typename Iter>
  bool add_constraints(Iter begin, Iter end);
  bool add_constraints(
      std::initializer_list<TensorDimConstraint> const &constraints);
  bool add_constraints(
      std::unordered_set<TensorDimConstraint> const &constraints);
  bool satisfiable() const;

  operator json() const;

private:
  RevertableSet<TensorDimConstraint> constraints;
};

} // namespace search
} // namespace mirage
