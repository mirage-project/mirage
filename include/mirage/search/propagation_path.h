#pragma once

#include <iostream>

namespace mirage {
namespace search {

template <typename T>
class PropagationPath {
public:
  PropagationPath() = default;

  bool is_visited(T const &idx) const {
    return existed.find(idx) != existed.end();
  }

  void visit(T const &idx) {
    existed.insert(idx);
  }

  template <typename U>
  friend std::ostream &operator<<(std::ostream &os,
                                  PropagationPath<U> const &path);

private:
  std::unordered_set<T> existed;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, PropagationPath<T> const &path) {
  os << "Path(visited nodes)={";
  for (auto const &idx : path.existed) {
    os << idx << ",";
  }
  os << "}";
  return os;
}

} // namespace search
} // namespace mirage