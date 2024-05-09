#include "mirage/search/order.h"

#include <iostream>

namespace mirage {
namespace search {

bool Order::operator<(Order const &other) const {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i < other.v.size() && v[i] < other.v[i]) {
      return true;
    }
    if (i < other.v.size() && v[i] > other.v[i]) {
      return false;
    }
    if (i >= other.v.size()) {
      return false;
    }
  }
  if (v.size() < other.v.size()) {
    return true;
  }
  return type < other.type;
}

bool Order::operator<=(Order const &other) const {
  return !(other < *this);
}

Order::Order(std::vector<int> const &v, int type) : v(v), type(type) {}

} // namespace search
} // namespace mirage
