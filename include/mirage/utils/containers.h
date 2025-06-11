#pragma once

#include <algorithm>
#include <unordered_set>
#include <vector>
#include <vector_types.h>

template <typename Container>
typename Container::const_iterator
    find(Container const &c, typename Container::value_type const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

template <typename Container>
bool contains(Container const &c, typename Container::value_type const &e) {
  return find<Container>(c, e) != c.cend();
}

template <typename C>
bool contains_key(C const &m, typename C::key_type const &k) {
  return m.count(k) > 0;
}

template <typename T>
bool contains(std::vector<T> const &v, T const &e) {
  for (auto const &x : v) {
    if (x == e) {
      return true;
    }
  }
  return false;
}

bool operator==(dim3 const &lhs, dim3 const &rhs);
bool operator==(int3 const &lhs, int3 const &rhs);

template <typename T>
std::vector<T> to_vector(int n, T *arr) {
  std::vector<T> v;
  for (int i = 0; i < n; ++i) {
    v.push_back(arr[i]);
  }
  return v;
}

template <typename T>
std::vector<T> vector_concat(std::vector<T> const &v1,
                             std::vector<T> const &v2) {
  std::vector<T> v = v1;
  v.insert(v.end(), v2.begin(), v2.end());
  return v;
}

template <typename T, typename F>
std::vector<std::invoke_result_t<F, T>> vector_map(std::vector<T> const &v,
                                                   F f) {
  std::vector<std::invoke_result_t<F, T>> new_v;
  for (auto const &x : v) {
    new_v.push_back(f(x));
  }
  return new_v;
}

template <typename T, typename F>
T filter(T const &c, F f) {
  T new_c;
  for (auto const &x : c) {
    if (f(x)) {
      new_c.push_back(x);
    }
  }
  return new_c;
}

template <typename T>
std::vector<T> deduplicate(std::vector<T> const &v) {
  std::unordered_set<T> s(v.begin(), v.end());
  return std::vector<T>(s.begin(), s.end());
}

template <typename T>
struct _reversed {
  T &iter;

  auto begin() const {
    return iter.rbegin();
  }

  auto end() const {
    return iter.rend();
  }
};

template <typename T>
_reversed<T> reversed(T &&iter) {
  return _reversed<T>{iter};
}

template <typename T>
std::unordered_set<T> set_union(std::unordered_set<T> const &lhs,
                                std::unordered_set<T> const &rhs) {
  std::unordered_set<T> s = lhs;
  for (auto const &x : rhs) {
    s.insert(x);
  }
  return s;
}
