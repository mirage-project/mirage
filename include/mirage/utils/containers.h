#pragma once

#include "mirage/vector_types.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

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

template <typename Container>
Container elementwise_add(Container const &c1, Container const &c2) {
  assert(c1.size() == c2.size());
  Container result;
  for (size_t i = 0; i < c1.size(); ++i) {
    result.push_back(c1[i] + c2[i]);
  }
  return result;
}

bool operator==(dim3 const &lhs, dim3 const &rhs);
bool operator==(int3 const &lhs, int3 const &rhs);
std::vector<unsigned int> to_vector(dim3 const &d);
std::vector<int> to_vector(int3 const &d);

int3 vec_to_int3(std::vector<int> const &v);
dim3 vec_to_dim3(std::vector<unsigned int> const &v);

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

template <typename T>
std::unordered_set<T>
    set_union(std::vector<std::unordered_set<T>> const &sets) {
  std::unordered_set<T> s;
  for (auto const &set : sets) {
    for (auto const &x : set) {
      s.insert(x);
    }
  }
  return s;
}

template <typename T>
std::vector<T> pad_vector(std::vector<T> const &v, size_t n, T const &val) {
  if (v.size() >= n) {
    return v;
  }
  std::vector<T> result = v;
  result.resize(n, val);
  return result;
}

template <typename T, typename Acc, typename F>
Acc foldl(std::vector<T> const &vec, Acc init, F f) {
  Acc acc = init;
  for (auto const &elem : vec) {
    acc = f(acc, elem);
  }
  return acc;
}

template <typename T>
std::vector<std::vector<T>>
    cartesian_product(std::vector<std::vector<T>> const &vecs) {
  if (vecs.empty()) {
    return {{}};
  }
  return foldl<std::vector<T>, std::vector<std::vector<T>>>(
      vecs,
      std::vector<std::vector<T>>{{}},
      [](std::vector<std::vector<T>> const &acc, std::vector<T> const &vec) {
        std::vector<std::vector<T>> result;
        for (const auto &x : acc) {
          for (const auto &y : vec) {
            std::vector<T> concatenated = x;
            concatenated.push_back(y);
            result.push_back(std::move(concatenated));
          }
        }
        return result;
      });
}

template <typename T, typename F>
bool all_of(std::vector<T> const &v, F f) {
  for (auto const &x : v) {
    if (!f(x)) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::vector<T> random_sample(std::vector<T> const &v, int k) {
  std::vector<T> result;
  for (int i = 0; i < k; ++i) {
    result.push_back(v[rand() % v.size()]);
  }
  return result;
}
