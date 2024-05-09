#pragma once

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <vector_types.h>

// tuple hashing pulled from
// https://www.variadic.xyz/2018/01/15/hashing-stdpair-and-stdtuple/
template <class T>
inline void hash_combine(std::size_t &seed, T const &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {
template <class... TupleArgs>
struct hash<std::tuple<TupleArgs...>> {
private:
  //  this is a termination condition
  //  N == sizeof...(TupleTypes)
  //
  template <size_t Idx, typename... TupleTypes>
  inline typename std::enable_if<Idx == sizeof...(TupleTypes), void>::type
      hash_combine_tup(size_t &seed,
                       std::tuple<TupleTypes...> const &tup) const {}

  //  this is the computation function
  //  continues till condition N < sizeof...(TupleTypes) holds
  //
  template <size_t Idx, typename... TupleTypes>
      inline typename std::enable_if < Idx<sizeof...(TupleTypes), void>::type
      hash_combine_tup(size_t &seed,
                       std::tuple<TupleTypes...> const &tup) const {
    hash_combine(seed, std::get<Idx>(tup));

    //  on to next element
    hash_combine_tup<Idx + 1>(seed, tup);
  }

public:
  size_t operator()(std::tuple<TupleArgs...> const &tupleValue) const {
    size_t seed = 0;
    //  begin with the first iteration
    hash_combine_tup<0>(seed, tupleValue);
    return seed;
  }
};

template <typename L, typename R>
struct hash<pair<L, R>> {
  size_t operator()(pair<L, R> const &p) const {
    size_t seed = 283746;

    hash_combine(seed, p.first);
    hash_combine(seed, p.second);

    return seed;
  }
};

template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(std::vector<T> const &v) const {
    size_t seed = 283746;

    hash_combine(seed, v.size());
    for (auto const &elem : v) {
      hash_combine(seed, elem);
    }

    return seed;
  }
};

template <>
struct hash<dim3> {
  size_t operator()(dim3 const &d) const {
    return hash<std::tuple<uint, uint, uint>>{}(std::make_tuple(d.x, d.y, d.z));
  }
};

template <>
struct hash<int3> {
  size_t operator()(int3 const &d) const {
    return hash<std::tuple<int, int, int>>{}(std::make_tuple(d.x, d.y, d.z));
  }
};

} // namespace std
