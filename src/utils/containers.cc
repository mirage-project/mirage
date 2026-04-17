#include "mirage/utils/containers.h"

#include <cassert>

bool operator==(dim3 const &lhs, dim3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool operator==(int3 const &lhs, int3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

std::vector<unsigned int> to_vector(dim3 const &d) {
  return {d.x, d.y, d.z};
}

std::vector<int> to_vector(int3 const &d) {
  return {d.x, d.y, d.z};
}

int3 vec_to_int3(std::vector<int> const &v) {
  assert(v.size() <= 3);
  std::vector<int> v_padded = pad_vector(v, 3, -1);
  return {v_padded[0], v_padded[1], v_padded[2]};
}

dim3 vec_to_dim3(std::vector<unsigned int> const &v) {
  assert(v.size() <= 3);
  std::vector<unsigned int> v_padded = pad_vector(v, 3, 1u);
  return {v_padded[0], v_padded[1], v_padded[2]};
}
