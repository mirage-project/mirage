#include "mirage/utils/containers.h"

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