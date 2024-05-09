#include "mirage/utils/containers.h"

bool operator==(dim3 const &lhs, dim3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool operator==(int3 const &lhs, int3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}
