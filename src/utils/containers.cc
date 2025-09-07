#include "mirage/utils/containers.h"

bool operator==(dim3 const &lhs, dim3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

bool operator==(int3 const &lhs, int3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

int get_block_size_when_has_tail(int size, int grid) {
  // This vector should be align with grid_for_linear_layer in qwen3 demo.py
  std::vector<int> block_size_list{32, 48, 64};

  for (auto const block_size : block_size_list) {
    if ((block_size * (grid - 1) < size) && (block_size * grid > size)) {
      return block_size;
    }
  }

  printf("Can't find suitable block size for when has tail for size: %d, grid: "
         "%d.\n",
         size,
         grid);
  abort();
}