#pragma once
#include "mirage/kernel/graph.h"
#include <vector_types.h>

namespace mirage {
namespace search_c {

struct MInt3 {
  int x, y, z;
};

struct MDim3 {
  unsigned int x, y, z;
};

int cython_optimize(mirage::kernel::Graph const *input_graph,
                    int max_num_graphs,
                    mirage::kernel::Graph** new_graphs,
                    std::vector<MInt3> imap_to_explore,
                    std::vector<MInt3> omap_to_explore,
                    std::vector<MDim3> grid_dim_to_explore,
                    std::vector<MDim3> block_dim_to_explore,
                    std::vector<int> fmap_to_explore,
                    std::vector<int> frange_to_explore,
                    const char *check_point_file_path,
                    const char *default_config);
} // namespace search_c
} // namespace mirage
