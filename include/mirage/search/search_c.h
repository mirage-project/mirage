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

int cython_search(mirage::kernel::Graph const *input_graph,
                  int max_num_graphs,
                  mirage::kernel::Graph **new_graphs,
                  std::vector<MInt3> imap_to_explore,
                  std::vector<MInt3> omap_to_explore,
                  std::vector<MDim3> grid_dim_to_explore,
                  std::vector<MDim3> block_dim_to_explore,
                  std::vector<int> fmap_to_explore,
                  std::vector<int> frange_to_explore,
                  char const *filename,
                  bool verbose,
                  char const *default_config,
                  bool is_formal_verified);

void cython_to_json(mirage::kernel::Graph const *input_graph,
                    char const *filename);
mirage::kernel::Graph *cython_from_json(char const *filename);
} // namespace search_c
} // namespace mirage
