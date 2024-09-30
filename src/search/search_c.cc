#include "mirage/search/search_c.h"
#include "mirage/kernel/customized.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/search.h"
#include "mirage/utils/containers.h"

#include <fstream>
#include <iostream>

namespace mirage {
namespace search_c {

int cython_search(mirage::kernel::Graph const *input_graph,
                  int max_num_graphs,
                  mirage::kernel::Graph **new_graphs,
                  std::vector<MInt3> imap_to_explore,
                  std::vector<MInt3> omap_to_explore,
                  std::vector<MDim3> grid_dim_to_explore,
                  std::vector<MDim3> block_dim_to_explore,
                  std::vector<int> fmap_to_explore,
                  std::vector<int> frange_to_explore,
                  bool verbose,
                  char const *default_config) {
  // NOTE(@wmdi): Checkpointing is disabled for now
  // Load from a checkpoint
  // if (check_point_file_path != nullptr) {
  //   search::KernelGraphGenerator gen(check_point_file_path);
  //   gen.config.print_config();
  //   // Only continue the search if we haven't discovered any graphs
  //   if (gen.generated_graphs.size() == 0) {
  //     gen.generate_kernel_graphs();
  //   }
  //   int num = 0;
  //   for (json const &j : gen.generated_graphs) {
  //     assert(num < max_num_graphs);
  //     new_graphs[num] = new kernel::Graph();
  //     from_json(j, *new_graphs[num]);
  //     num++;
  //   }
  //   return num;
  // } else
  {
    search::GeneratorConfig config =
        search::GeneratorConfig::get_default_config();
    if (default_config != nullptr) {
      if (!strcmp(default_config, "attention")) {
        config.enable_attention_specific_optimization();
      } else if (!strcmp(default_config, "lora")) {
        config.enable_concat_matmul_transformation();
      } else if (!strcmp(default_config, "mlp")) {
      }
    }
    // Customized imaps
    if (imap_to_explore.size() > 0) {
      config.imap_to_explore.clear();
      for (auto const &imap : imap_to_explore) {
        config.imap_to_explore.push_back({imap.x, imap.y, imap.z});
      }
    }
    // Customized omaps
    if (omap_to_explore.size() > 0) {
      config.omap_to_explore.clear();
      for (auto const &omap : omap_to_explore) {
        config.omap_to_explore.push_back({omap.x, omap.y, omap.z});
      }
    }
    // Customized griddims
    if (grid_dim_to_explore.size() > 0) {
      config.grid_dim_to_explore.clear();
      for (auto const &griddim : grid_dim_to_explore) {
        config.grid_dim_to_explore.push_back({griddim.x, griddim.y, griddim.z});
      }
    }
    // Customized blockdims
    if (block_dim_to_explore.size() > 0) {
      config.block_dim_to_explore.clear();
      for (auto const &blockdim : block_dim_to_explore) {
        config.block_dim_to_explore.push_back(
            {blockdim.x, blockdim.y, blockdim.z});
      }
    }
    // Customized fmap
    if (fmap_to_explore.size() > 0) {
      config.fmap_to_explore.clear();
      for (auto const &fmap : fmap_to_explore) {
        config.fmap_to_explore.push_back(fmap);
      }
    }
    // Customized frange
    if (frange_to_explore.size() > 0) {
      config.frange_to_explore.clear();
      for (auto const &frange : frange_to_explore) {
        config.frange_to_explore.push_back(frange);
      }
    }
    search::KernelGraphGenerator gen(
        *input_graph, config, "mirage_search_checkpoint.json", verbose);
    gen.config.show();
    gen.generate_kernel_graphs();
    int num = 0;
    for (json const &j : gen.generated_graphs) {
      assert(num < max_num_graphs);
      new_graphs[num] = new kernel::Graph();
      from_json(j, *new_graphs[num]);
      num++;
    }
    return num;
  }
}

} // namespace search_c
} // namespace mirage
