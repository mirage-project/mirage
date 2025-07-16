#include "mirage/search/dim_strategy.h"
#include "mirage/config.h"
#include "mirage/utils/containers.h"

namespace mirage {
namespace search {

DimStrategy::DimStrategy(GeneratorConfig const &config) : config(config) {}

std::vector<type::KNOperatorType> DimStrategy::get_knop_cand() {
  std::vector<type::KNOperatorType> cands = config.knop_to_explore;
  if (config.randomized_branches) {
    std::random_shuffle(cands.begin(), cands.end());
  }
  return cands;
}

std::vector<type::TBOperatorType> DimStrategy::get_tbop_cand() {
  std::vector<type::TBOperatorType> cands = config.tbop_to_explore;
  if (config._enable_concat_matmul_transformation) {
    cands.push_back(type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP);
    cands = deduplicate(cands);
  }
  if (config.randomized_branches) {
    std::random_shuffle(cands.begin(), cands.end());
  }
  return cands;
}

std::vector<dim3>
    DimStrategy::get_grid_dim_cand(std::vector<DTensor> const &tensors) {

  auto generate_1d_grids = [&](std::vector<int> const &dims) {
    std::vector<dim3> cands;
    for (size_t x = 8; x <= 256; x *= 2) {
      for (int dim : dims) {
        if (dim % x == 0) {
          cands.push_back({dim / x, 1, 1});
        }
      }
    }
    return cands;
  };

  auto generate_2d_grids = [&](int x, std::vector<int> const &dims) {
    std::vector<dim3> cands;
    for (size_t y : {8, 64, 128, 256}) {
      for (int dim : dims) {
        if (dim % y == 0) {
          cands.push_back({x, dim / y, 1});
        }
      }
    }
    return cands;
  };

  auto is_all_n_dim = [&](int n) {
    for (DTensor const &tensor : tensors) {
      if (tensor.num_dims != n) {
        return false;
      }
    }
    return true;
  };

  auto get_batch = [&] {
    if (is_all_n_dim(2)) {
      return tensors[0].dim[0];
    }
    if (is_all_n_dim(3)) {
      return tensors[0].dim[0];
    }
    return -1;
  };

  auto get_dims = [&] {
    std::unordered_set<int> dims;
    for (DTensor const &tensor : tensors) {
      for (int i = 0; i < tensor.num_dims; ++i) {
        dims.insert(tensor.dim[i]);
      }
    }
    return std::vector<int>(dims.begin(), dims.end());
  };

  std::vector<dim3> cands = config.grid_dim_to_explore;
  int batch = get_batch();

  cands = vector_concat(cands, generate_1d_grids(get_dims()));
  if (config._enable_attention_specific_optimization) {
    if (batch != -1) {
      cands = vector_concat(cands, generate_2d_grids(batch, get_dims()));
    }
    if (tensors.size() > 2) {
      cands.push_back({batch, 16, 4});
    }
  }
  cands = filter(cands, [](dim3 const &dim) {
    int num_threadblocks = dim.x * dim.y * dim.z;
    return 32 <= num_threadblocks &&
           num_threadblocks <= config::MAX_NUM_THREADBLOCKS_PER_KERNEL;
  });

  if (batch != -1 && batch <= config::MAX_NUM_THREADBLOCKS_PER_KERNEL) {
    cands.push_back({batch, 1, 1});
  }

  cands = deduplicate(cands);

  if (config.randomized_branches) {
    std::random_shuffle(cands.begin(), cands.end());
  }
  return cands;
}

std::vector<dim3>
    DimStrategy::get_block_dim_cand(std::vector<DTensor> const &tensors,
                                    dim3 grid_dim) {
  std::vector<dim3> cands = config.block_dim_to_explore;
  cands.push_back({128, 1, 1});
  if (config.randomized_branches) {
    std::random_shuffle(cands.begin(), cands.end());
  }
  return cands;
}

bool is_all_replicate_x(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.x != -1) {
      return false;
    }
  }
  return true;
}

bool is_all_replicate_y(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.y != -1) {
      return false;
    }
  }
  return true;
}

bool is_all_replicate_z(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.z != -1) {
      return false;
    }
  }
  return true;
}

void generate_input_map_cand(std::vector<DTensor> const &tensors,
                             dim3 grid_dim,
                             std::vector<int3> imap_to_explore,
                             std::vector<int3> cur,
                             std::vector<std::vector<int3>> &results) {
  if (cur.size() == tensors.size()) {
    if ((is_all_replicate_x(cur) && grid_dim.x > 1) ||
        (is_all_replicate_y(cur) && grid_dim.y > 1) ||
        (is_all_replicate_z(cur) && grid_dim.z > 1)) {
      return;
    }
    results.push_back(cur);
    return;
  }
  DTensor const &tensor = tensors[cur.size()];
  for (int3 input_map : imap_to_explore) {
    if (tensor.num_dims <= input_map.x || tensor.num_dims <= input_map.y ||
        tensor.num_dims <= input_map.z) {
      continue;
    }
    if ((grid_dim.x == 1 && input_map.x != -1) ||
        (grid_dim.y == 1 && input_map.y != -1) ||
        (grid_dim.z == 1 && input_map.z != -1)) {
      continue;
    }
    if ((input_map.x != -1 && tensor.dim[input_map.x] % grid_dim.x != 0) ||
        (input_map.y != -1 && tensor.dim[input_map.y] % grid_dim.y != 0) ||
        (input_map.z != -1 && tensor.dim[input_map.z] % grid_dim.z != 0)) {
      continue;
    }
    cur.push_back(input_map);
    generate_input_map_cand(tensors, grid_dim, imap_to_explore, cur, results);
    cur.pop_back();
  }
}

bool is_valid_input_map(std::vector<DTensor> const &tensors,
                        dim3 grid_dim,
                        std::vector<int3> const &input_maps) {
  if (tensors.size() != input_maps.size()) {
    return false;
  }
  for (size_t i = 0; i < tensors.size(); ++i) {
    DTensor const &tensor = tensors[i];
    int3 input_map = input_maps[i];
    if (tensor.num_dims <= input_map.x || tensor.num_dims <= input_map.y ||
        tensor.num_dims <= input_map.z) {
      return false;
    }
    if ((input_map.x != -1 && tensor.dim[input_map.x] % grid_dim.x != 0) ||
        (input_map.y != -1 && tensor.dim[input_map.y] % grid_dim.y != 0) ||
        (input_map.z != -1 && tensor.dim[input_map.z] % grid_dim.z != 0)) {
      return false;
    }
  }
  return true;
}

std::vector<std::vector<int3>>
    DimStrategy::get_input_map_cand(std::vector<DTensor> const &tensors,
                                    dim3 grid_dim) {
  std::vector<std::vector<int3>> results;
  if (!config.imap_comb_to_explore.empty()) {
    for (auto const &input_maps : config.imap_comb_to_explore) {
      if (is_valid_input_map(tensors, grid_dim, input_maps)) {
        results.push_back(input_maps);
      }
    }
  } else if (!config.imap_to_explore.empty()) {
    generate_input_map_cand(
        tensors, grid_dim, config.imap_to_explore, {}, results);
  } else {
    std::vector<int3> imap_to_explore = {
        {0, -1, 1},
        {0, 1, -1},
        {0, 2, -1},
    };
    if (!config._enable_attention_specific_optimization) {
      imap_to_explore.push_back({-1, -1, -1});
      imap_to_explore.push_back({1, -1, -1});
      imap_to_explore.push_back({0, -1, -1});
    }
    generate_input_map_cand(tensors, grid_dim, imap_to_explore, {}, results);
  }
  if (config.randomized_branches) {
    std::random_shuffle(results.begin(), results.end());
  }
  return results;
}

std::vector<int3> DimStrategy::get_output_map_cand(dim3 grid_dim) {
  std::vector<int3> results;
  std::vector<int3> omap_to_explore = config.omap_to_explore;
  omap_to_explore = vector_concat(omap_to_explore,
                                  {
                                      {0, 1, -1},
                                      {0, 2, 1},
                                      {0, 2, -1},
                                      {0, -1, -1},
                                      {-1, 2, 1},
                                      {-1, 1, -1},
                                      {-1, 2, -1},
                                      {-1, -1, -1},
                                  });
  if (!config._enable_attention_specific_optimization) {
    omap_to_explore.push_back({1, -1, -1});
  }
  omap_to_explore = deduplicate(omap_to_explore);
  for (int3 output_map : omap_to_explore) {
    if ((grid_dim.x == 1 && output_map.x != -1) ||
        (grid_dim.x > 1 && output_map.x == -1)) {
      continue;
    }
    if ((grid_dim.y == 1 && output_map.y != -1) ||
        (grid_dim.y > 1 && output_map.y == -1)) {
      continue;
    }
    if ((grid_dim.z == 1 && output_map.z != -1) ||
        (grid_dim.z > 1 && output_map.z == -1)) {
      continue;
    }
    results.push_back(output_map);
  }
  if (config.randomized_branches) {
    std::random_shuffle(results.begin(), results.end());
  }
  return results;
}

void generate_forloop_dim(std::vector<DTensor> const &input_tensors,
                          std::vector<int> fmap_to_explore,
                          std::vector<int> cur,
                          std::vector<std::vector<int>> &results) {
  if (cur.size() == input_tensors.size()) {
    results.push_back(cur);
    return;
  }

  DTensor const &tensor = input_tensors[cur.size()];
  for (int dim : fmap_to_explore) {
    if ((dim == -1) ||
        (dim != -1 && dim < tensor.num_dims && tensor.dim[dim] > 1)) {
      cur.push_back(dim);
      generate_forloop_dim(input_tensors, fmap_to_explore, cur, results);
      cur.pop_back();
    }
  }
}

std::vector<std::vector<int>> DimStrategy::get_forloop_dim_cand(
    std::vector<DTensor> const &input_tensors) {
  std::vector<std::vector<int>> results;
  std::vector<int> fmap_to_explore = {-1, 0, 1, 2};
  if (!config.fmap_to_explore.empty()) {
    fmap_to_explore = config.fmap_to_explore;
  }
  generate_forloop_dim(input_tensors, fmap_to_explore, {}, results);
  if (config.randomized_branches) {
    std::random_shuffle(results.begin(), results.end());
  }
  return results;
}

std::vector<int> DimStrategy::get_forloop_range_cand(
    std::vector<DTensor> const &input_tensors,
    std::vector<int3> const &input_map,
    dim3 grid_dim,
    dim3 block_dim,
    std::vector<int> const &forloop_dim) {
  bool no_use = true;
  for (int dim : forloop_dim) {
    if (dim >= 0) {
      no_use = false;
    }
  }
  if (no_use) {
    return {1};
  }

  std::vector<int> results;

  for (int x : config.frange_to_explore) {
    if (config._enable_attention_specific_optimization && x > 8) {
      continue;
    }
    bool feasible = true;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (forloop_dim[i] == -1) {
        continue;
      }
      int dim = input_tensors[i].dim[forloop_dim[i]];
      if (input_map[i].x == forloop_dim[i]) {
        assert(dim % grid_dim.x == 0);
        dim /= grid_dim.x;
      }
      if (input_map[i].y == forloop_dim[i]) {
        assert(dim % grid_dim.y == 0);
        dim /= grid_dim.y;
      }
      if (input_map[i].z == forloop_dim[i]) {
        assert(dim % grid_dim.z == 0);
        dim /= grid_dim.z;
      }
      if (dim % x != 0) {
        feasible = false;
        break;
      }
    }
    if (feasible) {
      results.push_back(x);
    }
  }
  if (config.randomized_branches) {
    std::random_shuffle(results.begin(), results.end());
  }
  return results;
}

std::vector<std::vector<int>> DimStrategy::get_unary_input(int num_tensors) {
  std::vector<std::vector<int>> result;
  for (int i = 0; i < num_tensors; ++i) {
    result.push_back({i});
  }
  if (config.randomized_branches) {
    std::random_shuffle(result.begin(), result.end());
  }
  return result;
}

std::vector<std::vector<int>> DimStrategy::get_binary_input(int num_tensors) {
  std::vector<std::vector<int>> result;
  for (int i = 0; i < num_tensors; ++i) {
    for (int j = 0; j < num_tensors; ++j) {
      result.push_back({i, j});
    }
  }
  if (config.randomized_branches) {
    std::random_shuffle(result.begin(), result.end());
  }
  return result;
}

void get_nary_input(int n,
                    int num_tensors,
                    std::vector<int> &cur,
                    std::vector<std::vector<int>> &result) {
  if ((int)cur.size() == n) {
    result.push_back(cur);
    return;
  }
  for (int i = 0; i < num_tensors; ++i) {
    if (!contains(cur, i)) {
      cur.push_back(i);
      get_nary_input(n, num_tensors, cur, result);
      cur.pop_back();
    }
  }
}

std::vector<std::vector<int>> DimStrategy::get_nary_input(int num_tensors,
                                                          int n) {
  std::vector<std::vector<int>> result;
  std::vector<int> cur;
  mirage::search::get_nary_input(n, num_tensors, cur, result);
  if (config.randomized_branches) {
    std::random_shuffle(result.begin(), result.end());
  }
  return result;
}

std::vector<std::vector<int>> DimStrategy::get_customized_input_cand_idx(
    std::vector<DTensor> const &all_input) {

  int num_inputs = all_input.size();

  if (config._enable_concat_matmul_transformation && all_input.size() == 4) {
    return {{0, 1, 2, 3}};
  }
  if (all_input.size() == 3) {
    return {{0, 1, 2}};
  } else {
    return {{num_inputs - 2, num_inputs - 1}};
  }
}

void generate_input_map_cand(std::vector<SymbolicDTensor> const &tensors,
                             std::vector<int3> imap_to_explore,
                             std::vector<int3> cur,
                             std::vector<std::vector<int3>> &results) {
  if (cur.size() == tensors.size()) {
    results.push_back(cur);
    return;
  }
  for (int3 input_map : imap_to_explore) {
    cur.push_back(input_map);
    generate_input_map_cand(tensors, imap_to_explore, cur, results);
    cur.pop_back();
  }
}

std::vector<std::vector<int3>> DimStrategy::get_input_map_cand(
    std::vector<SymbolicDTensor> const &tensors) {
  std::vector<std::vector<int3>> results;
  std::vector<int3> imap_to_explore = {
      {0, -1, 1},
      {0, 1, -1},
      {0, 2, -1},
      {-1, -1, -1},
      {1, -1, -1},
      {0, -1, -1},
  };
  generate_input_map_cand(tensors, imap_to_explore, {}, results);
  if (config.randomized_branches) {
    std::random_shuffle(results.begin(), results.end());
  }
  return results;
}

std::vector<int3>
    DimStrategy::get_output_map_cand(SymbolicTBGraph const &tb_graph) {
  std::vector<int3> results;
  std::vector<int3> omap_to_explore = config.omap_to_explore;
  omap_to_explore = vector_concat(omap_to_explore,
                                  {
                                      {0, 1, -1},
                                      {0, 2, 1},
                                      {0, 2, -1},
                                      {0, -1, -1},
                                  });
  if (!config._enable_attention_specific_optimization) {
    omap_to_explore.push_back({1, -1, -1});
  }
  omap_to_explore = deduplicate(omap_to_explore);
  if (config.randomized_branches) {
    std::random_shuffle(results.begin(), results.end());
  }
  return results;
}

void generate_forloop_dim(std::vector<SymbolicDTensor> const &input_tensors,
                          std::vector<int> fmap_to_explore,
                          std::vector<int> cur,
                          std::vector<std::vector<int>> &results) {
  if (cur.size() == input_tensors.size()) {
    results.push_back(cur);
    return;
  }

  for (int dim : fmap_to_explore) {
    cur.push_back(dim);
    generate_forloop_dim(input_tensors, fmap_to_explore, cur, results);
    cur.pop_back();
  }
}

std::vector<std::vector<int>> DimStrategy::get_forloop_dim_cand(
    std::vector<SymbolicDTensor> const &input_tensers) {
  std::vector<std::vector<int>> results;
  std::vector<int> fmap_to_explore = {-1, 0, 1, 2};
  if (!config.fmap_to_explore.empty()) {
    fmap_to_explore = config.fmap_to_explore;
  }
  generate_forloop_dim(input_tensers, fmap_to_explore, {}, results);
  if (config.randomized_branches) {
    std::random_shuffle(results.begin(), results.end());
  }
  return results;
}

std::vector<std::vector<int>> DimStrategy::get_customized_input_cand_idx(
    std::vector<SymbolicDTensor> const &all_inputs) {
  int num_inputs = all_inputs.size();

  if (all_inputs.size() == 3) {
    return {{0, 1, 2}};
  } else {
    return {{num_inputs - 2, num_inputs - 1}};
  }
}

} // namespace search
} // namespace mirage
