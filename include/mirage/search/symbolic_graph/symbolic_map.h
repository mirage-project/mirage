#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/json_utils.h"
#include <cassert>
#include <vector>

namespace mirage {
namespace search {

class SymbolicMap {
  int num_grid_dims_; // k
  int num_data_dims_; // n
  bool has_forloop_;  // if true, last row (index k) is forloop
  // Flat num_rows*n matrix, row-major: mat_[row * num_data_dims_ + d]
  // num_rows = num_grid_dims_ + (has_forloop_ ? 1 : 0)
  // Entry is TensorDimVar (symbolic) or TensorDimConst(0/1) (concrete)
  std::vector<SymbolicTensorDim> mat_;

  int num_rows() const {
    return num_grid_dims_ + (has_forloop_ ? 1 : 0);
  }

public:
  // All-symbolic: each cell gets a fresh TensorDimVar
  SymbolicMap(int num_grid_dims,
              int num_data_dims,
              bool has_forloop,
              tensor_dim_var_index_t &index_counter)
      : num_grid_dims_(num_grid_dims), num_data_dims_(num_data_dims),
        has_forloop_(has_forloop), mat_(num_rows() * num_data_dims) {
    for (int i = 0; i < (int)mat_.size(); ++i) {
      mat_[i] = dim_expr_make_var(index_counter++, /*is_boolean=*/true);
    }
  }

  // Concrete output map (no forloop): legacy_map[p] = data_dim or -1
  SymbolicMap(int num_grid_dims,
              int num_data_dims,
              std::vector<int> const &legacy_map)
      : num_grid_dims_(num_grid_dims), num_data_dims_(num_data_dims),
        has_forloop_(false), mat_(num_grid_dims * num_data_dims) {
    assert((int)legacy_map.size() == num_grid_dims_);
    for (int p = 0; p < num_grid_dims_; ++p) {
      for (int d = 0; d < num_data_dims_; ++d) {
        mat_[p * num_data_dims_ + d] =
            dim_expr_make_const(legacy_map[p] == d ? 1 : 0);
      }
    }
  }

  // Concrete input map (with forloop): legacy_map[p] = data_dim or -1,
  // forloop_dim = data dim for forloop or -1
  SymbolicMap(int num_grid_dims,
              int num_data_dims,
              std::vector<int> const &legacy_map,
              int forloop_dim)
      : num_grid_dims_(num_grid_dims), num_data_dims_(num_data_dims),
        has_forloop_(true), mat_(num_rows() * num_data_dims) {
    assert((int)legacy_map.size() == num_grid_dims_);
    // Grid rows
    for (int p = 0; p < num_grid_dims_; ++p) {
      for (int d = 0; d < num_data_dims_; ++d) {
        mat_[p * num_data_dims_ + d] =
            dim_expr_make_const(legacy_map[p] == d ? 1 : 0);
      }
    }
    // Forloop row (last)
    int fr = forloop_row();
    for (int d = 0; d < num_data_dims_; ++d) {
      mat_[fr * num_data_dims_ + d] =
          dim_expr_make_const(forloop_dim == d ? 1 : 0);
    }
  }

  // Mixed symbolic/concrete input map for ablation study.
  // Grid rows: symbolic if sym_grid_rows, else concrete from legacy_map.
  // Forloop row: symbolic if sym_forloop_row, else concrete from forloop_dim.
  SymbolicMap(int num_grid_dims,
              int num_data_dims,
              bool sym_grid_rows,
              bool sym_forloop_row,
              std::vector<int> const &legacy_map,
              int forloop_dim,
              tensor_dim_var_index_t &index_counter)
      : num_grid_dims_(num_grid_dims), num_data_dims_(num_data_dims),
        has_forloop_(true), mat_(num_rows() * num_data_dims) {
    // Grid rows
    for (int p = 0; p < num_grid_dims_; ++p) {
      for (int d = 0; d < num_data_dims_; ++d) {
        if (sym_grid_rows) {
          mat_[p * num_data_dims_ + d] =
              dim_expr_make_var(index_counter++, /*is_boolean=*/true);
        } else {
          mat_[p * num_data_dims_ + d] =
              dim_expr_make_const(legacy_map[p] == d ? 1 : 0);
        }
      }
    }
    // Forloop row
    int fr = forloop_row();
    for (int d = 0; d < num_data_dims_; ++d) {
      if (sym_forloop_row) {
        mat_[fr * num_data_dims_ + d] =
            dim_expr_make_var(index_counter++, /*is_boolean=*/true);
      } else {
        mat_[fr * num_data_dims_ + d] =
            dim_expr_make_const(forloop_dim == d ? 1 : 0);
      }
    }
  }

  int num_grid_dims() const {
    return num_grid_dims_;
  }
  int num_data_dims() const {
    return num_data_dims_;
  }
  bool has_forloop() const {
    return has_forloop_;
  }
  int forloop_row() const {
    assert(has_forloop_);
    return num_grid_dims_;
  }

  SymbolicTensorDim at(int row, int data_dim) const {
    assert(row >= 0 && row < num_rows());
    assert(data_dim >= 0 && data_dim < num_data_dims_);
    return mat_[row * num_data_dims_ + data_dim];
  }

  void set(int row, int data_dim, SymbolicTensorDim val) {
    assert(row >= 0 && row < num_rows());
    assert(data_dim >= 0 && data_dim < num_data_dims_);
    mat_[row * num_data_dims_ + data_dim] = val;
  }

  // Evaluate: which data dim does this row map to? (-1 if none)
  int data_dim_for(int row, DimVarAssignment const &a) const {
    for (int d = 0; d < num_data_dims_; ++d) {
      if (at(row, d)->get_value(a) == 1) {
        return d;
      }
    }
    return -1;
  }

  // Concrete shorthand (entries are TensorDimConst)
  int data_dim_for(int row) const {
    DimVarAssignment empty;
    return data_dim_for(row, empty);
  }

  // Grid rows → legacy map (size k)
  std::vector<int> to_legacy_map(DimVarAssignment const &a) const {
    std::vector<int> result(num_grid_dims_);
    for (int p = 0; p < num_grid_dims_; ++p) {
      result[p] = data_dim_for(p, a);
    }
    return result;
  }

  std::vector<int> to_legacy_map() const {
    DimVarAssignment empty;
    return to_legacy_map(empty);
  }

  // Forloop row → legacy forloop_dim (-1 if no forloop or none mapped)
  int to_legacy_forloop_dim(DimVarAssignment const &a) const {
    if (!has_forloop_) {
      return -1;
    }
    return data_dim_for(forloop_row(), a);
  }

  int to_legacy_forloop_dim() const {
    DimVarAssignment empty;
    return to_legacy_forloop_dim(empty);
  }

  bool is_concrete() const {
    for (auto const &entry : mat_) {
      if (!entry->is_const()) {
        return false;
      }
    }
    return true;
  }

  bool operator==(SymbolicMap const &other) const {
    return num_grid_dims_ == other.num_grid_dims_ &&
           num_data_dims_ == other.num_data_dims_ &&
           has_forloop_ == other.has_forloop_ && mat_ == other.mat_;
  }

  operator json() const {
    std::vector<json> mat_json;
    for (auto const &entry : mat_) {
      mat_json.push_back(*entry);
    }
    return json{{"num_grid_dims", num_grid_dims_},
                {"num_data_dims", num_data_dims_},
                {"has_forloop", has_forloop_},
                {"mat", mat_json}};
  }
};

inline void from_json(json const &j, SymbolicMap &map) {
  int ng = j.at("num_grid_dims").get<int>();
  int nd = j.at("num_data_dims").get<int>();
  bool hf = j.at("has_forloop").get<bool>();
  int nr = ng + (hf ? 1 : 0);
  auto const &mat_json = j.at("mat");
  assert((int)mat_json.size() == nr * nd);
  // Build a dummy concrete map, then overwrite entries
  std::vector<int> dummy(ng, -1);
  if (hf) {
    map = SymbolicMap(ng, nd, dummy, -1);
  } else {
    map = SymbolicMap(ng, nd, dummy);
  }
  for (int i = 0; i < nr * nd; ++i) {
    SymbolicTensorDim dim;
    mirage::search::from_json(mat_json[i], dim);
    map.set(i / nd, i % nd, dim);
  }
}

} // namespace search
} // namespace mirage
