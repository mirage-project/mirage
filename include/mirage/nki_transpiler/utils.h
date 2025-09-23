#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace nki_transpiler {

using mirage::transpiler::ceil_div;
using mirage::transpiler::fmt;
using mirage::transpiler::my_to_string;

constexpr int NKI_PMAX = 128;
constexpr int NKI_FMAX_STATIONARY = 128;
constexpr int NKI_FMAX_MOVING = 512;

std::string mirage_dtype_to_nki(type::DataType);

template <typename T>
int get_partition_dimension_degree(T const &tensor, int partition_dim) {
  assert(partition_dim >= 0 && partition_dim < tensor.num_dims);
  return ceil_div(tensor.dim[partition_dim], NKI_PMAX);
}

template <typename T>
int get_partition_dimension_size(T const &tensor, int partition_dim) {
  assert(partition_dim >= 0 && partition_dim < tensor.num_dims);
  return std::min(tensor.dim[partition_dim], NKI_PMAX);
}

template <typename T>
std::string get_tensor_shape_tuple(T const &tensor, int partition_dim) {
  std::string result;
  if (partition_dim != -1) {
    assert(tensor.dim[partition_dim] <= NKI_PMAX ||
           tensor.dim[partition_dim] % NKI_PMAX == 0);
    result += fmt("$,$,",
                  get_partition_dimension_size(tensor, partition_dim),
                  get_partition_dimension_degree(tensor, partition_dim));
  }
  for (int i = 0; i < tensor.num_dims; i++) {
    if (i != partition_dim) {
      result += std::to_string(tensor.dim[i]) + ",";
    }
  }
  return fmt("($)", result);
}

std::string get_python_literal(bool value);

std::string get_tensor_variable_name(kernel::DTensor const &tensor);
std::string get_tensor_variable_name(threadblock::STensor const &tensor);

template <typename T, typename U>
std::string get_dim_start_end(T const &start, U const &end) {
  return fmt("$:$", start, end);
}

template <typename T, typename U>
std::string get_dim_start_length(T const &start, U const &length) {
  return fmt("$:($)+($)", start, start, length);
}

enum class NKITensorInitializer {
  NONE,
  ZERO,
};

template <typename T>
std::string allocate_nki_tensor(T const &tensor,
                                int partition_dim,
                                NKITensorInitializer initializer,
                                std::string const &buffer) {
  std::string api = "nl.";
  switch (initializer) {
    case NKITensorInitializer::NONE:
      api += "ndarray";
      break;
    case NKITensorInitializer::ZERO:
      api += "zeros";
      break;
    default:
      assert(false && "Unknown NKI tensor initializer");
      break;
  }
  std::string tensor_shape = get_tensor_shape_tuple(tensor, partition_dim);
  std::string nki_dtype = "dtype";
  return fmt("$($, dtype=$, buffer=$)", api, tensor_shape, nki_dtype, buffer);
}

template <typename T>
std::string str_join(std::vector<T> const &vec,
                     std::string const &sep,
                     std::string empty = "",
                     std::string const &end = "") {
  if (vec.empty()) {
    return empty + end;
  }
  std::string result;
  for (size_t i = 0; i < vec.size(); ++i) {
    result += my_to_string(vec[i]);
    if (i < vec.size() - 1) {
      result += sep;
    }
  }
  result += end;
  return result;
}

} // namespace nki_transpiler
} // namespace mirage