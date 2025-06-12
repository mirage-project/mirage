#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace nki_transpiler {

using mirage::transpiler::fmt;

std::string mirage_dtype_to_nki(type::DataType);

template <typename T>
std::string get_tensor_shape_tuple(T const &tensor) {
  std::string result;
  for (int i = 0; i < tensor.num_dims; i++) {
    result += std::to_string(tensor.dim[i]) + ",";
  }
  return fmt("($)", result);
}

enum class NKITensorInitializer {
  NONE,
  ZERO,
};

template <typename T>
std::string allocate_nki_tensor(
    T const &tensor,
    NKITensorInitializer initializer,
    std::string const &buffer
)  {
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
  std::string tensor_shape = get_tensor_shape_tuple(tensor);
  std::string nki_dtype = mirage_dtype_to_nki(tensor.data_type);
  return fmt("$($, dtype=$, buffer=$)",
             api,
             tensor_shape,
             nki_dtype,
             buffer);
}

}
}