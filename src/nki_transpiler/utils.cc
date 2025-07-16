#include "mirage/nki_transpiler/utils.h"

namespace mirage {
namespace nki_transpiler {

using mirage::transpiler::fmt;

std::string mirage_dtype_to_nki(type::DataType dt) {
  std::string nki_type;
  switch (dt) {
    case type::DataType::DT_INT4:
      assert(false && "4-bit integer not supported in nki");
      break;
    case type::DataType::DT_INT8:
      nki_type = "nl.int8";
      break;
    case type::DataType::DT_UINT16:
      nki_type = "nl.uint16";
      break;
    case type::DataType::DT_FLOAT8:
      // todo: when we should use e5m2?
      nki_type = "nl.float8_e4m3";
      break;
    case type::DataType::DT_FLOAT16:
      nki_type = "nl.float16";
      break;
    case type::DataType::DT_BFLOAT16:
      nki_type = "nl.bfloat16";
      break;
    case type::DataType::DT_FLOAT32:
      nki_type = "nl.float32";
      break;
    default:
      assert(false && "unsupported nki type in mirage");
      break;
  }
  return nki_type;
}

std::string get_python_literal(bool value) {
  return value ? "True" : "False";
}

std::string get_tensor_variable_name(kernel::DTensor const &tensor) {
  return fmt("dtensor$", tensor.guid);
}

std::string get_tensor_variable_name(threadblock::STensor const &tensor) {
  return fmt("stensor$", tensor.guid);
}

} // namespace nki_transpiler
} // namespace mirage