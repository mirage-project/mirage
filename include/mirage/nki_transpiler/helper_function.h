#pragma once

#include <string>
#include <vector>

namespace mirage {
namespace nki_transpiler {

struct HelperFunction {
  std::vector<std::string> deps;
  std::string name;
  std::vector<std::string> params;
  std::string body;

  std::string get_code() const;
  std::string get_invocation(std::vector<std::string> const &args) const;
};

HelperFunction tiled_transpose_function();
HelperFunction tiled_matmul_function();
HelperFunction tiled_matmul_accum_function();

} // namespace nki_transpiler
} // namespace mirage