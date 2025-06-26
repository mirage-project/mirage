#include "mirage/nki_transpiler/helper_function.h"

#include <cassert>

namespace mirage {
namespace nki_transpiler {

std::string
    HelperFunction::get_invocation(std::vector<std::string> const &args) const {
  assert(args.size() == params.size());
  std::string invocation = name + "(";
  for (size_t i = 0; i < args.size(); ++i) {
    invocation += params[i] + " = " + args[i];
    if (i < args.size() - 1) {
      invocation += ", ";
    }
  }
  invocation += ")";
  return invocation;
}

std::string HelperFunction::get_code() const {
  std::string code = "def " + name + "(";
  for (size_t i = 0; i < params.size(); ++i) {
    code += params[i];
    if (i < params.size() - 1) {
      code += ", ";
    }
  }
  code += "):\n";
  code += body;
  return code;
}

HelperFunction tiled_matmul_function() {
  HelperFunction tiled_matmul;
  tiled_matmul.deps = {};
  tiled_matmul.name = "tiled_matmul";
  tiled_matmul.params = {
      "_lhs", "_rhs", "lhs_transposed", "rhs_transposed", "dtype"};
  tiled_matmul.body = "\
  lhsT = _lhs if lhs_transposed else nl.transpose(_lhs)\n\
  rhs = nl.transpose(_rhs) if rhs_transposed else _rhs\n\
  assert lhsT.shape[0] == rhs.shape[0], \"Matrix dimensions do not match for multiplication\"\n\
  M, N, K = lhsT.shape[1], rhs.shape[1], lhsT.shape[0]\n\
  result = nl.zeros((M, N), dtype=dtype, buffer=nl.sbuf)\n\
  TILE_M, TILE_N, TILE_K = min(M, 128), min(N, 512), min(K, 128)\n\
  for tile_id_M in nl.affine_range(M // TILE_M):\n\
    for tile_id_N in nl.affine_range(N // TILE_N):\n\
      result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)\n\
      for tile_id_K in nl.affine_range(K // TILE_K):\n\
        lhsT_tile = lhsT[tile_id_K * TILE_K + nl.arange(TILE_K)[:, None], tile_id_M * TILE_M + nl.arange(TILE_M)[None, :]]\n\
        rhs_tile = rhs[tile_id_K * TILE_K + nl.arange(TILE_K)[:, None], tile_id_N * TILE_N + nl.arange(TILE_N)[None, :]]\n\
        result_tile += nisa.nc_matmul(lhsT_tile, rhs_tile)\n\
      result[tile_id_M * TILE_M + nl.arange(TILE_M)[:, None], tile_id_N * TILE_N + nl.arange(TILE_N)[None, :]] += result_tile\n\
  return result\n\
";
  return tiled_matmul;
};

} // namespace nki_transpiler
} // namespace mirage
