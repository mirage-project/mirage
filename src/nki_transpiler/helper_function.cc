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

HelperFunction tiled_transpose_function() {
  HelperFunction tiled_transpose;
  tiled_transpose.deps = {""};
  tiled_transpose.name = "tiled_transpose";
  tiled_transpose.params = {"tensor"};
  tiled_transpose.body = "\
  assert len(tensor.shape) == 3\n\
  ftile_size, num_ftile, psize = tensor.shape[0], tensor.shape[1], tensor.shape[2]\n\
  num_ptile, ptile_size = psize // 128, 128\n\
  result = nl.ndarray((ptile_size, num_ptile, num_ftile * ftile_size), dtype=tensor.dtype, buffer=nl.sbuf)\n\
  for tile_id in nl.affine_range(num_ptile):\n\
    for f_tile_id in nl.affine_range(num_ftile):\n\
      result[:, tile_id, f_tile_id * ftile_size:(f_tile_id + 1) * ftile_size] = \\\n\
          nl.transpose(tensor[:, f_tile_id, tile_id * ptile_size:(tile_id + 1) * ptile_size])\n\
  return result\n";
  return tiled_transpose;
}

HelperFunction tiled_matmul_function() {
  HelperFunction tiled_matmul;
  tiled_matmul.deps = {"tiled_transpose", "tiled_matmul_accum"};
  tiled_matmul.name = "tiled_matmul";
  tiled_matmul.params = {
      "_lhs", "_rhs", "lhs_transposed", "rhs_transposed", "dtype"};
  tiled_matmul.body = "\
  rhs = tiled_transpose(_rhs) if rhs_transposed else _rhs\n\
  if lhs_transposed:\n\
    lhsT = _lhs\n\
    assert lhsT.shape[0] == rhs.shape[0] and lhsT.shape[1] == rhs.shape[1], \"Matrix dimensions do not match for multiplication\"\n\
    num_ktiles, M, N = lhsT.shape[1], lhsT.shape[-1], rhs.shape[-1]\n\
    TILE_M, TILE_N, TILE_K = min(M, 128), min(N, 512), 128\n\
    num_mtiles = M // TILE_M\n\
    result = nl.zeros((TILE_M, num_mtiles, N), dtype=dtype, buffer=nl.sbuf)\n\
  else:\n\
    lhs = _lhs\n\
    TILE_M, num_mtiles, K = lhs.shape[0], lhs.shape[1], lhs.shape[2]\n\
    TILE_K, num_ktiles, N = rhs.shape[0], rhs.shape[1], rhs.shape[2]\n\
    assert TILE_K * num_ktiles == K, \"Matrix dimensions do not match for multiplication\"\n\
    result = nl.zeros((TILE_M, num_mtiles, N), dtype=dtype, buffer=nl.sbuf)\n\
  tiled_matmul_accum(_lhs, rhs, result, lhs_transposed)\n\
  return result\n\
";
  return tiled_matmul;
};

HelperFunction tiled_matmul_accum_function() {
  HelperFunction tiled_matmul_accum;
  tiled_matmul_accum.deps = {};
  tiled_matmul_accum.name = "tiled_matmul_accum";
  tiled_matmul_accum.params = {"_lhs", "rhs", "result", "lhs_transposed"};
  tiled_matmul_accum.body = "\
  if lhs_transposed:\n\
    lhsT = _lhs\n\
    assert lhsT.shape[0] == rhs.shape[0] and lhsT.shape[1] == rhs.shape[1], \"Matrix dimensions do not match for multiplication\"\n\
    num_ktiles, M, N = lhsT.shape[1], lhsT.shape[-1], rhs.shape[-1]\n\
    TILE_M, TILE_N, TILE_K = min(M, 128), min(N, 512), 128\n\
    num_mtiles = M // TILE_M\n\
    # result = nl.zeros((TILE_M, num_mtiles, N), dtype=dtype, buffer=nl.sbuf)\n\
    assert lhsT.shape[0] == TILE_K\n\
    for tile_id_M in nl.affine_range(M // TILE_M):\n\
      for tile_id_N in nl.affine_range(N // TILE_N):\n\
        result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)\n\
        for tile_id_K in nl.affine_range(num_ktiles):\n\
          lhsT_tile = lhsT[:, tile_id_K, tile_id_M * TILE_M : tile_id_M * TILE_M + TILE_M]\n\
          rhs_tile = rhs[:, tile_id_K, tile_id_N * TILE_N : tile_id_N * TILE_N + TILE_N]\n\
          result_tile += nisa.nc_matmul(lhsT_tile, rhs_tile)\n\
        result[:, tile_id_M, tile_id_N * TILE_N : tile_id_N * TILE_N + TILE_N] += result_tile\n\
  else:\n\
    lhs = _lhs\n\
    TILE_M, num_mtiles, K = lhs.shape[0], lhs.shape[1], lhs.shape[2]\n\
    TILE_K, num_ktiles, N = rhs.shape[0], rhs.shape[1], rhs.shape[2]\n\
    assert TILE_K * num_ktiles == K, \"Matrix dimensions do not match for multiplication\"\n\
    TILE_N = min(N, 512)\n\
    num_ntiles = N // TILE_N\n\
    for tile_id_M in nl.affine_range(num_mtiles):\n\
      for tile_id_N in nl.affine_range(num_ntiles):\n\
        result_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)\n\
        for tile_id_K in nl.affine_range(num_ktiles):\n\
          lhs_tile = lhs[:, tile_id_M, tile_id_K * TILE_K:(tile_id_K + 1) * TILE_K]\n\
          lhsT_tile = nisa.nc_transpose(lhs_tile)\n\
          rhs_tile = rhs[:, tile_id_K, tile_id_N * TILE_N:(tile_id_N + 1) * TILE_N]\n\
          result_tile += nisa.nc_matmul(lhsT_tile, rhs_tile)\n\
        result[:, tile_id_M, tile_id_N * TILE_N : tile_id_N * TILE_N + TILE_N] += result_tile\n\
";
  return tiled_matmul_accum;
}

} // namespace nki_transpiler
} // namespace mirage
