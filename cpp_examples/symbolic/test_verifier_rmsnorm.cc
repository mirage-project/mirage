// Test: does the formal verifier accept the NS-style rmsnorm_mlp topology?
//
// NS topology:  mul(accum_up, accum_gate) / (rms * rms)
// Sym topology: div(accum_up, rms) * div(accum_gate, rms)
//
// These are mathematically equivalent:
//   (up/rms) * (gate/rms) == (up * gate) / (rms * rms)
//
// We construct the NS-style topology as a SymbolicKNGraph and check
// whether the formal verifier accepts it.

#include "mirage/kernel/graph.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/verification/formal_verifier.h"

#include <cassert>
#include <iostream>

using namespace mirage;

int main() {
  int const n = 8, d = 4096;

  // --- Build reference graph ---
  kernel::Graph ref;
  kernel::DTensor X = ref.new_input(
      {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_up = ref.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W_gate = ref.new_input(
      {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor X_norm = ref.rms_norm(X, {d});
  kernel::DTensor U      = ref.matmul(X_norm, W_up);
  kernel::DTensor Gx     = ref.matmul(X_norm, W_gate);
  kernel::DTensor O      = ref.mul(U, Gx);
  ref.mark_output(O);

  std::cout << "Reference graph built: O = (rms_norm(X) @ W_up) * (rms_norm(X) @ W_gate)"
            << std::endl;

  // --- Create formal verifier from reference ---
  search::FormalVerifier verifier(ref);
  std::cout << "Formal verifier initialized." << std::endl;

  // ===========================================================================
  // Test 1: Existing sym topology  (div, div, mul)
  //   accum_up / rms -> up_norm
  //   accum_gate / rms -> gate_norm
  //   up_norm * gate_norm -> output
  // ===========================================================================
  {
    std::cout << "\n=== Test 1: Sym topology (div, div, mul) ===" << std::endl;

    // Build a concrete kernel graph with this topology
    kernel::Graph g;
    kernel::DTensor gX = g.new_input(
        {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor gW_up = g.new_input(
        {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor gW_gate = g.new_input(
        {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

    dim3 grid_dim  = {64, 1, 1};
    dim3 block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim, block_dim, 64, 64);

    threadblock::STensor bX =
        bgraph.new_input(gX, {-1, -1, -1}, 1, layout::SmemRowMajor);
    threadblock::STensor bW_up =
        bgraph.new_input(gW_up, {1, -1, -1}, 0, layout::SmemRowMajor);
    threadblock::STensor bW_gate =
        bgraph.new_input(gW_gate, {1, -1, -1}, 0, layout::SmemRowMajor);

    threadblock::STensor bM_up   = bgraph.matmul(bX, bW_up);
    threadblock::STensor bM_gate = bgraph.matmul(bX, bW_gate);

    threadblock::STensor bAccRms =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
    threadblock::STensor bAccUp =
        bgraph.forloop_accum(bM_up, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    threadblock::STensor bAccGate =
        bgraph.forloop_accum(bM_gate, type::TB_FORLOOP_ACCUM_NO_RED_OP);

    // Sym style: div, div, mul
    threadblock::STensor bUp   = bgraph.div(bAccUp, bAccRms);
    threadblock::STensor bGate = bgraph.div(bAccGate, bAccRms);
    threadblock::STensor bO    = bgraph.mul(bUp, bGate);

    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    auto outs = g.customized({gX, gW_up, gW_gate}, bgraph);
    assert(outs.size() == 1);
    g.mark_output(outs[0]);

    auto match = verifier.verify(g);
    std::cout << "  Verifier result: " << (match.is_valid() ? "PASS" : "FAIL") << std::endl;
  }

  // ===========================================================================
  // Test 2: NS topology (mul, mul_rms, div)
  //   accum_up * accum_gate -> combined
  //   rms * rms -> rms_sq
  //   combined / rms_sq -> output
  // ===========================================================================
  {
    std::cout << "\n=== Test 2: NS topology (mul, mul_rms, div) ===" << std::endl;

    kernel::Graph g;
    kernel::DTensor gX = g.new_input(
        {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor gW_up = g.new_input(
        {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor gW_gate = g.new_input(
        {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

    dim3 grid_dim  = {64, 1, 1};
    dim3 block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim, block_dim, 64, 64);

    threadblock::STensor bX =
        bgraph.new_input(gX, {-1, -1, -1}, 1, layout::SmemRowMajor);
    threadblock::STensor bW_up =
        bgraph.new_input(gW_up, {1, -1, -1}, 0, layout::SmemRowMajor);
    threadblock::STensor bW_gate =
        bgraph.new_input(gW_gate, {1, -1, -1}, 0, layout::SmemRowMajor);

    threadblock::STensor bM_up   = bgraph.matmul(bX, bW_up);
    threadblock::STensor bM_gate = bgraph.matmul(bX, bW_gate);

    threadblock::STensor bAccRms =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
    threadblock::STensor bAccUp =
        bgraph.forloop_accum(bM_up, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    threadblock::STensor bAccGate =
        bgraph.forloop_accum(bM_gate, type::TB_FORLOOP_ACCUM_NO_RED_OP);

    // NS style: mul(up,gate), mul(rms,rms), div(combined, rms_sq)
    threadblock::STensor bCombined = bgraph.mul(bAccUp, bAccGate);
    threadblock::STensor bRmsSq    = bgraph.mul(bAccRms, bAccRms);
    threadblock::STensor bO        = bgraph.div(bCombined, bRmsSq);

    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    auto outs = g.customized({gX, gW_up, gW_gate}, bgraph);
    assert(outs.size() == 1);
    g.mark_output(outs[0]);

    auto match = verifier.verify(g);
    std::cout << "  Verifier result: " << (match.is_valid() ? "PASS" : "FAIL") << std::endl;
  }

  // ===========================================================================
  // Test 3: NS topology as symbolic graph (verify_symbolic_graph)
  // ===========================================================================
  {
    std::cout << "\n=== Test 3: NS topology as symbolic graph ===" << std::endl;

    search::SymbolicKNGraph sym_kn;

    // Add inputs matching the reference
    sym_kn.add_input({n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    sym_kn.add_input({d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    sym_kn.add_input({d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

    // Build TB graph
    search::SymbolicTBGraph tb(sym_kn.next_dim_variable_index, 1);

    // Add TB inputs
    tb.add_input(sym_kn.tensors[0], {-1}, 1);  // X, shared, forloop_dim=1
    tb.add_input(sym_kn.tensors[1], {1}, 0);   // W_up, partition cols, forloop_dim=0
    tb.add_input(sym_kn.tensors[2], {1}, 0);   // W_gate, partition cols, forloop_dim=0

    // matmul, matmul
    tb.add_operator(type::TBOperatorType::TB_MATMUL_OP, {0, 1});
    tb.add_operator(type::TBOperatorType::TB_MATMUL_OP, {0, 2});

    // forloop accumulators
    tb.add_operator(type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP, {0});  // rms(X)
    tb.add_operator(type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP, {3});       // accum_up
    tb.add_operator(type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP, {4});       // accum_gate

    // NS style: mul(up,gate), mul(rms,rms), div(combined, rms_sq)
    tb.add_operator(type::TBOperatorType::TB_MUL_OP, {6, 7});       // combined = up * gate
    tb.add_operator(type::TBOperatorType::TB_MUL_OP, {5, 5});       // rms_sq = rms * rms
    tb.add_operator(type::TBOperatorType::TB_DIV_OP, {8, 9});       // output = combined / rms_sq

    // Output
    tb.add_output(10, {1}, type::TB_EPILOGUE_NONE);

    // Add customized op to KN graph
    sym_kn.add_customized_operator(tb, {0, 1, 2});

    auto match = verifier.verify_symbolic_graph(sym_kn);
    std::cout << "  Verifier result: " << (match.is_valid() ? "PASS" : "FAIL") << std::endl;
  }

  std::cout << "\nDone." << std::endl;
  return 0;
}
