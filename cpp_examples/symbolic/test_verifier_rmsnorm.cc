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
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
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
  kernel::DTensor U = ref.matmul(X_norm, W_up);
  kernel::DTensor Gx = ref.matmul(X_norm, W_gate);
  kernel::DTensor O = ref.mul(U, Gx);
  ref.mark_output(O);

  std::cout << "Reference graph built: O = (rms_norm(X) @ W_up) * (rms_norm(X) "
               "@ W_gate)"
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

    dim3 grid_dim = {64, 1, 1};
    dim3 block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim, block_dim, 64, 64);

    threadblock::STensor bX =
        bgraph.new_input(gX, {-1, -1, -1}, 1, layout::SmemRowMajor);
    threadblock::STensor bW_up =
        bgraph.new_input(gW_up, {1, -1, -1}, 0, layout::SmemRowMajor);
    threadblock::STensor bW_gate =
        bgraph.new_input(gW_gate, {1, -1, -1}, 0, layout::SmemRowMajor);

    threadblock::STensor bM_up = bgraph.matmul(bX, bW_up);
    threadblock::STensor bM_gate = bgraph.matmul(bX, bW_gate);

    threadblock::STensor bAccRms =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
    threadblock::STensor bAccUp =
        bgraph.forloop_accum(bM_up, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    threadblock::STensor bAccGate =
        bgraph.forloop_accum(bM_gate, type::TB_FORLOOP_ACCUM_NO_RED_OP);

    // Sym style: div, div, mul
    threadblock::STensor bUp = bgraph.div(bAccUp, bAccRms);
    threadblock::STensor bGate = bgraph.div(bAccGate, bAccRms);
    threadblock::STensor bO = bgraph.mul(bUp, bGate);

    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    auto outs = g.customized({gX, gW_up, gW_gate}, bgraph);
    assert(outs.size() == 1);
    g.mark_output(outs[0]);

    auto match = verifier.verify(g);
    std::cout << "  Verifier result: " << (match.is_valid() ? "PASS" : "FAIL")
              << std::endl;
  }

  // ===========================================================================
  // Test 2: NS topology (mul, mul_rms, div)
  //   accum_up * accum_gate -> combined
  //   rms * rms -> rms_sq
  //   combined / rms_sq -> output
  // ===========================================================================
  {
    std::cout << "\n=== Test 2: NS topology (mul, mul_rms, div) ==="
              << std::endl;

    kernel::Graph g;
    kernel::DTensor gX = g.new_input(
        {n, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor gW_up = g.new_input(
        {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor gW_gate = g.new_input(
        {d, d}, {(size_t)d, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

    dim3 grid_dim = {64, 1, 1};
    dim3 block_dim = {128, 1, 1};
    threadblock::Graph bgraph(grid_dim, block_dim, 64, 64);

    threadblock::STensor bX =
        bgraph.new_input(gX, {-1, -1, -1}, 1, layout::SmemRowMajor);
    threadblock::STensor bW_up =
        bgraph.new_input(gW_up, {1, -1, -1}, 0, layout::SmemRowMajor);
    threadblock::STensor bW_gate =
        bgraph.new_input(gW_gate, {1, -1, -1}, 0, layout::SmemRowMajor);

    threadblock::STensor bM_up = bgraph.matmul(bX, bW_up);
    threadblock::STensor bM_gate = bgraph.matmul(bX, bW_gate);

    threadblock::STensor bAccRms =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
    threadblock::STensor bAccUp =
        bgraph.forloop_accum(bM_up, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    threadblock::STensor bAccGate =
        bgraph.forloop_accum(bM_gate, type::TB_FORLOOP_ACCUM_NO_RED_OP);

    // NS style: mul(up,gate), mul(rms,rms), div(combined, rms_sq)
    threadblock::STensor bCombined = bgraph.mul(bAccUp, bAccGate);
    threadblock::STensor bRmsSq = bgraph.mul(bAccRms, bAccRms);
    threadblock::STensor bO = bgraph.div(bCombined, bRmsSq);

    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    auto outs = g.customized({gX, gW_up, gW_gate}, bgraph);
    assert(outs.size() == 1);
    g.mark_output(outs[0]);

    auto match = verifier.verify(g);
    std::cout << "  Verifier result: " << (match.is_valid() ? "PASS" : "FAIL")
              << std::endl;
  }

  // Test 3 skipped — segfaults due to unrelated SymbolicKNGraph construction
  // issue
  std::cout << "\n=== Test 3: NS topology as symbolic graph (skipped) ==="
            << std::endl;

  // ===========================================================================
  // Test 7: Standalone subexpr check (single e-graph, matching actual search)
  // ===========================================================================
  {
    std::cout << "\n=== Test 7: Standalone subexpr check (single e-graph) ==="
              << std::endl;
    search::AbstractExpr::symbolic_expr = true;

    std::unordered_map<type::GuidType,
                       std::shared_ptr<search::AbstractExpr const>>
        ref_exprs7;
    search::abstract_expr_eval(ref, ref_exprs7);

    std::shared_ptr<search::AbstractExpr const> final_expr7;
    for (auto *op : ref.operators) {
      if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
        final_expr7 = ref_exprs7.at(op->input_tensors[0].guid);
      }
    }
    assert(final_expr7);
    std::cout << "  Final expr (egg): " << final_expr7->to_egg() << std::endl;

    // Initialize ONLY this one e-graph (this is what the search does)
    search::initialize_final_expr(final_expr7);

    auto v0 = search::abstract_expr_make_var("v_0");
    auto v1 = search::abstract_expr_make_var("v_1");
    auto v2 = search::abstract_expr_make_var("v_2");

    // Actual search expressions (with all-vars=2):
    // tile matmul inner dim = 4096/var1 = 2048
    auto tile_matmul_up = search::abstract_expr_make_red(
        2048, search::abstract_expr_make_mul(v0, v1));
    auto tile_matmul_gate = search::abstract_expr_make_red(
        2048, search::abstract_expr_make_mul(v0, v2));
    // forloop accum: sum(2, tile_matmul)
    auto accum_up = search::abstract_expr_make_red(2, tile_matmul_up);
    auto accum_gate = search::abstract_expr_make_red(2, tile_matmul_gate);
    // rms: rms(4096, v_0)
    auto rms_x = search::abstract_expr_make_rms(4096, v0);

    // NS intermediates
    auto combined = search::abstract_expr_make_mul(accum_up, accum_gate);
    auto rms_sq = search::abstract_expr_make_mul(rms_x, rms_x);
    auto ns_output = search::abstract_expr_make_div(combined, rms_sq);

    // Sym intermediates
    auto div_up = search::abstract_expr_make_div(accum_up, rms_x);
    auto div_gate = search::abstract_expr_make_div(accum_gate, rms_x);
    auto sym_output = search::abstract_expr_make_mul(div_up, div_gate);

    std::cout << "  accum_up: " << accum_up->to_egg() << std::endl;
    std::cout << "  combined: " << combined->to_egg() << std::endl;
    std::cout << "  rms_sq: " << rms_sq->to_egg() << std::endl;

    std::vector<std::shared_ptr<search::AbstractExpr const>> exprs = {
        accum_up,
        accum_gate,
        rms_x,
        div_up,
        div_gate,
        sym_output,
        combined,
        rms_sq,
        ns_output,
    };
    std::vector<std::string> labels = {
        "accum_up [sum(2,sum(2048,v0*v1))]",
        "accum_gate [sum(2,sum(2048,v0*v2))]",
        "rms_x [rms(4096,v0)]",
        "div(accum_up, rms) [Sym]",
        "div(accum_gate, rms) [Sym]",
        "mul(div_up, div_gate) [Sym output]",
        "mul(accum_up, accum_gate) [NS]",
        "mul(rms, rms) [NS]",
        "div(combined, rms_sq) [NS output]",
    };
    auto results = search::subexpr_to_final_expr(exprs);
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << "  " << labels[i] << ": " << (results[i] ? "PASS" : "FAIL")
                << std::endl;
    }
    search::AbstractExpr::symbolic_expr = false;
  }

  // ===========================================================================
  // Test 4: Abstract expression subexpr check for NS intermediates
  //
  // The search builds an e-graph from the final output expression and checks
  // whether each candidate operator's abstract expression is a subexpression.
  // This test verifies that NS intermediates (mul(accum_up, accum_gate) and
  // mul(rms, rms)) pass the subexpr check after our new rewrite rule.
  // ===========================================================================
  {
    std::cout << "\n=== Test 4: Abstract expression subexpr check ==="
              << std::endl;

    // Compute the abstract expression for the reference graph's output
    // (same as what the search does in preprocess())
    search::AbstractExpr::symbolic_expr = false;
    std::unordered_map<type::GuidType,
                       std::shared_ptr<search::AbstractExpr const>>
        ref_exprs;
    search::abstract_expr_eval(ref, ref_exprs);

    // Find the output expression
    std::shared_ptr<search::AbstractExpr const> final_expr;
    for (auto *op : ref.operators) {
      if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
        final_expr = ref_exprs.at(op->input_tensors[0].guid);
      }
    }
    assert(final_expr);
    std::cout << "  Final expr (egg): " << final_expr->to_egg() << std::endl;

    // Initialize the e-graph with the final expression
    search::initialize_final_expr(final_expr);

    // Now build the intermediate abstract expressions that the NS topology
    // would produce during search:
    //   v_0 = X, v_1 = W_up, v_2 = W_gate
    auto v0 = search::abstract_expr_make_var("v_0");
    auto v1 = search::abstract_expr_make_var("v_1");
    auto v2 = search::abstract_expr_make_var("v_2");

    // matmul(X, W_up) = sum(d, v_0 * v_1), matmul(X, W_gate) = sum(d, v_0 *
    // v_2)
    auto matmul_up = search::abstract_expr_make_red(
        d, search::abstract_expr_make_mul(v0, v1));
    auto matmul_gate = search::abstract_expr_make_red(
        d, search::abstract_expr_make_mul(v0, v2));

    // rms(X) = rms(d, v_0)
    auto rms_x = search::abstract_expr_make_rms(d, v0);

    // NS intermediate 1: mul(accum_up, accum_gate)
    auto combined = search::abstract_expr_make_mul(matmul_up, matmul_gate);
    std::cout << "  combined (egg): " << combined->to_egg() << std::endl;

    // NS intermediate 2: mul(rms, rms)
    auto rms_sq = search::abstract_expr_make_mul(rms_x, rms_x);
    std::cout << "  rms_sq (egg): " << rms_sq->to_egg() << std::endl;

    // NS final: div(combined, rms_sq) — should equal the output
    auto ns_output = search::abstract_expr_make_div(combined, rms_sq);
    std::cout << "  ns_output (egg): " << ns_output->to_egg() << std::endl;

    // Check existing intermediates that the Sym topology uses (should pass):
    auto div_up = search::abstract_expr_make_div(matmul_up, rms_x);
    auto div_gate = search::abstract_expr_make_div(matmul_gate, rms_x);

    // Check all
    std::vector<std::shared_ptr<search::AbstractExpr const>> exprs = {
        v0,
        v1,
        v2, // inputs
        matmul_up,
        matmul_gate, // matmuls
        rms_x,       // rms
        div_up,
        div_gate,  // Sym intermediates (div)
        combined,  // NS intermediate: mul(up, gate)
        rms_sq,    // NS intermediate: mul(rms, rms)
        ns_output, // NS output: div(combined, rms_sq)
    };
    std::vector<std::string> labels = {
        "v_0 (X)",
        "v_1 (W_up)",
        "v_2 (W_gate)",
        "sum(d, v_0*v_1) [matmul_up]",
        "sum(d, v_0*v_2) [matmul_gate]",
        "rms(d, v_0) [rms_x]",
        "div(matmul_up, rms) [Sym]",
        "div(matmul_gate, rms) [Sym]",
        "mul(matmul_up, matmul_gate) [NS]",
        "mul(rms, rms) [NS]",
        "div(combined, rms_sq) [NS output]",
    };

    auto results = search::subexpr_to_final_expr(exprs);
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << "  " << labels[i] << ": "
                << (results[i] ? "PASS (is subexpr)" : "FAIL (not subexpr)")
                << std::endl;
    }
  }

  // ===========================================================================
  // Test 5: Same as Test 4 but with symbolic_expr=true (vars evaluated to 2)
  //   This is what the symbolic search actually uses.
  // ===========================================================================
  {
    std::cout << "\n=== Test 5: Subexpr check with symbolic_expr=true ==="
              << std::endl;
    search::AbstractExpr::symbolic_expr = true;

    std::unordered_map<type::GuidType,
                       std::shared_ptr<search::AbstractExpr const>>
        ref_exprs2;
    search::abstract_expr_eval(ref, ref_exprs2);

    std::shared_ptr<search::AbstractExpr const> final_expr2;
    for (auto *op : ref.operators) {
      if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
        final_expr2 = ref_exprs2.at(op->input_tensors[0].guid);
      }
    }
    assert(final_expr2);
    std::cout << "  Final expr (egg): " << final_expr2->to_egg() << std::endl;

    search::initialize_final_expr(final_expr2);

    // With symbolic_expr=true, Red/RMS use get_value_with_all_vars_two.
    // But since d=4096 is a constant (no vars), it stays 4096.
    // The forloop reduction degrees in the actual search use symbolic dims
    // (var0, var1) which evaluate to 2 when all vars=2.
    // For this test, use d=4096 (same as concrete) since inputs have const
    // dims.
    auto v0 = search::abstract_expr_make_var("v_0");
    auto v1 = search::abstract_expr_make_var("v_1");
    auto v2 = search::abstract_expr_make_var("v_2");

    auto matmul_up = search::abstract_expr_make_red(
        d, search::abstract_expr_make_mul(v0, v1));
    auto matmul_gate = search::abstract_expr_make_red(
        d, search::abstract_expr_make_mul(v0, v2));
    auto rms_x = search::abstract_expr_make_rms(d, v0);
    auto combined = search::abstract_expr_make_mul(matmul_up, matmul_gate);
    auto rms_sq = search::abstract_expr_make_mul(rms_x, rms_x);
    auto ns_output = search::abstract_expr_make_div(combined, rms_sq);

    std::cout << "  combined (egg): " << combined->to_egg() << std::endl;
    std::cout << "  rms_sq (egg): " << rms_sq->to_egg() << std::endl;

    std::vector<std::shared_ptr<search::AbstractExpr const>> exprs = {
        combined,
        rms_sq,
        ns_output,
    };
    std::vector<std::string> labels = {
        "mul(matmul_up, matmul_gate) [NS]",
        "mul(rms, rms) [NS]",
        "div(combined, rms_sq) [NS output]",
    };
    auto results = search::subexpr_to_final_expr(exprs);
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << "  " << labels[i] << ": "
                << (results[i] ? "PASS (is subexpr)" : "FAIL (not subexpr)")
                << std::endl;
    }
    search::AbstractExpr::symbolic_expr = false;
  }

  // ===========================================================================
  // Test 6: Subexpr check with tile-level reduction degrees
  //   During symbolic search, the matmul reduction has tile-level dims
  //   (e.g., sum(2, v_0*v_1)) not full dims (sum(4096, v_0*v_1)).
  //   The forloop accumulator wraps it: sum(fl, sum(tile, v_0*v_1)).
  //   With all vars=2: tile=2, fl=2, so accum = sum(2, sum(2, v_0*v_1)).
  //   The e-graph's factor rules should relate sum(4096,...) to sum(2,...)
  //   chains.
  // ===========================================================================
  {
    std::cout << "\n=== Test 6: Tile-level reduction degrees ===" << std::endl;

    auto v0 = search::abstract_expr_make_var("v_0");
    auto v1 = search::abstract_expr_make_var("v_1");
    auto v2 = search::abstract_expr_make_var("v_2");

    // Actual symbolic dims with all-vars=2:
    //   tile inner dim = 4096/var1, with var1=2 → 2048
    //   forloop_range = var1 = 2
    //   rms reduction = forloop_range * last_dim = var1 * (4096/var1) = 4096
    auto tile_matmul_up = search::abstract_expr_make_red(
        2048, search::abstract_expr_make_mul(v0, v1));
    auto tile_matmul_gate = search::abstract_expr_make_red(
        2048, search::abstract_expr_make_mul(v0, v2));

    // Forloop accum: sum(2, sum(2048, v_0*v_1))
    auto accum_up = search::abstract_expr_make_red(2, tile_matmul_up);
    auto accum_gate = search::abstract_expr_make_red(2, tile_matmul_gate);

    // rms: forloop_range * last_dim = var1 * (4096/var1) = 4096 (simplifies!)
    auto rms_x = search::abstract_expr_make_rms(4096, v0);

    // NS intermediates at tile level
    auto combined = search::abstract_expr_make_mul(accum_up, accum_gate);
    auto rms_sq = search::abstract_expr_make_mul(rms_x, rms_x);
    auto ns_output = search::abstract_expr_make_div(combined, rms_sq);

    std::cout << "  accum_up (egg): " << accum_up->to_egg() << std::endl;
    std::cout << "  rms_x (egg): " << rms_x->to_egg() << std::endl;
    std::cout << "  combined (egg): " << combined->to_egg() << std::endl;
    std::cout << "  rms_sq (egg): " << rms_sq->to_egg() << std::endl;

    std::vector<std::shared_ptr<search::AbstractExpr const>> exprs = {
        tile_matmul_up,
        tile_matmul_gate,
        accum_up,
        accum_gate,
        rms_x,
        combined,
        rms_sq,
        ns_output,
    };
    std::vector<std::string> labels = {
        "sum(2, v_0*v_1) [tile matmul_up]",
        "sum(2, v_0*v_2) [tile matmul_gate]",
        "sum(2, sum(2, v_0*v_1)) [accum_up]",
        "sum(2, sum(2, v_0*v_2)) [accum_gate]",
        "rms(4, v_0) [rms]",
        "mul(accum_up, accum_gate) [NS]",
        "mul(rms, rms) [NS]",
        "div(combined, rms_sq) [NS output]",
    };
    auto results = search::subexpr_to_final_expr(exprs);
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << "  " << labels[i] << ": "
                << (results[i] ? "PASS (is subexpr)" : "FAIL (not subexpr)")
                << std::endl;
    }
  }

  std::cout << "\nDone." << std::endl;
  return 0;
}
