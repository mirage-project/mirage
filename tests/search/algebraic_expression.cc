#include <gtest/gtest.h>

#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

using namespace mirage;
using namespace search;

TEST(algebraic_expression, basic) {
  kernel::Graph graph;

  int red_dim = 4096;

  kernel::DTensor Q = graph.new_input({64, red_dim}, mirage::type::DT_FLOAT16);
  kernel::DTensor K = graph.new_input({red_dim, 16384}, mirage::type::DT_FLOAT16);

  std::shared_ptr<AlgebraicPattern> Q_pattern = std::make_shared<Var>("q");
  std::shared_ptr<AlgebraicPattern> K_pattern = std::make_shared<Var>("k");

  kernel::DTensor matmul = graph.matmul(Q, K);

  {
    std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>>
        input_expression_map{{Q, Q_pattern}, {K, K_pattern}};
    std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> results =
        pattern_eval(graph, input_expression_map);

    std::shared_ptr<AlgebraicPattern> output_pattern = results.at(matmul);
    std::shared_ptr<AlgebraicPattern> target_pattern = std::make_shared<Red>(
        Q.dim[1], std::make_shared<Mul>(K_pattern, Q_pattern));
    std::shared_ptr<AlgebraicPattern> mul_pattern =
        std::make_shared<Mul>(Q_pattern, K_pattern);
    std::shared_ptr<AlgebraicPattern> another_pattern =
        std::make_shared<Mul>(K_pattern, std::make_shared<Exp>(Q_pattern));
    std::shared_ptr<AlgebraicPattern> V_pattern = std::make_shared<Var>("v");
    std::shared_ptr<AlgebraicPattern> larger_pattern = std::make_shared<Red>(
        4, std::make_shared<Red>(red_dim / 2, mul_pattern));
    std::shared_ptr<AlgebraicPattern> larger_pattern2 =
        std::make_shared<Red>(2, std::make_shared<Red>(red_dim, mul_pattern));
    std::shared_ptr<AlgebraicPattern> larger_pattern3 =
        std::make_shared<Red>(red_dim * 2, mul_pattern);
    std::shared_ptr<AlgebraicPattern> exp_pattern =
        std::make_shared<Exp>(V_pattern);

    EXPECT_TRUE(output_pattern->subpattern_to(*target_pattern));
    EXPECT_TRUE(target_pattern->subpattern_to(*output_pattern));
    EXPECT_TRUE(mul_pattern->subpattern_to(*output_pattern));
    EXPECT_TRUE(output_pattern->subpattern_to(*larger_pattern));
    EXPECT_TRUE(output_pattern->subpattern_to(*larger_pattern2));
    EXPECT_TRUE(output_pattern->subpattern_to(*larger_pattern3));

    EXPECT_FALSE(output_pattern->subpattern_to(*another_pattern));
    EXPECT_FALSE(Q_pattern->subpattern_to(*V_pattern));
    EXPECT_FALSE(V_pattern->subpattern_to(*output_pattern));
    EXPECT_FALSE(exp_pattern->subpattern_to(*another_pattern));
  }
}