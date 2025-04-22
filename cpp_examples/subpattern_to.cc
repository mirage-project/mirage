#include "mirage/search/search.h"
#include "mirage/search/abstract_expr/abstract_expr.h"

using namespace mirage::search;


int main() {
    // Example usage of the AbstractExpr class and its derived classes
    auto v0 = std::make_shared<Var>("v_0");
    auto v1 = std::make_shared<Var>("v_1");
    auto v2 = std::make_shared<Var>("v_2");

    auto red_v0_4 = std::make_shared<Red>(4, v0);
    auto red_v2_4 = std::make_shared<Red>(4, v2);
    auto mul_1 = std::make_shared<Mul>(red_v0_4, red_v2_4);
    auto expr_1 = std::make_shared<Red>(4, mul_1);

    auto red_v1_16 = std::make_shared<Red>(16, v1);
    auto red_v0_16 = std::make_shared<Red>(16, v0);
    auto mul_2 = std::make_shared<Mul>(red_v1_16, red_v0_16);
    auto expr_2 = std::make_shared<Red>(2, mul_2);

    auto red_v1_4 = std::make_shared<Red>(4, v1);
    auto red_v0_4_b = std::make_shared<Red>(4, v0);  
    auto mul_3 = std::make_shared<Mul>(red_v1_4, red_v0_4_b);
    auto expr_3 = std::make_shared<Red>(4, mul_3);

    auto red_v2_64 = std::make_shared<Red>(64, v2);
    auto red_v0_64 = std::make_shared<Red>(64, v0);
    auto mul_4 = std::make_shared<Mul>(red_v2_64, red_v0_64);
    auto expr_4 = std::make_shared<Red>(2, mul_4);

    auto mul_v0_v1 = std::make_shared<Mul>(v0, v1);
    auto red_mul_4096 = std::make_shared<Red>(4096, mul_v0_v1);
    auto silu_expr = std::make_shared<Silu>(red_mul_4096);
    auto red_silu_16 = std::make_shared<Red>(16, silu_expr);
    auto red_v2_16 = std::make_shared<Red>(16, v2);
    auto mul_5 = std::make_shared<Mul>(red_silu_16, red_v2_16);
    auto expr_5 = std::make_shared<Red>(4096, mul_5);

    auto mul_6 = std::make_shared<Mul>(red_silu_16, red_v2_16);
    auto expr_6 = std::make_shared<Red>(8, mul_6);

    auto mul_v0_v2 = std::make_shared<Mul>(v0, v2);
    auto red_mul_v0v2 = std::make_shared<Red>(4096, mul_v0_v2);
    auto mul_v0_v1_b = std::make_shared<Mul>(v0, v1); 
    auto red_mul_v0v1 = std::make_shared<Red>(4096, mul_v0_v1_b);
    auto silu_expr_7 = std::make_shared<Silu>(red_mul_v0v1);
    auto expr_7 = std::make_shared<Mul>(silu_expr_7, red_mul_v0v2);

    std::shared_ptr<AbstractExpr> expr_null = nullptr;

    std::vector<std::shared_ptr<AbstractExpr>> input_patterns = {expr_1, expr_2, expr_3, expr_4, expr_5, expr_6, expr_null};
    std::unordered_map<int, bool> result = expr_7->subpattern_to(input_patterns);
    
    bool result1 = result[0];
    bool result2 = result[1];
    bool result3 = result[2];
    bool result4 = result[3];   
    bool result5 = result[4];    
    bool result6 = result[5];
    bool result7 = result[6];
    // Print the Z3 expression
    std::cout << "Result 1: " << result1 << std::endl;
    std::cout << "Result 2: " << result2 << std::endl;
    std::cout << "Result 3: " << result3 << std::endl;
    std::cout << "Result 4: " << result4 << std::endl;
    std::cout << "Result 5: " << result5 << std::endl;
    std::cout << "Result 6: " << result6 << std::endl;
    std::cout << "Result 7: " << result7 << std::endl;

    return 0;
}
