#include <iostream>

using namespace std;

extern "C" {
    bool egg_equiv(const char* expr1, const char* expr2);
}

int main() {
    std::string expr1 = "(sum 12 v_0)";
    std::string expr2 = "(sum 48 v_0)";

    std::string expr3 = "(sum 4 (sum 12 (* v_0 v_2)))";
    std::string expr4 = "(+ (sum 12 (* v_0 v_1)) (sum 4 (* (sum 12 (* v_0 v_2)) v_3)))";
    bool result1 = egg_equiv(expr1.c_str(), expr4.c_str());
    bool result2 = egg_equiv(expr2.c_str(), expr4.c_str());
    bool result3 = egg_equiv(expr3.c_str(), expr4.c_str());
    std::cout << "Result: " << result1 << result2 << result3 << std::endl;

    return 0;

}
