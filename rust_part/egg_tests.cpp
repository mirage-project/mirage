#include <iostream>

using namespace std;

extern "C" {
    bool egg_equiv(const char* expr1, const char* expr2);
}

int main() {
    cout << "Hello, World!" << endl;

    std::string expr1 = "(+ a b)";
    std::string expr2 = "b";
    bool result = egg_equiv(expr1.c_str(), expr2.c_str());
    std::cout << "Result: " << result << std::endl;

    return 0;

}
