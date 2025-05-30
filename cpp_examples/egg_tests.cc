#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

extern "C" {

struct KVPair {
  int key;
  bool value;
};

KVPair *egg_equiv(char const **inputs, int len);
void get_egraph(char const *expr);
}

int main() {
  std::vector<std::string> sub_exprs = {
      "(sum 4096 (* v_0 v_1))",
      "(* (sum 32 v_0) (sum 128 v_2))",
      "(sum 4 (* (sum 4 v_0) (sum 4 v_2)))",
      "(sum 2 (* (sum 16 v_1) (sum 16 v_0)))",
      "(sum 4 (* (sum 4 v_1) (sum 4 v_0)))",
      "(sum 2 (* (sum 64 v_2) (sum 64 v_0)))",
      "null",
      "(sum 4096 (* (sum 16 (silu (sum 4096 (* v_0 v_1)))) (sum 16 v_2)))",
      "(sum 8 (* (sum 16 (silu (sum 4096 (* v_0 v_1)))) (sum 16 v_2)))",
  };

  std::vector<bool> is_valid;
  for (auto &s : sub_exprs) {
    if (s == "null") {
      is_valid.push_back(false);
    } else {
      is_valid.push_back(true);
    }
  }

  std::vector<char const *> c_subexpr;
  c_subexpr.reserve(sub_exprs.size());
  for (auto const &s : sub_exprs) {
    c_subexpr.push_back(s.c_str());
  }

  std::string expr = "(* (silu (sum 4096 (* v_0 v_1))) (sum 4096 (* v_0 v_2)))";
  get_egraph(expr.c_str());
  KVPair *datas =
      egg_equiv(c_subexpr.data(), static_cast<int>(c_subexpr.size()));

  std::unordered_map<int, bool> result;
  size_t len = c_subexpr.size();
  for (size_t i = 0; i < len; ++i) {
    if (is_valid[i]) {
      result[datas[i].key] = datas[i].value;
    }
  }

  for (auto const &pair : result) {
    std::cout << "Key: " << pair.first << ", Value: " << pair.second
              << std::endl;
  }

  return 0;
}
