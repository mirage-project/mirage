#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

#include <iostream>

using namespace mirage;
using namespace mirage::search;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Missing checkpoint file" << std::endl;
    return 1;
  }

  auto st = std::chrono::steady_clock::now();

  std::unordered_set<int> index_to_skip;
  for (int i = 2; i < argc; ++i) {
    index_to_skip.insert(std::atoi(argv[i]));
  }

  search::KernelGraphGenerator gen(argv[1]);

  int index = 0;
  for (json const &j : gen.generated_graphs) {
    std::cout << "optimizing " << j << std::endl;
    if (index_to_skip.find(index) == index_to_skip.end()) {
      kernel::Graph g;
      from_json(j, g);
      gen.optimize_layout(g);
      gen.save_checkpoint();
      while (!g.operators.empty()) {
        delete g.operators.back();
        g.operators.pop_back();
      }
    }
    std::cout << "finished graph" << (index++) << std::endl;
  }

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n", std::chrono::duration<double>(et - st).count());

  return 0;
}
