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

  search::KernelGraphGenerator gen(argv[1]);

  gen.generate_kernel_graphs();

  auto et = std::chrono::steady_clock::now();

  printf("Search time = %.4lfsec\n",
         std::chrono::duration<double>(et - st).count());

  return 0;
}
