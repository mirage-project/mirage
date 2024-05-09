#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

#include <iostream>
#include <fstream>

using namespace mirage;
using namespace mirage::search;

int main(int argc, char **argv) {  
  if (argc < 2) {
    std::cerr << "Miss graph file name" << std::endl;
    return 1;
  }

  kernel::Graph g;
  std::ifstream ifs(argv[1]);
  json j;
  ifs >> j;
  from_json(j, g);

  std::cout << json(g) << std::endl;

  for (auto op : g.operators) {
    op->fingerprint();
  }

  float run_time = 0;
  for (auto op : g.operators) {
    ProfileResult op_result;
    op->profile(op_result);
    run_time += op_result.run_time;
  }

  std::cout << "Profiled running time: " << run_time << std::endl;

  return 0;
}
