#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"

namespace mirage {
namespace search {

struct ProfileResult {
  float run_time;
  bool is_success;
  std::string error_message;
  std::string cuda_code;
};

ProfileResult profile(kernel::Graph *graph);

} // namespace search
} // namespace mirage