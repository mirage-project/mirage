#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"

#include <string>
#include <vector>

namespace mirage {
namespace search {

struct ProfileResult {
  float run_time;
  bool is_success;
  std::string error_message;
  std::string cuda_code;
};

/** Result of compiling a kernel graph to a shared library (no GPU run). */
struct ProfileCompileResult {
  bool is_success = false;
  std::string error_message;
  std::string so_file;
  std::vector<size_t> input_num_elements;
  std::vector<size_t> output_alloc_sizes;
  size_t buf_size = 0;
  size_t profiler_buf_size = 0;
  std::string cuda_code;
};

/** Compile graph to .so (CPU-only, safe to call from multiple threads). */
ProfileCompileResult profile_compile(kernel::Graph *graph);

/** Run compiled kernel on GPU and return runtime. Uses a global mutex so only
 *  one thread runs on GPU at a time (avoids GPU resource conflict). */
ProfileResult profile_run(ProfileCompileResult const &compiled);

ProfileResult profile(kernel::Graph *graph);

} // namespace search
} // namespace mirage