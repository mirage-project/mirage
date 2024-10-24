#pragma once

#include <dlfcn.h>

#include <cassert>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/transpile.h"

#include "config.h"

using std::cout, std::cerr, std::endl, std::string, std::vector, std::optional,
    std::nullopt, std::pair, std::array;
using namespace mirage;
namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;
namespace trans = mirage::transpiler;

template <typename T>
inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

// Round to the next multiple
template <typename T>
inline T round_to_multiple(T value, T multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

size_t get_cumulative_mul(vector<int> const &v) {
  size_t result = 1;
  for (int x : v) {
    result *= x;
  }
  return result;
}

// Print out a dtensor
void print_dtensor(kn::DTensor const &dtensor, vector<half> const &data) {
  int cur_elem_idx = 0;
  std::function<void(int, int)> print = [&](int dim_idx, int indent) {
    if (dim_idx == dtensor.num_dims - 1) {
      cout << string(indent, ' ') << "[";
      for (int i = 0; i < dtensor.dim[dim_idx]; i++) {
        cout << (float)data[cur_elem_idx] << ",";
        cur_elem_idx++;
      }
      cout << "],\n";
      return;
    }
    cout << string(indent, ' ') << "[\n";
    for (int i = 0; i < dtensor.dim[dim_idx]; i++) {
      print(dim_idx + 1, indent + 2);
    }
    cout << string(indent, ' ') << "],\n";
  };
  print(0, 0);
}

// Check the return value of the CUDA runtime API call
#define CHECK_CUDA(status)                                                     \
  do {                                                                         \
    if (status != 0) {                                                         \
      std::cerr << "At file " << __FILE__ << ", line " << __LINE__             \
                << std::endl;                                                  \
      std::cerr << "Cuda failure: " << status << " "                           \
                << cudaGetErrorString(status) << std::endl;                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// A timer for measuring time on GPU
class GPUTimer {
private:
  cudaEvent_t start_event, stop_event;

public:
  GPUTimer() {
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
  }

  ~GPUTimer() {
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
  }

  void start() {
    CHECK_CUDA(cudaEventRecord(start_event, 0));
  }

  float stop() {
    CHECK_CUDA(cudaEventRecord(stop_event, 0));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    float time_usage_ms;
    CHECK_CUDA(cudaEventElapsedTime(&time_usage_ms, start_event, stop_event));
    return time_usage_ms;
  }
};

// Checking whether two numbers are the same
template <typename T>
inline bool is_equal(T const &a, T const &b) {
  return a == b;
}

template <typename T>
inline bool is_equal_helper(T const &a_,
                            T const &b_,
                            float max_abs_err,
                            float max_rel_err) {
  float a = (float)a_, b = (float)b_;
  float abs_err = std::abs(a - b);
  float rel_err = abs_err / std::max(std::abs(a), std::abs(b));
  return abs_err <= max_abs_err || rel_err <= max_rel_err;
}

template <>
inline bool is_equal(half const &a, half const &b) {
  return is_equal_helper(a, b, 2e-2, 5e-2);
}

// Functions for generating a list of numbers
namespace Gen {

class ARange {
private:
  half lower, upper;

public:
  ARange() : lower(-1.0), upper(1.0) {}
  ARange(half lower, half upper) : lower(lower), upper(upper) {}

  vector<half> operator()(vector<int> const &dims) const {
    size_t numel = get_cumulative_mul(dims);
    vector<half> result(numel);
    float start_f = (float)lower;
    float step = (float)(upper - lower) / (float)numel;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numel; i++) {
      result[i] = (half)(start_f + i * step);
    }
    return result;
  }
};

} // namespace Gen

// An input/output in the mugraph
struct IOTensor {
  vector<int> dims;
  vector<size_t> strides;

  // For input tensor, it's the input data
  // For output tensor, it's the standard answer (can be empty)
  vector<half> data;

  IOTensor() {}
  IOTensor(vector<int> const &dims,
           vector<size_t> const &strides,
           vector<half> const &data)
      : dims(dims), strides(strides), data(data) {
    assert(data.empty() || data.size() == numel());
  }

  size_t numel() const {
    size_t result = 1;
    for (int x : dims) {
      result *= x;
    }
    return result;
  }

  // Get the sizeof physical space (in elems) for a tensor
  size_t get_physical_size() const {
    size_t result = 0;
    for (size_t i = 0; i < dims.size(); ++i) {
      size_t cur_result = (size_t)dims[i] * strides[i];
      result = std::max(result, cur_result);
    }
    return result;
  }

  size_t logical_pos2phy_pos(size_t logical_pos) const {
    size_t result = 0;
    for (int i = (int)dims.size() - 1; i >= 0; --i) {
      result += (logical_pos % dims[i]) * strides[i];
      logical_pos /= dims[i];
    }
    return result;
  }

  vector<size_t> get_logical_coord(size_t logical_pos) const {
    vector<size_t> result(dims.size());
    for (int i = (int)dims.size() - 1; i >= 0; --i) {
      result[i] = logical_pos % dims[i];
      logical_pos /= dims[i];
    }
    return result;
  }

  // Allocate the tensor on the host
  half *allocate_space_host() const {
    size_t phy_size = get_physical_size() * sizeof(half);
    half *result = (half *)malloc(phy_size);
    return result;
  }

  // Allocate the tensor on the device
  half *allocate_space_dev() const {
    size_t phy_size = get_physical_size() * sizeof(half);
    half *result;
    CHECK_CUDA(cudaMalloc((void **)&result, phy_size));
    return result;
  }

  // Transpile the tensor to an allocated space on the host
  void transcribe_host(half *result) const {
    assert(!data.empty() && "Trying to transcribe a tensor without data");
    size_t numel = this->numel();
    for (size_t i = 0; i < numel; ++i) {
      result[logical_pos2phy_pos(i)] = data[i];
    }
  }

  // Transpile the tensor to an allocated space on the device
  void transcribe_dev(half *result) const {
    half *ptr_h = allocate_space_host();
    transcribe_host(ptr_h);
    size_t phy_size = get_physical_size() * sizeof(half);
    CHECK_CUDA(cudaMemcpy(result, ptr_h, phy_size, cudaMemcpyHostToDevice));
    free(ptr_h);
  }
};

class Subcase {
public:
  struct SubcaseConfig {
    int num_warmups;                           // Number of warmup iterations
    int num_runs;                              // Number of runs
    bool check_answer;                         // Whether to check the answer
    trans::TranspilerConfig transpiler_config; // Transpiler configuration
  };

  struct RunResult {
    string subcase_name;
    bool is_passed;
    float avg_time_ms;
    string msg;
  };

private:
  string subcase_name;
  SubcaseConfig config;
  vector<IOTensor> inputs;
  vector<pair<IOTensor, kn::DTensor>> outputs;

  using epilogue_t = std::function<string(Subcase::RunResult)>;
  optional<epilogue_t> epilogue;

  int subcase_id;
  static std::atomic<int> next_subcase_id;
  string cu_file_path;
  string so_file_path;

public:
  std::shared_ptr<kn::Graph> g;

  Subcase(SubcaseConfig const &config,
          optional<string> const &subcase_name = nullopt,
          optional<epilogue_t> const &epilogue = nullopt)
      : subcase_name(subcase_name.value_or("")), config(config),
        epilogue(epilogue), g(std::make_shared<kn::Graph>()) {
    subcase_id = next_subcase_id++;
    cu_file_path = env_config::get_generated_cu_dir() + "/" +
                   std::to_string(subcase_id) + ".cu";
    so_file_path = env_config::get_generated_so_dir() + "/" +
                   std::to_string(subcase_id) + ".so";
  }

  // Remove the generated .so file
  void remove_so_file() {
    remove(so_file_path.c_str());
  }

  // Create a new input to the graph, and add the input to the `inputs` list
  // If `strides_` is not provided, will infer a row-major layout
  template <typename generator_t>
  kn::DTensor new_input(vector<int> const &dims,
                        generator_t const &generator,
                        optional<vector<size_t>> const &strides_ = nullopt) {
    vector<size_t> strides;
    if (strides_.has_value()) {
      strides = strides_.value();
    } else {
      strides.resize(dims.size());
      size_t last_stride = 1;
      for (int i = (int)dims.size() - 1; i >= 0; --i) {
        strides[i] = last_stride;
        last_stride *= round_to_multiple((size_t)dims[i], 16 / sizeof(half));
      }
    }
    vector<half> data = generator(dims);
    this->inputs.push_back({dims, strides, data});
    kn::DTensor result =
        g->new_input(dims, type::DT_FLOAT16, layout::DmemRowMajor);
    // print_dtensor(result, data);
    return result;
  }

  void mark_output(kn::DTensor const &dtensor,
                   vector<int> const &dims,
                   vector<half> const &std_answer) {
    // def print_tensor(t): print(','.join(map(lambda x: "%.3f"%x,
    // t.flatten().tolist())))
    vector<size_t> strides = {}; // Will be filled after transpile()
    this->outputs.push_back({{dims, strides, std_answer}, dtensor});
  }

  // A handy function for adding a custom operator
  // Here we use std::array as the return type in order to let the user leverage
  // structured binding (a C++17 feature)
  template <int NUM_OUTPUTS>
  using add_custom_op_ret_t =
      pair<std::shared_ptr<tb::Graph>, array<kn::DTensor, NUM_OUTPUTS>>;

  template <int NUM_OUTPUTS>
  array<kn::DTensor, NUM_OUTPUTS>
      add_custom_op(vector<kn::DTensor> const &inputs,
                    std::function<add_custom_op_ret_t<NUM_OUTPUTS>(
                        vector<kn::DTensor> const &)> const &constructor) {
    auto [tb_graph_ptr, result_dtensors] = constructor(inputs);
    std::vector<kn::DTensor> output_dtensors =
        this->g->customized(inputs, *tb_graph_ptr);
    std::array<kn::DTensor, NUM_OUTPUTS> result_dtensors_arr;
    assert(output_dtensors.size() == NUM_OUTPUTS);
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
      result_dtensors_arr[i] = output_dtensors[i];
    }
    return result_dtensors_arr;
  }

private:
  // Call `mirage::transpiler::Transpile` to transpile the graph, and return the
  // result
  trans::TranspileResult transpile() const {
    // Transpile
    printf("Transpiling subcase %s...\n", subcase_name.c_str());
    vector<vector<size_t>> input_strides;
    for (auto const &input : inputs) {
      input_strides.push_back(input.strides);
    }
    vector<kn::DTensor const *> output_tensor_ptrs;
    for (auto const &output : outputs) {
      output_tensor_ptrs.push_back(&output.second);
    }
    trans::TranspileResult trans_result = trans::transpile(
        g.get(), config.transpiler_config, input_strides, output_tensor_ptrs);

    // Print transpiled result
    printf("Code: see %s\n", cu_file_path.c_str());
    printf("Buf size: %lu B (%.2f GB)\n",
           trans_result.buf_size,
           (float)trans_result.buf_size / (1l << 30));
    printf("Output tensor data:\n");
    for (auto const &output_directive : trans_result.output_directives) {
      printf("  phy_size: %lu B (%lu elems), ",
             output_directive.alloc_size,
             output_directive.alloc_size / sizeof(half));
      printf("shape: [");
      for (size_t x : output_directive.shape) {
        printf("%lu, ", x);
      }
      printf("\b\b], ");
      printf("strides: [");
      for (size_t x : output_directive.strides) {
        printf("%lu, ", x);
      }
      printf("\b\b]\n");
    }

    return trans_result;
  }

  // Save the generated code to a file, and compile it
  // Can be parallelized to speed up
  void save_and_compile(string const &code) const {
    FILE *fp = fopen(cu_file_path.c_str(), "w");
    if (fp == nullptr) {
      perror("fopen");
      exit(1);
    }
    fprintf(fp, "%s\n", code.c_str());
    fclose(fp);
    printf("Compiling subcase %s (%s -> %s)...\n",
           subcase_name.c_str(),
           cu_file_path.c_str(),
           so_file_path.c_str());
    vector<string> compile_options = {
        "-o " + so_file_path,
        "-shared",
        "-Xcompiler=-fPIC",
        "--expt-relaxed-constexpr",
        "-arch=native",
        "-use_fast_math",
        "-Xcompiler=-Wall",
        "-std=c++17",
        "-I" + env_config::get_cutlass_root() + "/include",
        "-I" + env_config::get_mirage_runtime_root()};
    string compile_command = "nvcc " + cu_file_path;
    for (string const &option : compile_options) {
      compile_command += " " + option;
    }
    int ret_code = system(compile_command.c_str());
    if (ret_code != 0) {
      printf("Compilation failed for subcase %s\n", subcase_name.c_str());
      exit(1);
    }
    printf("Compiled subcase %s.\n", subcase_name.c_str());
  }

  // Run the testcase
  RunResult run(trans::TranspileResult const &trans_result) {
    // Import the symbol
    printf("Running subcase %s (%s)...\n",
           subcase_name.c_str(),
           cu_file_path.c_str());
    printf("Loading generated CUDA program...\n");
    static int so_file_idx =
        0; // Here rename every generated .so file to a different name,
           // otherwise the so won't be reloaded
    so_file_idx += 1;
    string id_str = std::to_string(so_file_idx);
    void *handle = dlopen(so_file_path.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
      fprintf(stderr, "Failed to open generated.so: %s\n", dlerror());
      exit(1);
    }
    auto execute_mugraph_func =
        (void (*)(std::vector<void *>, std::vector<void *>, void *))dlsym(
            handle, "execute_mugraph");
    if (execute_mugraph_func == nullptr) {
      fprintf(stderr, "Failed to load execute_mugraph: %s\n", dlerror());
      exit(1);
    }

    // Allocate space for input tensor & output tensor & buf
    printf("Allocating tensors...\n");
    // Allocate input tensors
    vector<void *> input_ptrs_d;
    for (IOTensor const &input : this->inputs) {
      half *data_d = input.allocate_space_dev();
      input.transcribe_dev(data_d);
      input_ptrs_d.push_back(data_d);
    }
    // Update strides for output tensors and allocate them
    vector<void *> output_ptrs_d;
    for (size_t i = 0; i < outputs.size(); ++i) {
      IOTensor &output_iotensor = outputs[i].first;
      output_iotensor.strides = trans_result.output_directives[i].strides;
      half *data_d;
      CHECK_CUDA(cudaMalloc((void **)&data_d,
                            trans_result.output_directives[i].alloc_size *
                                sizeof(half)));
      output_ptrs_d.push_back(data_d);
    }
    // Allocate the buf
    void *buf_ptr_d;
    CHECK_CUDA(cudaMalloc(&buf_ptr_d, trans_result.buf_size));

    // Warm up
    bool is_correct = true;
    printf("Warming up...\n");
    for (int warmup_iter = 0; warmup_iter < config.num_warmups; ++warmup_iter) {
      execute_mugraph_func(input_ptrs_d, output_ptrs_d, buf_ptr_d);
      CHECK_CUDA(cudaDeviceSynchronize());
      if (warmup_iter == 0 && config.check_answer) {
        printf("Checking answer...\n");
        for (int output_idx = 0; output_idx < (int)this->outputs.size();
             ++output_idx) {
          IOTensor const &output = this->outputs[output_idx].first;
          printf("Output #%d:\n", output_idx);

          // Copy the answer to the host
          half *output_ptr_h = output.allocate_space_host();
          size_t phy_size = output.get_physical_size() * sizeof(half);
          CHECK_CUDA(cudaMemcpy(output_ptr_h,
                                output_ptrs_d[output_idx],
                                phy_size,
                                cudaMemcpyDeviceToHost));

          // Create the standard answer on the host
          half *std_answer_h = output.allocate_space_host();
          output.transcribe_host(std_answer_h);

          size_t numel = output.numel();
          vector<size_t> mismatch_indexes;
          for (size_t i = 0; i < numel; ++i) {
            size_t phy_pos = output.logical_pos2phy_pos(i);
            if (!is_equal(output_ptr_h[phy_pos], std_answer_h[phy_pos])) {
              mismatch_indexes.push_back(i);
            }
          }
          printf("%lu out of %lu (%.2f%%) mismatched.\n",
                 mismatch_indexes.size(),
                 numel,
                 (float)mismatch_indexes.size() / numel * 100.0);
          for (size_t i = 0; i < std::min(10ul, mismatch_indexes.size()); ++i) {
            size_t logical_pos = mismatch_indexes[i];
            size_t phy_pos = output.logical_pos2phy_pos(logical_pos);
            vector<size_t> coord = output.get_logical_coord(logical_pos);
            float truth = (float)std_answer_h[phy_pos];
            float answer = (float)output_ptr_h[phy_pos];
            float abs_err = std::abs(truth - answer);
            float rel_err =
                abs_err / std::max(std::abs(truth), std::abs(answer));
            printf("Mismatch at index %lu, phy index %lu, coord: [",
                   logical_pos,
                   phy_pos);
            for (size_t x : coord) {
              printf("%lu,", x);
            }
            printf("\b]:\n");
            printf("  expected %f, got %f (abs: %f, rel: %.2f%%)\n",
                   truth,
                   answer,
                   abs_err,
                   rel_err * 100);
          }
          if (mismatch_indexes.size() < numel * 0.01) {
            printf("\e[97;42;1m Passed. \e[0m\n");
          } else {
            printf("\e[97;41;1m Failed. \e[0m\n");
            is_correct = false;
          }
          free(output_ptr_h);
          free(std_answer_h);
        }
      }
    }

    // Run and benchmark
    float avg_time = 0;
    if (config.num_runs > 0) {
      printf("Running...\n");
      GPUTimer timer;
      timer.start();
      for (int run_iter = 0; run_iter < config.num_runs; ++run_iter) {
        execute_mugraph_func(input_ptrs_d, output_ptrs_d, buf_ptr_d);
      }
      float elapsed_time = timer.stop();
      CHECK_CUDA(cudaDeviceSynchronize());
      avg_time = elapsed_time / config.num_runs;
      printf("Avg time consumption: %.2f ms\n", avg_time);
    }

    // Clean up
    dlclose(handle);
    cudaFree(buf_ptr_d);
    for (void *ptr : output_ptrs_d) {
      cudaFree(ptr);
    }
    for (void *ptr : input_ptrs_d) {
      cudaFree(ptr);
    }

    RunResult result = {subcase_name, is_correct, avg_time, ""};
    if (epilogue) {
      result.msg = epilogue.value()(result);
      printf("%s\n", result.msg.c_str());
    }
    return result;
  }
  friend class Testcase;
};
std::atomic<int> Subcase::next_subcase_id = 0;

class Testcase {
private:
public:
  string name;
  vector<string> tags;
  string description;
  std::function<void(Testcase *)> logic;

  vector<Subcase> subcases;
  vector<Subcase::RunResult> run_results;

  Testcase(string const &name,
           vector<string> const &tags,
           string const &description,
           std::function<void(Testcase *)> const &logic)
      : name(name), tags(tags), description(description), logic(logic) {}

  vector<Subcase::RunResult> run() {
    // Call `logic()` to add all subcases
    logic(this);

    // Process testcase in groups. Within each group, we transpile all subcases
    // first, then compile them in parallel, and finally run them one by one
    vector<trans::TranspileResult> trans_results;
    int num_subcases = subcases.size();
    int num_compile_threads = env_config::get_num_compile_threads();
    printf("Will use %d threads to compile in parallel\n", num_compile_threads);
    for (int start_subcase = 0; start_subcase < num_subcases;
         start_subcase += num_compile_threads) {
      int end_subcase =
          std::min(start_subcase + num_compile_threads, num_subcases);
      printf(
          "================================================================\n");
      printf("Processing subcase %d ~ %d (%d in total)\n",
             start_subcase,
             end_subcase - 1,
             num_subcases);
      // Transpile them one by one (serial)
      printf("Step 0: Transpile\n");
      for (int i = start_subcase; i < end_subcase; ++i) {
        printf("----------------\n");
        auto trans_result = subcases[i].transpile();
        trans_results.push_back(trans_result);
      }
      assert((int)trans_results.size() == end_subcase);
      // Compile them in parallel
      printf("Step 1: Compile\n");
#pragma omp parallel for schedule(dynamic) num_threads(num_compile_threads)
      for (int i = start_subcase; i < end_subcase; ++i) {
        subcases[i].save_and_compile(trans_results[i].code);
      }
      // Run them one by one
      printf("Step 2: Run\n");
      for (int i = start_subcase; i < end_subcase; ++i) {
        printf("----------------\n");
        Subcase::RunResult result = subcases[i].run(trans_results[i]);
        run_results.push_back(result);
        subcases[i].remove_so_file();
      }
    }
    return run_results;
  }

  // APIs for the `logic()` function to call
  void add_subcase(Subcase &subcase) {
    subcases.emplace_back(subcase);
  }
};

vector<Testcase> all_testcases;

#define CONCAT_IMPL(x, y) x##y
#define CONCAT(x, y) CONCAT_IMPL(x, y)

#define ADD_TESTCASE_HELPER(testcase, counter)                                 \
  namespace {                                                                  \
  class CONCAT(Testcase, counter) {                                            \
  public:                                                                      \
    CONCAT(Testcase, counter)() { all_testcases.push_back(testcase); }         \
  } CONCAT(testcase_adder_, counter);                                          \
  }

#define ADD_TESTCASE(testcase) ADD_TESTCASE_HELPER(testcase, __COUNTER__)
