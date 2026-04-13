#include "mirage/search/profile.h"

#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"
#include "mirage/transpiler/error_types.h"
#include "mirage/transpiler/transpile.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace mirage {
namespace search {

// Helper function to get target compute capability
int get_target_cc() {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, 0);
  if (err != cudaSuccess) {
    return 80; // Default to A100
  }
  return prop.major * 10 + prop.minor;
}

// Helper function to create local directory for generated CUDA files
std::string create_local_cuda_dir() {
  std::string local_dir = "generated_cuda";

  // Create the directory if it doesn't exist
  if (!std::filesystem::exists(local_dir)) {
    std::filesystem::create_directories(local_dir);
  }

  return local_dir;
}

// Helper function to generate unique filename with timestamp
std::string generate_cuda_filename(std::string const &base_dir) {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;

  std::ostringstream oss;
  oss << base_dir << "/kernel_";
  oss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  oss << "_" << std::setfill('0') << std::setw(3) << ms.count();

  return oss.str();
}

static std::atomic<uint64_t> g_cuda_file_counter{0};
std::string generate_cuda_filename_unique(std::string const &base_dir) {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  uint64_t id = g_cuda_file_counter.fetch_add(1, std::memory_order_relaxed);
  std::ostringstream oss;
  oss << base_dir << "/kernel_";
  oss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  oss << "_" << std::setfill('0') << std::setw(3) << ms.count();
  oss << "_" << id;
  return oss.str();
}

static std::mutex g_profile_run_mutex;

// Helper function to find nvcc
std::string find_nvcc() {
  char const *nvcc_path = std::getenv("CUDA_HOME");
  if (nvcc_path != nullptr) {
    std::string path = std::string(nvcc_path) + "/bin/nvcc";
    if (std::filesystem::exists(path)) {
      return path;
    }
  }

  // Try common locations
  char const *paths[] = {"/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"};
  for (char const *path : paths) {
    if (std::filesystem::exists(path)) {
      return std::string(path);
    }
  }

  throw std::runtime_error("nvcc not found. Please ensure CUDA is installed.");
}

// Helper function to build compile command
std::string build_compile_cmd(int target_cc,
                              std::string const &nvcc_path,
                              std::string const &cu_file,
                              std::string const &so_file,
                              std::string const &include_path,
                              std::string const &deps_path,
                              bool profiling) {
  std::ostringstream cmd;
  cmd << nvcc_path << " " << cu_file << " -O3";

  // Add include paths
  cmd << " -I" << include_path << "/mirage/transpiler/runtime";
  cmd << " -I" << deps_path << "/cutlass/include";

  // Add architecture flags
  if (target_cc == 90) {
    cmd << " -arch=sm_90a -gencode=arch=compute_90a,code=sm_90a";
  } else if (target_cc == 100) {
    cmd << " -arch=sm_100a -gencode=arch=compute_100a,code=sm_100a";
  } else {
    cmd << " -arch=sm_80";
  }

  if (profiling) {
    cmd << " -DMIRAGE_ENABLE_PROFILER";
  }

  cmd << " -DMIRAGE_BACKEND_USE_CUDA -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL";
  cmd << " -shared -std=c++17 -use_fast_math -lcublas -Xcompiler=-fPIC";
  cmd << " --expt-relaxed-constexpr -o " << so_file;

  return cmd.str();
}

// Get MIRAGE paths
void get_mirage_paths(std::string &include_path, std::string &deps_path) {
  char const *mirage_root = std::getenv("MIRAGE_ROOT");
  std::string root;

  if (mirage_root != nullptr) {
    root = std::string(mirage_root);
  } else {
    // Try to infer from current location
    root = std::filesystem::current_path().string();
  }

  // Determine include path
  include_path = root + "/include";

  // Determine deps path - check if cutlass exists in deps
  std::string test_deps_path;
  if (std::filesystem::exists(root + "/deps")) {
    test_deps_path = root + "/deps";
  } else {
    test_deps_path = root + "/include/deps";
  }

  // Verify cutlass exists in the deps path, if not, try parent directory
  // (source directory)
  if (!std::filesystem::exists(test_deps_path +
                               "/cutlass/include/cute/layout.hpp")) {
    std::filesystem::path parent = std::filesystem::path(root).parent_path();
    std::string source_deps = parent.string() + "/deps";
    if (std::filesystem::exists(source_deps +
                                "/cutlass/include/cute/layout.hpp")) {
      deps_path = source_deps;
    } else {
      // Fallback to original logic
      deps_path = test_deps_path;
    }
  } else {
    deps_path = test_deps_path;
  }
}

// ============================================================
// Persistent compile + run cache
// ============================================================

namespace {

struct CachedCompileEntry {
  bool is_success;
  std::string error_message;
  std::string so_filename; // basename only, lives inside cache_dir()
  std::vector<size_t> input_num_elements;
  std::vector<size_t> output_alloc_sizes;
  size_t buf_size;
  size_t profiler_buf_size;
  std::string cuda_code;
};

struct CachedRunEntry {
  bool is_success;
  float run_time;
  std::string error_message;
};

static std::mutex g_cache_mutex;
static std::unordered_map<size_t, CachedCompileEntry> g_compile_cache;
static std::unordered_map<size_t, CachedRunEntry> g_run_cache;
static bool g_cache_loaded = false;

static std::string cache_dir() {
  return "generated_cuda/cache";
}
static std::string cache_file() {
  return cache_dir() + "/profile_cache.json";
}

static std::string key_to_hex(size_t k) {
  std::ostringstream o;
  o << std::hex << std::setw(16) << std::setfill('0') << k;
  return o.str();
}

static size_t compile_cache_key(kernel::Graph const *g, int target_cc) {
  size_t h = g->get_owner_independent_hash();
  h ^= std::hash<int>{}(target_cc) + 0x9e3779b9 + (h << 6) + (h >> 2);
  return h;
}

static size_t run_cache_key(std::string const &cuda_code) {
  return std::hash<std::string>{}(cuda_code);
}

// Must be called with g_cache_mutex held.
static void load_cache_locked() {
  if (g_cache_loaded) {
    return;
  }
  g_cache_loaded = true;

  if (!std::filesystem::exists(cache_file())) {
    return;
  }

  try {
    std::ifstream f(cache_file());
    if (!f.is_open()) {
      return;
    }
    nlohmann::json j;
    f >> j;

    if (j.contains("compile_entries")) {
      for (auto const &[hkey, val] : j["compile_entries"].items()) {
        size_t key = std::stoull(hkey, nullptr, 16);
        CachedCompileEntry ce;
        ce.is_success = val["is_success"].get<bool>();
        ce.error_message = val["error_message"].get<std::string>();
        ce.so_filename = val["so_filename"].get<std::string>();
        ce.input_num_elements =
            val["input_num_elements"].get<std::vector<size_t>>();
        ce.output_alloc_sizes =
            val["output_alloc_sizes"].get<std::vector<size_t>>();
        ce.buf_size = val["buf_size"].get<size_t>();
        ce.profiler_buf_size = val["profiler_buf_size"].get<size_t>();
        ce.cuda_code = val["cuda_code"].get<std::string>();
        g_compile_cache[key] = std::move(ce);
      }
    }

    if (j.contains("run_entries")) {
      for (auto const &[hkey, val] : j["run_entries"].items()) {
        size_t key = std::stoull(hkey, nullptr, 16);
        CachedRunEntry re;
        re.is_success = val["is_success"].get<bool>();
        re.run_time = val["run_time"].get<float>();
        re.error_message = val["error_message"].get<std::string>();
        g_run_cache[key] = std::move(re);
      }
    }
  } catch (...) {
    // Corrupt file — start fresh
    g_compile_cache.clear();
    g_run_cache.clear();
  }
}

// Must be called with g_cache_mutex held.
static void save_cache_locked() {
  nlohmann::json j;

  for (auto const &[key, ce] : g_compile_cache) {
    std::string hkey = key_to_hex(key);
    j["compile_entries"][hkey]["is_success"] = ce.is_success;
    j["compile_entries"][hkey]["error_message"] = ce.error_message;
    j["compile_entries"][hkey]["so_filename"] = ce.so_filename;
    j["compile_entries"][hkey]["input_num_elements"] = ce.input_num_elements;
    j["compile_entries"][hkey]["output_alloc_sizes"] = ce.output_alloc_sizes;
    j["compile_entries"][hkey]["buf_size"] = ce.buf_size;
    j["compile_entries"][hkey]["profiler_buf_size"] = ce.profiler_buf_size;
    j["compile_entries"][hkey]["cuda_code"] = ce.cuda_code;
  }

  for (auto const &[key, re] : g_run_cache) {
    std::string hkey = key_to_hex(key);
    j["run_entries"][hkey]["is_success"] = re.is_success;
    j["run_entries"][hkey]["run_time"] = re.run_time;
    j["run_entries"][hkey]["error_message"] = re.error_message;
  }

  std::filesystem::create_directories(cache_dir());
  std::string tmp = cache_file() + ".tmp";
  std::ofstream f(tmp);
  if (f.is_open()) {
    f << j.dump(2);
    f.close();
    std::error_code ec;
    std::filesystem::rename(tmp, cache_file(), ec);
  }
}

} // anonymous namespace

// ============================================================

ProfileCompileResult profile_compile(kernel::Graph *graph) {
  ProfileCompileResult compiled;
  compiled.is_success = false;
  compiled.error_message = "";
  compiled.cuda_code = "";

  try {
    if (graph == nullptr) {
      compiled.error_message = "Invalid graph";
      return compiled;
    }

    kernel::DTensor *input_dtensors[1024];
    int num_inputs = graph->get_input_dtensors(input_dtensors);

    std::vector<std::vector<size_t>> input_strides;
    for (kernel::KNOperator *graph_op : graph->operators) {
      if (graph_op->op_type == mirage::type::KN_INPUT_OP) {
        kernel::KNInputOp *input_op =
            static_cast<kernel::KNInputOp *>(graph_op);
        input_strides.push_back(input_op->input_strides);
      }
    }

    int target_cc = get_target_cc();
    size_t ck = compile_cache_key(graph, target_cc);

    // Cache lookup
    {
      std::lock_guard<std::mutex> lk(g_cache_mutex);
      load_cache_locked();
      auto it = g_compile_cache.find(ck);
      if (it != g_compile_cache.end()) {
        auto const &ce = it->second;
        if (!ce.is_success) {
          compiled.error_message = ce.error_message;
          return compiled; // cached failure
        }
        std::string so_path = cache_dir() + "/" + ce.so_filename;
        if (std::filesystem::exists(so_path)) {
          compiled.is_success = true;
          compiled.so_file = so_path;
          compiled.input_num_elements = ce.input_num_elements;
          compiled.output_alloc_sizes = ce.output_alloc_sizes;
          compiled.buf_size = ce.buf_size;
          compiled.profiler_buf_size = ce.profiler_buf_size;
          compiled.cuda_code = ce.cuda_code;
          return compiled; // cache hit
        }
        // .so was deleted — fall through and recompile
      }
    }

    transpiler::TranspilerConfig config;
    config.target_cc = target_cc;
    config.profiling = false;
    config.enable_online_softmax = false;
    config.num_producer_wgs = 1;
    config.num_consumer_wgs = 1;
    config.pipeline_stages = 2;

    transpiler::TranspileResult transpile_result =
        transpiler::transpile(graph, config, input_strides);

    if (transpile_result.error_type != transpiler::CUDA_T_SUCCESS) {
      compiled.error_message = "Failed to transpile graph";
      return compiled;
    }

    std::string include_path, deps_path;
    get_mirage_paths(include_path, deps_path);
    std::string local_dir = create_local_cuda_dir();
    std::string base_filename = generate_cuda_filename_unique(local_dir);
    std::string cu_file = base_filename + ".cu";
    std::string so_file = base_filename + ".so";

    std::ofstream out_file(cu_file);
    if (!out_file.is_open()) {
      compiled.error_message = "Failed to create CUDA file: " + cu_file;
      return compiled;
    }
    out_file << transpile_result.code;
    out_file.close();
    compiled.cuda_code = transpile_result.code;

    std::string nvcc_path = find_nvcc();
    std::string compile_cmd = build_compile_cmd(
        target_cc, nvcc_path, cu_file, so_file, include_path, deps_path, false);
    std::string error_output_file = base_filename + "_compile_error.txt";
    std::string full_compile_cmd =
        compile_cmd + " > " + error_output_file + " 2>&1";
    int compile_status = std::system(full_compile_cmd.c_str());
    if (compile_status != 0) {
      compiled.error_message = "CUDA compilation failed";
      std::string compile_error;
      std::ifstream error_file(error_output_file);
      if (error_file.is_open()) {
        std::ostringstream error_stream;
        error_stream << error_file.rdbuf();
        compile_error = error_stream.str();
        error_file.close();
      }
      std::ofstream log_file("profile_log.txt");
      log_file << "=== CUDA Source Code ===\n";
      log_file << transpile_result.code;
      log_file << "\n\n=== Compilation Command ===\n";
      log_file << compile_cmd;
      log_file << "\n\n=== NVCC Compilation Error ===\n";
      if (!compile_error.empty()) {
        log_file << compile_error;
      } else {
        log_file << "No error output captured (compilation failed with status "
                 << compile_status << ")";
      }
      log_file.close();
      return compiled;
    }

    compiled.so_file = so_file;
    compiled.input_num_elements.clear();
    for (int i = 0; i < num_inputs; i++) {
      size_t num_elements = 1;
      for (int j = 0; j < input_dtensors[i]->num_dims; j++) {
        num_elements *= input_dtensors[i]->dim[j];
      }
      compiled.input_num_elements.push_back(num_elements);
    }
    compiled.output_alloc_sizes.clear();
    for (size_t i = 0; i < transpile_result.output_directives.size(); i++) {
      compiled.output_alloc_sizes.push_back(
          transpile_result.output_directives[i].alloc_size);
    }
    compiled.buf_size = transpile_result.buf_size;
    compiled.profiler_buf_size = transpile_result.profiler_buf_size;
    compiled.is_success = true;

    // Cache save (success)
    {
      std::lock_guard<std::mutex> lk(g_cache_mutex);
      std::filesystem::create_directories(cache_dir());
      std::string so_base = "compiled_" + key_to_hex(ck) + ".so";
      std::string so_dest = cache_dir() + "/" + so_base;
      std::error_code ec;
      std::filesystem::copy_file(
          compiled.so_file,
          so_dest,
          std::filesystem::copy_options::overwrite_existing,
          ec);
      CachedCompileEntry ce;
      ce.is_success = true;
      ce.error_message = "";
      ce.so_filename = ec ? std::string{} : so_base;
      ce.input_num_elements = compiled.input_num_elements;
      ce.output_alloc_sizes = compiled.output_alloc_sizes;
      ce.buf_size = compiled.buf_size;
      ce.profiler_buf_size = compiled.profiler_buf_size;
      ce.cuda_code = compiled.cuda_code;
      g_compile_cache[ck] = std::move(ce);
      save_cache_locked();
    }

  } catch (std::exception const &e) {
    compiled.error_message = std::string("Exception: ") + e.what();
  } catch (...) {
    compiled.error_message = "Unknown error occurred";
  }
  return compiled;
}

ProfileResult profile_run(ProfileCompileResult const &compiled) {
  ProfileResult result;
  result.is_success = false;
  result.run_time = std::numeric_limits<float>::max();
  result.error_message = "";
  result.cuda_code = compiled.cuda_code;

  if (!compiled.is_success) {
    result.error_message = compiled.error_message.empty()
                               ? "Compile failed"
                               : compiled.error_message;
    return result;
  }

  // Run cache lookup
  size_t rk = run_cache_key(compiled.cuda_code);
  {
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    load_cache_locked();
    auto it = g_run_cache.find(rk);
    if (it != g_run_cache.end()) {
      ProfileResult r;
      r.is_success = it->second.is_success;
      r.run_time = it->second.run_time;
      r.error_message = it->second.error_message;
      r.cuda_code = compiled.cuda_code;
      return r;
    }
  }

  std::lock_guard<std::mutex> lock(g_profile_run_mutex);

  try {
    void *handle = dlopen(compiled.so_file.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
      result.error_message =
          "Failed to load compiled library: " + std::string(dlerror());
      return result;
    }

    using ExecuteFunc = int (*)(std::vector<void const *>,
                                std::vector<void *>,
                                void *,
                                cudaStream_t,
                                void *);
    ExecuteFunc execute_func = (ExecuteFunc)dlsym(handle, "execute_mugraph");
    if (execute_func == nullptr) {
      result.error_message =
          "Failed to find execute_mugraph: " + std::string(dlerror());
      dlclose(handle);
      return result;
    }

    std::vector<void *> input_ptrs;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < compiled.input_num_elements.size(); i++) {
      size_t num_elements = compiled.input_num_elements[i];
      size_t size_bytes = num_elements * sizeof(__half);
      void *ptr;
      cudaMalloc(&ptr, size_bytes);
      std::vector<__half> host_data(num_elements);
      for (size_t j = 0; j < num_elements; j++) {
        host_data[j] = __float2half(dis(gen));
      }
      cudaMemcpy(ptr, host_data.data(), size_bytes, cudaMemcpyHostToDevice);
      input_ptrs.push_back(ptr);
    }

    std::vector<void *> output_ptrs;
    for (size_t i = 0; i < compiled.output_alloc_sizes.size(); i++) {
      void *ptr;
      cudaMalloc(&ptr, compiled.output_alloc_sizes[i] * sizeof(__half));
      output_ptrs.push_back(ptr);
    }

    void *buf_ptr;
    cudaMalloc(&buf_ptr, compiled.buf_size);
    void *profiler_buffer = nullptr;
    if (compiled.profiler_buf_size > 0) {
      cudaMalloc(&profiler_buffer, compiled.profiler_buf_size);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<void const *> const_input_ptrs(input_ptrs.begin(),
                                               input_ptrs.end());

    int err = execute_func(
        const_input_ptrs, output_ptrs, buf_ptr, stream, profiler_buffer);
    if (err != (int)cudaSuccess) {
      result.error_message = "Kernel launch failed: " +
                             std::string(cudaGetErrorString((cudaError_t)err));
      cudaStreamDestroy(stream);
      cudaFree(buf_ptr);
      if (profiler_buffer != nullptr) {
        cudaFree(profiler_buffer);
      }
      for (void *ptr : output_ptrs) {
        cudaFree(ptr);
      }
      for (void *ptr : input_ptrs) {
        cudaFree(ptr);
      }
      dlclose(handle);
      return result;
    }

    int const warmup_iters = 16;
    for (int i = 0; i < warmup_iters; i++) {
      execute_func(
          const_input_ptrs, output_ptrs, buf_ptr, stream, profiler_buffer);
    }
    cudaStreamSynchronize(stream);

    int const profile_iters = 100;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float elapsed_ms = 0.0f;
    cudaError_t ev_err = cudaEventCreate(&start_event);
    if (ev_err != cudaSuccess) {
      result.error_message = "cudaEventCreate(start) failed: " +
                             std::string(cudaGetErrorString(ev_err));
      goto run_cleanup;
    }
    ev_err = cudaEventCreate(&stop_event);
    if (ev_err != cudaSuccess) {
      result.error_message = "cudaEventCreate(stop) failed: " +
                             std::string(cudaGetErrorString(ev_err));
      cudaEventDestroy(start_event);
      goto run_cleanup;
    }
    ev_err = cudaEventRecord(start_event, stream);
    if (ev_err != cudaSuccess) {
      result.error_message = "cudaEventRecord(start) failed: " +
                             std::string(cudaGetErrorString(ev_err));
      cudaEventDestroy(stop_event);
      cudaEventDestroy(start_event);
      goto run_cleanup;
    }
    for (int i = 0; i < profile_iters; i++) {
      execute_func(
          const_input_ptrs, output_ptrs, buf_ptr, stream, profiler_buffer);
    }
    ev_err = cudaEventRecord(stop_event, stream);
    if (ev_err != cudaSuccess) {
      result.error_message = "cudaEventRecord(stop) failed: " +
                             std::string(cudaGetErrorString(ev_err));
      cudaEventDestroy(stop_event);
      cudaEventDestroy(start_event);
      goto run_cleanup;
    }
    ev_err = cudaEventSynchronize(stop_event);
    if (ev_err != cudaSuccess) {
      result.error_message = "cudaEventSynchronize(stop) failed: " +
                             std::string(cudaGetErrorString(ev_err));
      cudaEventDestroy(stop_event);
      cudaEventDestroy(start_event);
      goto run_cleanup;
    }
    ev_err = cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    if (ev_err != cudaSuccess) {
      result.error_message = "cudaEventElapsedTime failed: " +
                             std::string(cudaGetErrorString(ev_err));
      goto run_cleanup;
    }
    result.run_time = elapsed_ms / profile_iters;
    result.is_success = true;
    result.error_message = "Success";

  run_cleanup:
    cudaStreamDestroy(stream);
    cudaFree(buf_ptr);
    if (profiler_buffer != nullptr) {
      cudaFree(profiler_buffer);
    }
    for (void *ptr : output_ptrs) {
      cudaFree(ptr);
    }
    for (void *ptr : input_ptrs) {
      cudaFree(ptr);
    }
    dlclose(handle);

    // Cache save (after cleanup, result is fully populated)
    {
      std::lock_guard<std::mutex> lk(g_cache_mutex);
      CachedRunEntry re;
      re.is_success = result.is_success;
      re.run_time = result.run_time;
      re.error_message = result.error_message;
      g_run_cache[rk] = re;
      save_cache_locked();
    }

  } catch (std::exception const &e) {
    result.error_message = std::string("Exception: ") + e.what();
  } catch (...) {
    result.error_message = "Unknown error occurred";
  }
  return result;
}

ProfileResult profile(kernel::Graph *graph) {
  ProfileResult result;
  result.is_success = false;
  result.run_time = std::numeric_limits<float>::max();
  result.error_message = "";
  result.cuda_code = "";

  try {
    ProfileCompileResult compiled = profile_compile(graph);
    if (!compiled.is_success) {
      result.error_message = compiled.error_message;
      result.cuda_code = compiled.cuda_code;
      return result;
    }
    result = profile_run(compiled);
  } catch (std::exception const &e) {
    result.error_message = std::string("Exception: ") + e.what();
  } catch (...) {
    result.error_message = "Unknown error occurred";
  }
  return result;
}

} // namespace search
} // namespace mirage
