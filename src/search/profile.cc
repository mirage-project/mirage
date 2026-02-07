#include "mirage/search/profile.h"

#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"
#include "mirage/transpiler/transpile.h"
#include "mirage/transpiler/error_types.h"
#include "mirage/kernel/device_tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unistd.h>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <random>
#include <chrono>
#include <iomanip>

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
std::string generate_cuda_filename(const std::string& base_dir) {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;
  
  std::ostringstream oss;
  oss << base_dir << "/kernel_";
  oss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
  oss << "_" << std::setfill('0') << std::setw(3) << ms.count();
  
  return oss.str();
}

// Helper function to find nvcc
std::string find_nvcc() {
  const char* nvcc_path = std::getenv("CUDA_HOME");
  if (nvcc_path != nullptr) {
    std::string path = std::string(nvcc_path) + "/bin/nvcc";
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  
  // Try common locations
  const char* paths[] = {"/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"};
  for (const char* path : paths) {
    if (std::filesystem::exists(path)) {
      return std::string(path);
    }
  }
  
  throw std::runtime_error("nvcc not found. Please ensure CUDA is installed.");
}

// Helper function to build compile command
std::string build_compile_cmd(int target_cc, const std::string& nvcc_path,
                               const std::string& cu_file, const std::string& so_file,
                               const std::string& include_path, const std::string& deps_path,
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
    cmd << " -arch=native";
  }
  
  if (profiling) {
    cmd << " -DMIRAGE_ENABLE_PROFILER";
  }
  
  cmd << " -DMIRAGE_BACKEND_USE_CUDA";
  cmd << " -shared -std=c++17 -use_fast_math -lcublas -Xcompiler=-fPIC";
  cmd << " --expt-relaxed-constexpr -o " << so_file;
  
  return cmd.str();
}

// Get MIRAGE paths
void get_mirage_paths(std::string& include_path, std::string& deps_path) {
  const char* mirage_root = std::getenv("MIRAGE_ROOT");
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
  
  // Verify cutlass exists in the deps path, if not, try parent directory (source directory)
  if (!std::filesystem::exists(test_deps_path + "/cutlass/include/cute/layout.hpp")) {
    std::filesystem::path parent = std::filesystem::path(root).parent_path();
    std::string source_deps = parent.string() + "/deps";
    if (std::filesystem::exists(source_deps + "/cutlass/include/cute/layout.hpp")) {
      deps_path = source_deps;
    } else {
      // Fallback to original logic
      deps_path = test_deps_path;
    }
  } else {
    deps_path = test_deps_path;
  }
}

ProfileResult profile(kernel::Graph *graph) {
  ProfileResult result;
  result.is_success = false;
  result.run_time = std::numeric_limits<float>::max();
  result.error_message = "";
  result.cuda_code = "";

  try {
    // Validate graph
    if (graph == nullptr) {
      result.error_message = "Invalid graph";
      return result;
    }
    
    // Get input tensors and their strides
    kernel::DTensor* input_dtensors[1024];
    int num_inputs = graph->get_input_dtensors(input_dtensors);
    
    std::vector<std::vector<size_t>> input_strides;
    // Get strides from input operators
    for (kernel::KNOperator* graph_op : graph->operators) {
      if (graph_op->op_type == mirage::type::KN_INPUT_OP) {
        kernel::KNInputOp* input_op = static_cast<kernel::KNInputOp*>(graph_op);
        input_strides.push_back(input_op->input_strides);
      }
    }
    
    // Get target compute capability
    int target_cc = get_target_cc();
    
    // Configure transpiler
    transpiler::TranspilerConfig config;
    config.target_cc = target_cc;
    config.profiling = false; // Profile without profiler overhead
    config.enable_online_softmax = false;
    config.num_producer_wgs = 1;
    config.num_consumer_wgs = 1;
    config.pipeline_stages = 2;
    
    // Generate CUDA code
    transpiler::TranspileResult transpile_result = 
        transpiler::transpile(graph, config, input_strides);
    
    if (transpile_result.error_type != transpiler::CUDA_T_SUCCESS) {
      result.error_message = "Failed to transpile graph";
      return result;
    }
    
    // Get MIRAGE paths
    std::string include_path, deps_path;
    get_mirage_paths(include_path, deps_path);
    
    // Create local directory for generated CUDA files
    std::string local_dir = create_local_cuda_dir();
    std::string base_filename = generate_cuda_filename(local_dir);
    std::string cu_file = base_filename + ".cu";
    std::string so_file = base_filename + ".so";
    
    // Write CUDA code to file
    // Note: The transpiled code already includes runtime, so we just write it
    std::ofstream out_file(cu_file);
    if (!out_file.is_open()) {
      result.error_message = "Failed to create CUDA file: " + cu_file;
      return result;
    }
    out_file << transpile_result.code;
    out_file.close();

    result.cuda_code = transpile_result.code;
    
    // Compile the code
    std::string nvcc_path = find_nvcc();
    std::string compile_cmd = build_compile_cmd(target_cc, nvcc_path, cu_file, so_file,
                                                 include_path, deps_path, false);
    
    // Capture compilation errors
    std::string error_output_file = base_filename + "_compile_error.txt";
    std::string full_compile_cmd = compile_cmd + " > " + error_output_file + " 2>&1";
    int compile_status = std::system(full_compile_cmd.c_str());
    if (compile_status != 0) {
      result.error_message = "CUDA compilation failed";
      
      // Read the compilation error message
      std::string compile_error;
      std::ifstream error_file(error_output_file);
      if (error_file.is_open()) {
        std::ostringstream error_stream;
        error_stream << error_file.rdbuf();
        compile_error = error_stream.str();
        error_file.close();
      }
      
      // Save CUDA source code and compilation error to log file
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
      
      // Note: CUDA files are kept in local_dir for inspection
      return result;
    }
    
    // Load the shared library
    void* handle = dlopen(so_file.c_str(), RTLD_LAZY);
    if (handle == nullptr) {
      result.error_message = "Failed to load compiled library: " + std::string(dlerror());
      // Note: CUDA files are kept in local_dir for inspection
      return result;
    }
    
    // Get the execute function
    using ExecuteFunc = int(*)(std::vector<void const*>, std::vector<void*>, void*, cudaStream_t, void*);
    ExecuteFunc execute_func = (ExecuteFunc)dlsym(handle, "execute_mugraph");
    if (execute_func == nullptr) {
      result.error_message = "Failed to find execute_mugraph: " + std::string(dlerror());
      dlclose(handle);
      // Note: CUDA files are kept in local_dir for inspection
      return result;
    }
    
    // Allocate input tensors
    std::vector<void*> input_ptrs;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < num_inputs; i++) {
      size_t num_elements = 1;
      for (int j = 0; j < input_dtensors[i]->num_dims; j++) {
        num_elements *= input_dtensors[i]->dim[j];
      }
      size_t size_bytes = num_elements * sizeof(__half);
      
      void* ptr;
      cudaMalloc(&ptr, size_bytes);
      
      // Initialize with random data
      std::vector<__half> host_data(num_elements);
      for (size_t j = 0; j < num_elements; j++) {
        host_data[j] = __float2half(dis(gen));
      }
      cudaMemcpy(ptr, host_data.data(), size_bytes, cudaMemcpyHostToDevice);
      
      input_ptrs.push_back(ptr);
    }
    
    // Allocate output tensors
    std::vector<void*> output_ptrs;
    for (size_t i = 0; i < transpile_result.output_directives.size(); i++) {
      size_t alloc_size = transpile_result.output_directives[i].alloc_size;
      void* ptr;
      cudaMalloc(&ptr, alloc_size * sizeof(__half));
      output_ptrs.push_back(ptr);
    }
    
    // Allocate buffer
    void* buf_ptr;
    cudaMalloc(&buf_ptr, transpile_result.buf_size);
    
    // Allocate profiler buffer (not used but required by signature)
    void* profiler_buffer = nullptr;
    if (transpile_result.profiler_buf_size > 0) {
      cudaMalloc(&profiler_buffer, transpile_result.profiler_buf_size);
    }
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Convert input pointers to const void*
    std::vector<void const*> const_input_ptrs(input_ptrs.begin(), input_ptrs.end());
    
    // Check if the kernel can be successfully executed
    int err = execute_func(const_input_ptrs, output_ptrs, buf_ptr, stream, profiler_buffer);
    if (err != (int)cudaSuccess) {
      result.error_message = "Kernel launch failed: " + std::string(cudaGetErrorString((cudaError_t)err));
      cudaStreamDestroy(stream);
      cudaFree(buf_ptr);
      if (profiler_buffer != nullptr) cudaFree(profiler_buffer);
      for (void* ptr : output_ptrs) cudaFree(ptr);
      for (void* ptr : input_ptrs) cudaFree(ptr);
      dlclose(handle);
      return result;
    }

    // Warmup runs
    const int warmup_iters = 16;
    for (int i = 0; i < warmup_iters; i++) {
      execute_func(const_input_ptrs, output_ptrs, buf_ptr, stream, profiler_buffer);
    }
    cudaStreamSynchronize(stream);
    
    // Profile runs
    const int profile_iters = 100;
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    cudaEventRecord(start_event, stream);
    for (int i = 0; i < profile_iters; i++) {
      execute_func(const_input_ptrs, output_ptrs, buf_ptr, stream, profiler_buffer);
    }
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start_event, stop_event);
    result.run_time = elapsed_ms / profile_iters;
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(buf_ptr);
    if (profiler_buffer != nullptr) {
      cudaFree(profiler_buffer);
    }
    for (void* ptr : output_ptrs) {
      cudaFree(ptr);
    }
    for (void* ptr : input_ptrs) {
      cudaFree(ptr);
    }
    
    dlclose(handle);
    // Note: CUDA source files (.cu) and compiled libraries (.so) are kept in 
    // local_dir (generated_cuda/) for inspection. They are not automatically cleaned up.
    
    result.is_success = true;
    result.error_message = "Success";
    
  } catch (const std::exception& e) {
    result.error_message = std::string("Exception: ") + e.what();
  } catch (...) {
    result.error_message = "Unknown error occurred";
  }
  
  return result;
}

} // namespace search
} // namespace mirage