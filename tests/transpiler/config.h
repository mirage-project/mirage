#pragma once

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <omp.h>

namespace env_config {

// Create a directory if it does not exist
inline void make_dir(std::string const &dir) {
  int ret_code = system(("mkdir -p " + dir).c_str());
  if (ret_code != 0) {
    throw std::runtime_error("Failed to create directory: " + dir);
  }
}

inline std::string get_generated_cu_dir() {
  char const *config_from_env_str = getenv("TRANSPILER_TEST_GENERATED_CU_DIR");
  if (config_from_env_str != nullptr) {
    std::string config_from_env = config_from_env_str;
    make_dir(config_from_env);
    return config_from_env.back() == '/'
               ? config_from_env.substr(0, config_from_env.size() - 1)
               : config_from_env;
  }
  int uid = getuid();
  std::string answer =
      "/tmp/mirage-transpiler-test-" + std::to_string(uid) + "/generated-cu";
  make_dir(answer);
  return answer;
}

inline std::string get_generated_so_dir() {
  char const *config_from_env_str = getenv("TRANSPILER_TEST_GENERATED_SO_DIR");
  if (config_from_env_str != nullptr) {
    std::string config_from_env = config_from_env_str;
    make_dir(config_from_env);
    return config_from_env.back() == '/'
               ? config_from_env.substr(0, config_from_env.size() - 1)
               : config_from_env;
  }
  int uid = getuid();
  std::string answer =
      "/tmp/mirage-transpiler-test-" + std::to_string(uid) + "/generated-so";
  make_dir(answer);
  return answer;
}

inline int get_num_compile_threads() {
  char const *config_from_env_str =
      getenv("TRANSPILER_TEST_NUM_COMPILE_THREADS");
  if (config_from_env_str != nullptr) {
    return std::stoi(config_from_env_str);
  }
  return omp_get_max_threads();
}

inline std::string get_cutlass_root() {
  std::string result;
  char const *config_from_env_str = getenv("CUTLASS_ROOT");
  if (config_from_env_str != nullptr) {
    return config_from_env_str;
  } else {
    result =
        std::filesystem::relative(
            std::filesystem::canonical("/proc/self/exe").remove_filename() /
            ".." / ".." / ".." / "deps" / "cutlass")
            .string();
  }
  if (!std::filesystem::exists(result)) {
    printf("CUTLASS_ROOT (%s) not found. Please properly set the CUTLASS_ROOT "
           "environment variable.\n",
           result.c_str());
    throw std::runtime_error("CUTLASS_ROOT not found");
  }
  if (result.back() == '/') {
    result.pop_back();
  }
  return result;
}

inline std::string get_mirage_runtime_root() {
  std::string result;
  char const *config_from_env_str = getenv("MIRAGE_RUNTIME_ROOT");
  if (config_from_env_str != nullptr) {
    return config_from_env_str;
  } else {
    result =
        std::filesystem::relative(
            std::filesystem::canonical("/proc/self/exe").remove_filename() /
            ".." / ".." / ".." / "include" / "mirage" / "transpiler" /
            "runtime")
            .string();
  }
  if (!std::filesystem::exists(result)) {
    printf("MIRAGE_RUNTIME_ROOT (%s) not found. Please properly set the "
           "MIRAGE_RUNTIME_ROOT environment variable.\n",
           result.c_str());
    throw std::runtime_error("MIRAGE_RUNTIME_ROOT not found");
  }
  if (result.back() == '/') {
    result.pop_back();
  }
  return result;
}

} // namespace env_config
