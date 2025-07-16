cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(MIRAGE LANGUAGES C CXX CUDA)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

if (EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include (${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}) # to include protobuf headeer files
include_directories("include")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

#add_subdirectory(deps/cutlass)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
#set(CUTLASS_NVCC_ARCHS "80;86")

file(GLOB_RECURSE MIRAGE_SRCS
  src/*.cc
)

file(GLOB_RECURSE MIRAGE_CUDA_SRCS
  src/*.cu
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

#set CUDA version
include(deps/cutlass/CUDA.cmake)
string(REPLACE "." ";" CUDA_VERSION_PARTS ${CUDA_VERSION})
list(GET CUDA_VERSION_PARTS 0 CUDA_VERSION_MAJOR)
list(GET CUDA_VERSION_PARTS 1 CUDA_VERSION_MINOR)
list(APPEND CMAKE_CUDA_FLAGS "-D__CUDACC_VER_MAJOR__=${CUDA_VERSION_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_VERSION_MINOR}")

# TODO: Currently disable CUTLASS_ARCH_MMA_SM80_ENABLED flag
# since we target Triton to perform codegen
#list(APPEND CMAKE_CUDA_FLAGS "-DCUTLASS_ARCH_MMA_SM80_ENABLED")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fopenmp -O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fopenmp -O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

#set CUDA
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(cmake/cuda.cmake)
  find_cuda(${USE_CUDA})
  if (CUDA_FOUND)
    list(APPEND MIRAGE_SRCS ${MIRAGE_CUDA_SRCS})
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    list(APPEND MIRAGE_LINK_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND MIRAGE_LINK_LIBS ${CUDA_CUDA_LIBRARY})
  else()
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif(CUDA_FOUND)
endif()

#set CUTLASS
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/deps/cutlass)
include_directories(deps/cutlass/include)
include_directories(deps/cutlass/tools/util/include)

#include_directories(${MIRAGE_INCLUDE_DIRS})

#set Z3
if(Z3_CXX_INCLUDE_DIRS AND Z3_LIBRARIES)
  message(STATUS "Z3_CXX_INCLUDE_DIRS: ${Z3_CXX_INCLUDE_DIRS}")
  message(STATUS "Z3_LIBRARIES: ${Z3_LIBRARIES}")
else()
  find_package(Z3 REQUIRED)
  message(STATUS "Z3_FOUND: ${Z3_FOUND}")
  message(STATUS "Found Z3 ${Z3_VERSION_STRING}")
  message(STATUS "Z3_DIR: ${Z3_DIR}")
endif()
include_directories(${Z3_CXX_INCLUDE_DIRS})
#target_link_libraries(mirage_runtime PUBLIC ${Z3_LIBRARIES})
list(APPEND MIRAGE_LINK_LIBS ${Z3_LIBRARIES})

add_subdirectory(deps/json)
list(APPEND MIRAGE_LINK_LIBS nlohmann_json::nlohmann_json)

#add_subdirectory(deps/z3 z3 EXCLUDE_FROM_ALL)
#include_directories(deps/z3/src/api/c++)
#include_directories(deps/z3/src/api)
#list(APPEND MIRAGE_LINK_LIBS z3)

add_library(mirage_runtime ${MIRAGE_SRCS})

# Note(zhihao): CUDA_SEPARABLE_COMPILATION is non-compatible with
# cython's installation, since enabling separable compilation
# will let cmake generate a seperate object file called
# cmake_device_link.o, and cython's installation cannot
# automatically link to that file, resulting in:
# undefined symbol: __cudaRegisterLinkedBinary_...
#set_target_properties(mirage_runtime
#  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Ensure Rust is installed
execute_process(COMMAND rustc --version
  RESULT_VARIABLE RUST_COMMAND_RESULT
  OUTPUT_VARIABLE RUSTC_OUTPUT
  ERROR_QUIET)

if(NOT RUST_COMMAND_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Rust is not installed on the system. Please install it by running: \n"
    "'curl https://sh.rustup.rs -sSf | sh -s -- -y' \n"
    "and follow the instructions on the screen.")
endif()

# Ensure Cargo is installed
execute_process(COMMAND cargo --version
  RESULT_VARIABLE CARGO_RESULT
  OUTPUT_QUIET ERROR_QUIET)

if(NOT CARGO_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Rust is installed, but cargo is not. Please install it by running: \n"
    "'curl https://sh.rustup.rs -sSf | sh -s -- -y' \n"
    "and follow the instructions on the screen.")
endif()

#set RUST
if(ABSTRACT_SUBEXPR_LIB AND ABSTRACT_SUBEXPR_LIBRARIES)
  message(STATUS "ABSTRACT_SUBEXPR_LIB: ${ABSTRACT_SUBEXPR_LIB}")
  message(STATUS "ABSTRACT_SUBEXPR_LIBRARIES: ${ABSTRACT_SUBEXPR_LIBRARIES}")
else()
  message(STATUS "ABSTRACT_SUBEXPR_LIB not set, will build it")
endif()
include_directories(${ABSTRACT_SUBEXPR_LIB})
list(APPEND MIRAGE_LINK_LIBS ${ABSTRACT_SUBEXPR_LIBRARIES})

if(FORMAL_VERIFIER_LIB AND FORMAL_VERIFIER_LIBRARIES)
  message(STATUS "FORMAL_VERIFIER_LIB: ${FORMAL_VERIFIER_LIB}")
  message(STATUS "FORMAL_VERIFIER_LIBRARIES: ${FORMAL_VERIFIER_LIBRARIES}")
else()
  message(STATUS "FORMAL_VERIFIER_LIB not set, will build it")
endif()
include_directories(${FORMAL_VERIFIER_LIB})
list(APPEND MIRAGE_LINK_LIBS ${FORMAL_VERIFIER_LIBRARIES})

set_target_properties(mirage_runtime PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")

target_compile_features(mirage_runtime PUBLIC cxx_std_17)

target_link_libraries(mirage_runtime ${MIRAGE_LINK_LIBS})

target_include_directories(mirage_runtime
  PUBLIC ${PROJECT_SOURCE_DIR}/include)

install(TARGETS mirage_runtime
    LIBRARY DESTINATION lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)

install(FILES ${ABSTRACT_SUBEXPR_LIBRARIES} 
    DESTINATION lib)

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN dnn)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Multi-Query Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_query_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Multi-Query Attention Speculative Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_query_attn_spec_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Multi-Query Attention Prefilling
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_query_attn_prefill)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Group-Query Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN group_query_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Group-Query Attention Spec Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN group_query_attn_spec_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Group-Query Attention Prefilling
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN group_query_attn_prefill)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Multi-Head Attention Incremental Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_head_attn_inc_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Multi-Head Attention Speculative Decoding
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_head_attn_spec_decode)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Multi-Head Attention Prefilling
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN multi_head_attn_prefill)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# LoRA
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN lora)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# MLP
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN mlp)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# MoE
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN moe)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# RMS
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN rms)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# GatedMLP
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN gated_mlp)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# Chameleon
if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN chameleon)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

# egg_tests
if("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN egg_tests)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

if ("${BUILD_CPP_EXAMPLES}" STREQUAL "ON")
  set(CPP_EXAMPLES_DIR cpp_examples)
  set(CPP_EXAMPLES_BIN profile)
  file(GLOB CPP_EXAMPLES_SRCS ${CPP_EXAMPLES_DIR}/${CPP_EXAMPLES_BIN}.cc)
  add_executable(${CPP_EXAMPLES_BIN} ${CPP_EXAMPLES_SRCS})
  set_target_properties(${CPP_EXAMPLES_BIN}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CPP_EXAMPLES_DIR})
  target_link_libraries(${CPP_EXAMPLES_BIN} mirage_runtime)
endif()

option(MIRAGE_BUILD_UNIT_TEST "build unit tests" OFF)

if (MIRAGE_BUILD_UNIT_TEST)
  enable_testing()
  add_subdirectory(tests)
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
  target_link_libraries(mirage_runtime stdc++fs)
  endif()
endif()
