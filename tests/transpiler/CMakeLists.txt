# add_mirage_test(test-cuda-transpiler test_cuda_transpiler.cc)
find_package(OpenMP REQUIRED)

add_executable(test-cuda-transpiler test_cuda_transpiler.cc)
target_link_libraries(test-cuda-transpiler mirage_runtime OpenMP::OpenMP_CXX)
target_compile_options(test-cuda-transpiler PRIVATE -march=native -Ofast)
add_test(test-cuda-transpiler test-cuda-transpiler)
