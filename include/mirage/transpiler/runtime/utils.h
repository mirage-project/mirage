#pragma once
#include <cstdlib>
#include <iostream>

#include <cuda_runtime_api.h>

#define CHECK_CUDA(status)                                                     \
  do {                                                                         \
    if (status != 0) {                                                         \
      std::cerr << "Cuda failure: " << status << cudaGetErrorString(status)    \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
