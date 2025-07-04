#######################################################
# Enhanced version of find CUDA.
#
# Usage:
#   find_cuda(${USE_CUDA})
#
# - When USE_CUDA=ON, use auto search
# - When USE_CUDA=/path/to/cuda-path, use the cuda path
#
# Provide variables:
#
# - CUDA_FOUND
# - CUDA_INCLUDE_DIRS
# - CUDA_TOOLKIT_ROOT_DIR
# - CUDA_CUDA_LIBRARY
# - CUDA_CUDART_LIBRARY
# - [Disabled] CUDA_NVRTC_LIBRARY
#
macro(find_cuda use_cuda)
  set(__use_cuda ${use_cuda})
  if(__use_cuda STREQUAL "ON")
    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17")
      find_package(CUDAToolkit QUIET)
      if(CUDAToolkit_FOUND)
        set(CUDA_FOUND TRUE)
        set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
        set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})
        set(CUDA_CUDA_LIBRARY ${CUDAToolkit_LIBRARY_DIR}/libcuda.so)
        set(CUDA_CUDART_LIBRARY ${CUDAToolkit_LIBRARY_DIR}/libcudart.so)
        # set(CUDA_NVRTC_LIBRARY ${CUDAToolkit_LIBRARY_DIR}/libnvrtc.so)
      endif()
    else()
      find_package(CUDA QUIET)
    endif()
  elseif(IS_DIRECTORY ${__use_cuda})
    set(CUDA_TOOLKIT_ROOT_DIR ${__use_cuda})
    message(STATUS "Custom CUDA_PATH=" ${CUDA_TOOLKIT_ROOT_DIR})
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    set(CUDA_FOUND TRUE)
    if(MSVC)
      find_library(CUDA_CUDART_LIBRARY cudart
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
    else(MSVC)
      find_library(CUDA_CUDART_LIBRARY cudart
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    endif(MSVC)
  endif()

  # additional libraries
  if(CUDA_FOUND)
    if(MSVC)
      find_library(CUDA_CUDA_LIBRARY cuda
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
      #find_library(CUDA_NVRTC_LIBRARY nvrtc
      #  ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
      #  ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
    else(MSVC)
      find_library(_CUDA_CUDA_LIBRARY cuda
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib64/stubs
        NO_DEFAULT_PATH)
      if(_CUDA_CUDA_LIBRARY)
        set(CUDA_CUDA_LIBRARY ${_CUDA_CUDA_LIBRARY})
      endif()
      #find_library(CUDA_NVRTC_LIBRARY nvrtc
      #  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
      #  PATH_SUFFIXES lib lib64 lib64/stubs lib/x86_64-linux-gnu
      #  NO_DEFAULT_PATH)
    endif(MSVC)
    #list(APPEND CUDA_INCLUDE_DIRS /opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2/include)
    message(STATUS "CUDA_INCLUDE_DIRS=" ${CUDA_INCLUDE_DIRS})
    message(STATUS "Found CUDA_TOOLKIT_ROOT_DIR=" ${CUDA_TOOLKIT_ROOT_DIR})
    message(STATUS "Found CUDA_CUDA_LIBRARY=" ${CUDA_CUDA_LIBRARY})
    message(STATUS "Found CUDA_CUDART_LIBRARY=" ${CUDA_CUDART_LIBRARY})
    # message(STATUS "Found CUDA_NVRTC_LIBRARY=" ${CUDA_NVRTC_LIBRARY})
  endif(CUDA_FOUND)
endmacro(find_cuda)
