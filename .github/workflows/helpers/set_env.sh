#!/bin/bash
# Script to set up environment variables for Mirage
# This script can be sourced by other scripts or called directly

# Set CUDA related environment variables
export CUDACXX=$(which nvcc)
echo "CUDACXX=${CUDACXX}" >> $GITHUB_ENV

# Use CUDA_PATH if provided, otherwise try auto-detection
if [ -n "$CUDA_PATH" ]; then
  echo "CUDA_HOME=${CUDA_PATH}" >> $GITHUB_ENV
else
  # If CUDA_PATH is not provided, try auto-detection
  export CUDA_HOME="/usr/local/cuda"
  echo "CUDA_HOME=${CUDA_HOME}" >> $GITHUB_ENV
fi

# Set C and C++ compilers
export CC=$(which gcc)
export CXX=$(which g++)
echo "CC=${CC}" >> $GITHUB_ENV
echo "CXX=${CXX}" >> $GITHUB_ENV

# Get Z3 paths - use absolute paths from find command
export Z3_LIB_PATH=$(find /usr/lib -name "libz3.so" | head -1)
export Z3_INCLUDE_PATH="/usr/include"
echo "Z3_LIB_PATH=${Z3_LIB_PATH}" >> $GITHUB_ENV
echo "Z3_INCLUDE_PATH=${Z3_INCLUDE_PATH}" >> $GITHUB_ENV

# Set LD_LIBRARY_PATH
if [ -n "$CUDA_PATH" ]; then
  export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
fi
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> $GITHUB_ENV

# Print environment variables for debugging
echo "Environment variables set:"
echo "CUDACXX = ${CUDACXX}"
echo "CUDA_HOME = ${CUDA_HOME}"
echo "CC = ${CC}"
echo "CXX = ${CXX}"
echo "Z3_LIB_PATH = ${Z3_LIB_PATH}"
echo "Z3_INCLUDE_PATH = ${Z3_INCLUDE_PATH}"
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}" 