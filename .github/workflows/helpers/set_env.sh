#!/bin/bash
# Script to set up environment variables for Mirage
# This script can be sourced by other scripts or called directly

# Function to set environment variables in GitHub Actions or locally
set_env() {
  local name="$1"
  local value="$2"
  export "$name"="$value"
  
  # If in GitHub Actions, append to GITHUB_ENV
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "$name=$value" >> "$GITHUB_ENV"
  fi
}

# Set CUDA related environment variables
CUDACXX=$(which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")
export CUDACXX
set_env "CUDACXX" "${CUDACXX}"

# Use CUDA_PATH if provided, otherwise try auto-detection
if [ -n "${CUDA_PATH:-}" ]; then
  set_env "CUDA_HOME" "${CUDA_PATH}"
else
  # If CUDA_PATH is not provided, try auto-detection
  CUDA_HOME="/usr/local/cuda"
  export CUDA_HOME
  set_env "CUDA_HOME" "${CUDA_HOME}"
fi

# Set C and C++ compilers
CC=$(which gcc)
export CC
CXX=$(which g++)
export CXX
set_env "CC" "${CC}"
set_env "CXX" "${CXX}"

# Get Z3 paths - use absolute paths from find command
Z3_LIB_PATH=$(find /usr/lib /usr/lib/x86_64-linux-gnu -name "libz3.so" 2>/dev/null | head -1)
export Z3_LIB_PATH
if [ -z "$Z3_LIB_PATH" ]; then
  echo "Error: Could not find libz3.so. Please install Z3 library."
  exit 1
fi
Z3_INCLUDE_PATH="/usr/include"
export Z3_INCLUDE_PATH
set_env "Z3_LIB_PATH" "${Z3_LIB_PATH}"
set_env "Z3_INCLUDE_PATH" "${Z3_INCLUDE_PATH}"

# Set LD_LIBRARY_PATH
if [ -n "$CUDA_PATH" ]; then
  export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
fi
set_env "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH}"

# Print environment variables for debugging
echo "Environment variables set:"
echo "CUDACXX = ${CUDACXX}"
echo "CUDA_HOME = ${CUDA_HOME}"
echo "CC = ${CC}"
echo "CXX = ${CXX}"
echo "Z3_LIB_PATH = ${Z3_LIB_PATH}"
echo "Z3_INCLUDE_PATH = ${Z3_INCLUDE_PATH}"
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}" 