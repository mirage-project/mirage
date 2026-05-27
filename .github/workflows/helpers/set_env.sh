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

# Set LD_LIBRARY_PATH
if [ -n "${CUDA_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
set_env "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH}"

# Print environment variables for debugging
echo "Environment variables set:"
echo "CUDACXX = ${CUDACXX}"
echo "CUDA_HOME = ${CUDA_HOME}"
echo "CC = ${CC}"
echo "CXX = ${CXX}"
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
