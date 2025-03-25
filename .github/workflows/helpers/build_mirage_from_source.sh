#!/bin/bash
# Script to build Mirage from source

# Setup environment variables
# Use current directory if not in GitHub Actions
if [ -n "$GITHUB_WORKSPACE" ]; then
  export MIRAGE_ROOT="$GITHUB_WORKSPACE"
else
  export MIRAGE_ROOT="$(pwd)"
fi

# Detect CUDA path from environment or use default
if [ -n "$CUDA_TOOLKIT_PATH" ]; then
  export CUDA_PATH="$CUDA_TOOLKIT_PATH"
elif [ -d "/usr/local/cuda" ]; then
  export CUDA_PATH="/usr/local/cuda"
fi

# Import environment variables setup script
source $(dirname "$0")/set_env.sh

cd "$MIRAGE_ROOT"
mkdir -p build && cd build

# Configure with CMake using Z3 location
cmake .. \
-DZ3_CXX_INCLUDE_DIRS="${Z3_INCLUDE_PATH}" \
-DZ3_LIBRARIES="${Z3_LIB_PATH}" \
-DCMAKE_C_COMPILER="$CC" \
-DCMAKE_CXX_COMPILER="$CXX"

# Build with multiple cores
make -j$(nproc)

# Install Mirage
cd "$MIRAGE_ROOT"
# Add LD_LIBRARY_PATH for Z3
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
pip install -e .