#!/bin/bash
# Script to build Mirage from source

# Setup environment variables
export MIRAGE_ROOT=${{ github.workspace }}
# Pass CUDA path to set_env.sh
export CUDA_PATH="${CUDA_PATH:-${{ steps.cuda-toolkit.outputs.CUDA_PATH }}}"

# Import environment variables setup script
source $(dirname "$0")/set_env.sh

cd $MIRAGE_ROOT
mkdir -p build && cd build

# Configure with CMake using Z3 location
cmake .. \
-DZ3_CXX_INCLUDE_DIRS=${Z3_INCLUDE_PATH} \
-DZ3_LIBRARIES=${Z3_LIB_PATH} \
-DCMAKE_C_COMPILER=$CC \
-DCMAKE_CXX_COMPILER=$CXX

# Build with multiple cores
make -j$(nproc)

# Install Mirage
cd $MIRAGE_ROOT
# Add LD_LIBRARY_PATH for Z3
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
pip install -e .