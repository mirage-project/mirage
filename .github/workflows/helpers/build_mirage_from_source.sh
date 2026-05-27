#!/bin/bash
# Script to build Mirage from source
set -xeuo pipefail

# Setup environment variables
# Use current directory if not in GitHub Actions
if [ -n "${GITHUB_WORKSPACE:-}" ]; then
  MIRAGE_ROOT="$GITHUB_WORKSPACE"
else
  MIRAGE_ROOT="$(pwd)"
fi
export MIRAGE_ROOT

# Detect CUDA path from environment or use default
if [ -n "${CUDA_TOOLKIT_PATH:-}" ]; then
  export CUDA_PATH="$CUDA_TOOLKIT_PATH"
elif [ -d "/usr/local/cuda" ]; then
  export CUDA_PATH="/usr/local/cuda"
fi

# Import environment variables setup script
# shellcheck source=.github/workflows/helpers/set_env.sh
source "$(dirname "$0")/set_env.sh"

# Build and install Mirage via pip (setup.py handles cargo, cmake, cython)
cd "$MIRAGE_ROOT" || { echo "Error: Could not change directory to $MIRAGE_ROOT"; exit 1; }
pip install -e .
