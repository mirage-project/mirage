#!/bin/bash
# Script to install system dependencies required by Mirage
# Usage: install_dependencies.sh [CUDA_VERSION]
#   CUDA_VERSION: e.g. "12.1.1", "12.4.1", "12.6.3" (default: "12.1.1")

set -e

CUDA_VERSION="${1:-12.1.1}"
# Extract major.minor for torch index (e.g., 12.1.1 -> cu121, 12.4.1 -> cu124)
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
CUDA_SHORT="${CUDA_MAJOR}${CUDA_MINOR}"
TORCH_CUDA="cu${CUDA_SHORT}"

sudo apt update
sudo apt install -y software-properties-common lsb-release wget python3-pip g++ make libboost-all-dev

# Install Z3
sudo apt-get install -y libz3-4 libz3-dev

# Make sure Z3 lib is found (enforces Z3 version 4.16)
sudo ln -s /usr/lib/x86_64-linux-gnu/libz3.so /usr/lib/libz3.so.4.16 || true
sudo ldconfig

# Install CMake
CMAKE_VERSION=3.27.0
wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh
sudo mkdir -p /opt/cmake && sudo sh cmake.sh --skip-license --prefix=/opt/cmake
sudo ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake

# Install Python dependencies (including torch)
pip3 install --upgrade pip build setuptools wheel cython

# Install PyTorch matching the target CUDA version
# PyTorch wheel indices don't always match every CUDA toolkit version.
# CUDA is backward compatible within a major version, so we map to the
# closest available PyTorch index.
TORCH_INDEX="cu${CUDA_SHORT}"
case "${CUDA_SHORT}" in
  121) TORCH_INDEX="cu124" ;;  # PyTorch dropped cu121 from 2.6.0+
  128) TORCH_INDEX="cu126" ;;  # cu128 index doesn't exist yet
esac
echo "Installing PyTorch for CUDA ${CUDA_VERSION} (using ${TORCH_INDEX} index)..."
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/${TORCH_INDEX}

# Install project requirements (skip git+ dependencies that break wheel metadata)
if [ -f requirements.txt ]; then
  grep -v '^[[:space:]]*#' requirements.txt | grep -v 'git+' | pip3 install -r /dev/stdin
  # Install git+ dependencies separately (won't be in wheel metadata)
  grep 'git+' requirements.txt | while read -r dep; do
    pip3 install "$dep" || echo "WARNING: Failed to install $dep"
  done
fi

# Install cuDNN
UBUNTU_VERSION=$(lsb_release -rs | tr -d '.')
wget -c -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update -y
rm -f cuda-keyring_1.1-1_all.deb
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples
sudo ldconfig

# Install Rust and Cargo
sudo rm -rf /var/lib/apt/lists/*
# Install Rust
curl https://sh.rustup.rs -sSf | sh -s -- -y
# shellcheck source=/dev/null
. "$HOME/.cargo/env"

# Install auditwheel and patchelf for wheel repair
pip3 install auditwheel patchelf