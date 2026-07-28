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

# Install Z3 (system headers/lib; the runtime lib actually linked against is
# the pip 'z3-solver' package installed with the Python requirements below).
sudo apt-get install -y libz3-4 libz3-dev
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

# Expose the pip-installed z3-solver's bundled libz3 to the dynamic linker.
# The mirage extension links against z3-solver's own libz3.so (see setup.py),
# whose SONAME tracks the z3 *library* version, which can differ across
# package releases (e.g. 4.16.0.0 -> "libz3.so.4.16", 5.0.0.0 -> "libz3.so.5.0").
# Derive the real SONAME at install time so we never hard-code (and drift) it.
Z3_PY_LIB=$(python3 -c "import os, z3; print(os.path.join(os.path.dirname(z3.__file__), 'lib', 'libz3.so'))" 2>/dev/null)
if [ -n "$Z3_PY_LIB" ] && [ -f "$Z3_PY_LIB" ]; then
  Z3_PY_DIR=$(dirname "$Z3_PY_LIB")
  # SONAME recorded inside the ELF (readelf -> objdump fallback).
  Z3_SONAME=$(readelf -d "$Z3_PY_LIB" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\].*/\1/p' | head -1)
  if [ -z "$Z3_SONAME" ]; then
    Z3_SONAME=$(objdump -p "$Z3_PY_LIB" 2>/dev/null | awk '/SONAME/{print $2; exit}')
  fi
  echo "z3-solver libz3.so = ${Z3_PY_LIB} (SONAME=${Z3_SONAME:-<unknown>})"
  # Register the z3-solver lib directory, plus symlink the unversioned name
  # and the real SONAME into /usr/lib so the loader always resolves it.
  echo "$Z3_PY_DIR" | sudo tee /etc/ld.so.conf.d/z3-solver.conf >/dev/null
  sudo ln -sf "$Z3_PY_LIB" /usr/lib/libz3.so
  if [ -n "$Z3_SONAME" ]; then
    sudo ln -sf "$Z3_PY_LIB" "/usr/lib/${Z3_SONAME}"
  fi
  sudo ldconfig
else
  echo "WARNING: could not locate z3-solver's bundled libz3.so; relying on system Z3"
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
