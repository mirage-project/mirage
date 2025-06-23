#!/bin/bash
# Script to install system dependencies required by Mirage
sudo apt update
sudo apt install -y software-properties-common lsb-release wget python3-pip g++ make libboost-all-dev

# Install Z3
sudo apt-get install -y libz3-4 libz3-dev

# Make sure Z3 lib is found (enforces Z3 version 4.15)
sudo ln -s /usr/lib/x86_64-linux-gnu/libz3.so /usr/lib/libz3.so.4.15 || true
sudo ldconfig

# Install CMake
CMAKE_VERSION=3.27.0
wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh
sudo mkdir -p /opt/cmake && sudo sh cmake.sh --skip-license --prefix=/opt/cmake
sudo ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake

# Install Python dependencies (including torch)
pip3 install --upgrade pip build setuptools wheel cython

# Install PyTorch temporarily since it is not included in requirements.txt so far
pip3 install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install project requirements
if [ -f requirements.txt ]; then
pip3 install -r requirements.txt
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
