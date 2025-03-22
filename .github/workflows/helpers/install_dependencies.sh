#!/bin/bash
# Script to install system dependencies required by Mirage

set -e

echo "Starting installation of system dependencies required by Mirage..."

# Update apt repositories
sudo apt update

# Install basic development tools and libraries
sudo apt install -y software-properties-common lsb-release wget
sudo apt install -y python3-pip g++ make libboost-all-dev

# Install latest version of CMake
CMAKE_VERSION=3.27.0
wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh
sudo mkdir -p /opt/cmake && sudo sh cmake.sh --skip-license --prefix=/opt/cmake
sudo ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake

# Install Python dependencies
pip3 install --upgrade cython

# Install cuDNN
UBUNTU_VERSION=$(lsb_release -rs | tr -d '.')
wget -c -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update -y
rm -f cuda-keyring_1.1-1_all.deb
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-samples
sudo ldconfig

echo "System dependencies installation completed." 