# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget sudo binutils git python3 python3-pip python3-venv \
    g++ make libboost-all-dev curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install CMake
RUN wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh && \
    mkdir -p /opt/cmake && sh cmake.sh --skip-license --prefix=/opt/cmake && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    rm cmake.sh

# Install PyTorch
RUN pip3 install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

ENV MIRAGE_HOME=/usr/mirage/

COPY install_mirage.sh /install/install_mirage.sh
RUN bash /install/install_mirage.sh

WORKDIR /usr/mirage
