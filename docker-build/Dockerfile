# Use the base image
FROM quay.io/pypa/manylinux_2_28_x86_64:latest

# Update dnf and install required packages
RUN dnf install wget gcc-toolset-9 -y && \
    dnf clean all

# Set the working directory to /root
WORKDIR /root

# Install CUDA 11.8 and cuDNN
RUN wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.5.0.50_cuda11-archive.tar.xz && \
    tar -xJf cudnn-linux-x86_64-9.5.0.50_cuda11-archive.tar.xz && \
    sh ./cuda_11.8.0_520.61.05_linux.run --toolkit --silent && \
    cp -R cudnn-linux-x86_64-9.5.0.50_cuda11-archive/include/* /usr/local/cuda-11.8/include/ && \
    cp -R cudnn-linux-x86_64-9.5.0.50_cuda11-archive/lib/* /usr/local/cuda-11.8/lib64/ && \
    rm -rf cuda_11.8.0_520.61.05_linux.run cudnn-linux-x86_64-9.5.0.50_cuda11-archive.tar.xz cudnn-linux-x86_64-9.5.0.50_cuda11-archive

# Install CUDA 12.2 and cuDNN
RUN wget -q https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run && \
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz && \
    tar -xJf cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz && \
    sh ./cuda_12.2.0_535.54.03_linux.run --toolkit --silent && \
    cp -R cudnn-linux-x86_64-9.5.0.50_cuda12-archive/include/* /usr/local/cuda-12.2/include/ && \
    cp -R cudnn-linux-x86_64-9.5.0.50_cuda12-archive/lib/* /usr/local/cuda-12.2/lib64/ && \
    rm -rf cuda_12.2.0_535.54.03_linux.run cudnn-linux-x86_64-9.5.0.50_cuda12-archive.tar.xz cudnn-linux-x86_64-9.5.0.50_cuda12-archive

# Clone mirage from GitHub
RUN git clone --recurse-submodules https://github.com/mirage-project/mirage.git

# Install Python requirements
WORKDIR /root/mirage
RUN /opt/python/cp38-cp38/bin/pip install -r requirements.txt && \
    /opt/python/cp39-cp39/bin/pip install -r requirements.txt && \
    /opt/python/cp310-cp310/bin/pip install -r requirements.txt && \
    /opt/python/cp311-cp311/bin/pip install -r requirements.txt && \
    /opt/python/cp312-cp312/bin/pip install -r requirements.txt

# Clean up
RUN dnf clean all && \
    rm -rf /var/cache/dnf