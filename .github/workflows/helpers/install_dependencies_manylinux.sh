#!/bin/bash
# Script to install dependencies inside manylinux_2_28 container for wheel builds.
# Usage: install_dependencies_manylinux.sh <CUDA_VERSION> <PYTHON_VERSION>
#   CUDA_VERSION: e.g. "12.1.1", "12.4.1", "12.6.2", "12.8.1"
#   PYTHON_VERSION: e.g. "3.10", "3.11", "3.12"

set -e

CUDA_VERSION="${1:?Usage: $0 <CUDA_VERSION> <PYTHON_VERSION>}"
PYTHON_VERSION="${2:?Usage: $0 <CUDA_VERSION> <PYTHON_VERSION>}"

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
CUDA_SHORT="${CUDA_MAJOR}${CUDA_MINOR}"

# --- Python (pre-installed in manylinux image) ---
PY_TAG="cp$(echo "$PYTHON_VERSION" | tr -d '.')"
PY_BIN="/opt/python/${PY_TAG}-${PY_TAG}/bin"
if [ ! -d "$PY_BIN" ]; then
  echo "ERROR: Python ${PYTHON_VERSION} not found at ${PY_BIN}"
  exit 1
fi
export PATH="${PY_BIN}:${PATH}"
echo "Using Python: $(python --version) from ${PY_BIN}"

# Persist PATH for subsequent GitHub Actions steps
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "${PY_BIN}" >> "$GITHUB_PATH"
fi

# --- System dependencies (AlmaLinux 8 / RHEL 8) ---
dnf install -y make boost-devel wget

# --- GCC 12 (required: CUDA < 12.8 does not support GCC > 13) ---
dnf install -y gcc-toolset-12-gcc gcc-toolset-12-gcc-c++
GCC12_BIN="/opt/rh/gcc-toolset-12/root/usr/bin"
GCC12_LIB="/opt/rh/gcc-toolset-12/root/usr/lib64"
export PATH="${GCC12_BIN}:${PATH}"
export CC="${GCC12_BIN}/gcc"
export CXX="${GCC12_BIN}/g++"
export LD_LIBRARY_PATH="${GCC12_LIB}:${LD_LIBRARY_PATH:-}"

# Persist GCC 12 for subsequent GitHub Actions steps
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "${GCC12_BIN}" >> "$GITHUB_PATH"
fi
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "CC=${GCC12_BIN}/gcc" >> "$GITHUB_ENV"
  echo "CXX=${GCC12_BIN}/g++" >> "$GITHUB_ENV"
  echo "LD_LIBRARY_PATH=${GCC12_LIB}:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
fi

# --- CUDA toolkit via NVIDIA repo ---
dnf config-manager --add-repo \
  "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo"
dnf install -y "cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR}"

CUDA_PATH="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}"
export PATH="${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# Persist for GitHub Actions
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "CUDA_HOME=${CUDA_PATH}" >> "$GITHUB_ENV"
  echo "CUDA_PATH=${CUDA_PATH}" >> "$GITHUB_ENV"
fi
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "${CUDA_PATH}/bin" >> "$GITHUB_PATH"
fi

# --- cuDNN ---
dnf install -y libcudnn9-cuda-12 libcudnn9-devel-cuda-12 || \

  echo "WARNING: cuDNN install failed, continuing without it"
ldconfig

# --- Z3 (install from pip, then expose paths for CMake) ---
pip install z3-solver
Z3_PKG_DIR=$(python -c "import z3, os; print(os.path.dirname(z3.__file__))")
Z3_LIB_PATH="${Z3_PKG_DIR}/lib/libz3.so"
Z3_INCLUDE_PATH="${Z3_PKG_DIR}/include"

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "Z3_LIB_PATH=${Z3_LIB_PATH}" >> "$GITHUB_ENV"
  echo "Z3_INCLUDE_PATH=${Z3_INCLUDE_PATH}" >> "$GITHUB_ENV"
  echo "Z3_LIB_DIR=${Z3_PKG_DIR}/lib" >> "$GITHUB_ENV"
fi

# --- CMake ---
pip install cmake

# --- Python build dependencies ---
pip install --upgrade pip build setuptools wheel cython pyproject-metadata

# --- PyTorch ---
TORCH_INDEX="cu${CUDA_SHORT}"
case "${CUDA_SHORT}" in
  121) TORCH_INDEX="cu124" ;;  # PyTorch dropped cu121 from 2.6.0+
  128) TORCH_INDEX="cu126" ;;  # cu128 index doesn't exist yet
esac
echo "Installing PyTorch for CUDA ${CUDA_VERSION} (using ${TORCH_INDEX} index)..."
pip install torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

# --- Project requirements ---
if [ -f requirements.txt ]; then
  grep -v '^[[:space:]]*#' requirements.txt | grep -v 'git+' | pip install -r /dev/stdin || true
  { grep 'git+' requirements.txt || true; } | while read -r dep; do
    pip install "$dep" || echo "WARNING: Failed to install $dep"
  done
fi

# --- Rust (needed for tokenizers etc.) ---
curl https://sh.rustup.rs -sSf | sh -s -- -y
# shellcheck source=/dev/null
. "$HOME/.cargo/env"

# Persist cargo PATH for subsequent GitHub Actions steps
if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
fi

# --- auditwheel and patchelf ---
pip install auditwheel patchelf

echo "=== manylinux dependency installation complete ==="
echo "Python: $(python --version)"
echo "CUDA: $(nvcc --version | tail -1)"
echo "GCC: $(gcc --version | head -1)"
echo "GLIBC: $(ldd --version | head -1)"
