# syntax=docker/dockerfile:1.4
# MANYLINUX_IMAGE is always provided by docker-bake.hcl (pinned by digest
# there). This default only applies to direct `docker build` invocations that
# bypass bake, and is intentionally an unpinned tag to surface the misuse.
ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64:latest
FROM ${MANYLINUX_IMAGE}

ARG CUDA_VERSION

RUN set -eux; \
  dnf install -y dnf-plugins-core wget curl gcc-toolset-12-gcc gcc-toolset-12-gcc-c++; \
  dnf clean all; \
  rm -rf /var/cache/dnf

# CUDA toolkit from NVIDIA's RPM repo. The rhel8 repo path is correct for
# manylinux_2_28 (AlmaLinux 8 / RHEL 8). If the base image is bumped to a
# RHEL 9-based manylinux, this URL needs to change to rhel9.
RUN set -eux; \
  CUDA_MAJOR="$(echo "${CUDA_VERSION}" | cut -d. -f1)"; \
  CUDA_MINOR="$(echo "${CUDA_VERSION}" | cut -d. -f2)"; \
  CUDA_VER_DASH="${CUDA_MAJOR}-${CUDA_MINOR}"; \
  dnf config-manager --add-repo "https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo"; \
  dnf install -y "cuda-compiler-${CUDA_VER_DASH}.x86_64" "cuda-libraries-${CUDA_VER_DASH}.x86_64" "cuda-libraries-devel-${CUDA_VER_DASH}.x86_64"; \
  dnf install -y libcudnn9-cuda-12 libcudnn9-devel-cuda-12 || true; \
  dnf clean all; \
  rm -rf /var/cache/dnf

# Python interpreter set is passed as a comma-separated list from bake
# (PYTHON_VERSIONS). We expand each "cpXYZ" into "cpXYZ-cpXYZ" (the form
# manylinux-interpreters expects) here so bake stays terse.
ARG PYTHON_VERSIONS="cp310,cp311,cp312"
RUN set -eux; \
  interpreters=""; \
  IFS=','; for v in ${PYTHON_VERSIONS}; do interpreters="${interpreters} ${v}-${v}"; done; \
  unset IFS; \
  manylinux-interpreters ensure ${interpreters}

# Rust toolchain. Both the rustup-init installer and the toolchain version are
# pinned. The installer is fetched from the archive URL (not sh.rustup.rs,
# which always redirects to latest) and verified against a known sha256. To
# bump the installer: check https://github.com/rust-lang/rustup/releases, then
# update RUSTUP_VERSION + RUSTUP_INIT_SHA256 below. To bump the toolchain: set
# RUST_TOOLCHAIN in docker-bake.hcl.
ARG RUST_TOOLCHAIN=1.86.0
ARG RUSTUP_VERSION=1.28.2
ARG RUSTUP_INIT_SHA256=20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
RUN set -eux; \
  url="https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init"; \
  curl --proto '=https' --tlsv1.2 -fsSL "$url" -o /tmp/rustup-init; \
  echo "${RUSTUP_INIT_SHA256}  /tmp/rustup-init" | sha256sum -c -; \
  chmod +x /tmp/rustup-init; \
  /tmp/rustup-init -y --no-modify-path --default-toolchain "${RUST_TOOLCHAIN}" --profile minimal; \
  rm /tmp/rustup-init

ENV PATH="/opt/rh/gcc-toolset-12/root/usr/bin:/opt/cargo/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rh/gcc-toolset-12/root/usr/lib64:${LD_LIBRARY_PATH}"
ENV CC="/opt/rh/gcc-toolset-12/root/usr/bin/gcc"
ENV CXX="/opt/rh/gcc-toolset-12/root/usr/bin/g++"

WORKDIR /work
