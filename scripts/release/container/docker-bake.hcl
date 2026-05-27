# Wheel builder images.
#
# Single matrix-ed target where we can add aarch64/musl/etc as needed as
#  matrix axes.

# To bump,
#   docker buildx imagetools inspect quay.io/pypa/manylinux_2_28_x86_64:latest
# Update builder tag below as needed, rebuild and make sure wheels are tagged correctly.
# Place in a standalone commit.
variable "MANYLINUX_IMAGE" {
  default = "quay.io/pypa/manylinux_2_28_x86_64@sha256:925c46d0d0da06a61a1285b5facceb958431fc35902a865adeceb8d48433658a"
}

# To bump, pick a newer stable from https://forge.rust-lang.org/infra/channel-layout.html
# Place in a standalone commit so any MSRV fallout is isolated.
variable "RUST_TOOLCHAIN" {
  default = "1.86.0"
}

variable "CUDA_VERSIONS" {
  default = ["12.1", "12.4", "12.6", "12.8"]
}

# CPython versions we build for.
# Passed to `manylinux-interpreters ensure` in the builder image.
# Also read by `build-wheels.sh` via `bake --print` as the single source
#  of truth for which interpreters the matrix supports.
variable "PYTHON_VERSIONS" {
  default = ["cp310", "cp311", "cp312"]
}

group "default" {
  targets = ["builder"]
}

target "builder" {
  name = "builder-cu${replace(cuda, ".", "")}"
  matrix = {
    cuda = CUDA_VERSIONS
  }
  context    = "."
  dockerfile = "scripts/release/container/Dockerfile"
  args = {
    MANYLINUX_IMAGE = MANYLINUX_IMAGE
    CUDA_VERSION    = cuda
    RUST_TOOLCHAIN  = RUST_TOOLCHAIN
    PYTHON_VERSIONS = join(",", PYTHON_VERSIONS)
  }
  tags = ["mpk-wheel-builder:cu${replace(cuda, ".", "")}-manylinux_2_28-x86_64"]
}
