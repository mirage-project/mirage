# Wheel release tooling

Builds manylinux wheels per CUDA version. The supported matrix (CUDA versions,
Python interpreters, base image, Rust toolchain) is declared in
[`container/docker-bake.hcl`](container/docker-bake.hcl), that file is the source
of truth for supported matrix.

## Quick start

```bash
# PR cuda version (cp311 + cu124):
scripts/release/build-wheels.sh --src . --out dist/wheels

# Specific tuple:
scripts/release/build-wheels.sh --src . --out dist/wheels --cuda cu124 --python cp311

# Full release matrix:
scripts/release/build-wheels.sh --src . --out dist/wheels --target release
```

Builder images are created automatically via `docker buildx bake` on the first
run and reused thereafter.

## Smoke testing

```bash
# Quick import check:
scripts/release/run-install-smoke.sh --wheel dist/wheels/<name>.whl

# Interactive debug shell:
scripts/release/run-install-smoke.sh --wheel dist/wheels/<name>.whl --shell

# DT_NEEDED / RPATH / .libs/ inspection:
scripts/release/run-install-smoke.sh --wheel dist/wheels/<name>.whl --inspect

# Test in a user-like runtime image instead of the builder:
scripts/release/run-install-smoke.sh --wheel dist/wheels/<name>.whl --image nvidia/cuda:12.4.1-runtime-ubuntu22.04
```

## Wheel naming

Wheels use a PEP 440 local version suffix for the CUDA version:
`mirage_project-0.2.4+cu124-cp311-cp311-manylinux_2_28_x86_64.whl`.
PyPI rejects local versions, so only the `cu121` variant is published there
(with the suffix stripped). All variants are available on GitHub Releases.

## Updating pins

Pins live in `container/docker-bake.hcl` (manylinux image digest, Rust
toolchain) and `container/Dockerfile` (rustup-init version + SHA256). Bump
instructions are in comments next to each pin.

For GitHub Actions SHA pins in `.github/workflows/`:

```bash
# Find the latest patch release for a major version:
gh api repos/<owner>/<action>/releases --jq '.[].tag_name' | grep '^v4\.' | head -1

# Get the commit SHA for that tag:
gh api repos/<owner>/<action>/git/ref/tags/<tag> --jq '.object.sha'
```
