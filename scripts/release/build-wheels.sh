#!/usr/bin/env bash
# Drives wheel builds across the CUDA/Python matrix.
#
# Source of truth for the matrix is the bake file: this script queries
#  `docker buildx bake --print` for the set of builder targets and their
#  PYTHON_VERSIONS arg.

set -euo pipefail

# None of these are currently matrix axes
readonly LIBC="manylinux_2_28"
readonly ARCH="x86_64"
readonly DEFAULT_IMAGE_PREFIX="mpk-wheel-builder"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

STAGE="build-wheels"

log() {
  printf '[%s] %s\n' "$STAGE" "$*" >&2
}

die() {
  log "error: $*"
  exit 1
}

on_err() {
  local ec=$? line=${BASH_LINENO[0]:-?} cmd=${BASH_COMMAND:-?}
  log "error at line ${line}: \`${cmd}\` (exit ${ec})"
}
trap on_err ERR

usage() {
  cat <<'EOF'
Usage:
  build-wheels.sh --src <repo> --out <dir> --cuda <cu121|cu124|cu126|cu128> [--python <cp310|cp311|cp312>]
  build-wheels.sh --src <repo> --out <dir> [--target <pr|release>] [--smoke]

Options:
  --src <dir>          Repo root (will be copied into the builder container).
  --out <dir>          Directory to receive built wheels and auditwheel reports.
  --cuda <tag>         Single cuda tag (cuXXX). If set, sweeps interpreters.
  --python <tag>       Narrow to a single interpreter (cpXYZ).
  --target <pr|release>
                       When --cuda is not set: pr builds cp311+cu124 only;
                       release sweeps the full matrix from docker-bake.hcl.
  --smoke              After build, run run-install-smoke.sh on each wheel.
  --image-prefix <s>   Image tag prefix (default: mpk-wheel-builder).
EOF
}

SRC=""
OUT_DIR=""
PYTHON_TAG=""
CUDA_TAG=""
TARGET_SET="pr"
IMAGE_PREFIX="$DEFAULT_IMAGE_PREFIX"
SMOKE=0

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --src) SRC=${2:-}; shift 2 ;;
      --out) OUT_DIR=${2:-}; shift 2 ;;
      --python) PYTHON_TAG=${2:-}; shift 2 ;;
      --cuda) CUDA_TAG=${2:-}; shift 2 ;;
      --target) TARGET_SET=${2:-}; shift 2 ;;
      --image-prefix) IMAGE_PREFIX=${2:-}; shift 2 ;;
      --smoke) SMOKE=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; die "unknown argument: $1" ;;
    esac
  done

  [[ -n "$SRC" ]] || { usage >&2; die "--src is required"; }
  [[ -n "$OUT_DIR" ]] || { usage >&2; die "--out is required"; }
  [[ -d "$SRC" ]] || die "source directory not found: $SRC"
  SRC=$(cd "$SRC" && pwd)
  OUT_DIR=$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)
}

require_tools() {
  command -v jq >/dev/null 2>&1 || die "jq is required"
  command -v docker >/dev/null 2>&1 || die "docker is required"
}

# Read matrix
MATRIX_JSON=""

read_bake_matrix() {
  local bake_file="$SRC/scripts/release/container/docker-bake.hcl"
  [[ -f "$bake_file" ]] || die "bake file missing: $bake_file"

  # progress lines go to stderr, json to stdout, need to separate them
  MATRIX_JSON=$(docker buildx bake -f "$bake_file" --print 2>/dev/null)
  [[ -n "$MATRIX_JSON" ]] || die "docker buildx bake --print produced no output"
}

# Emit one cuda tag (cuXXX) per line, ordered as declared in bake.
all_cuda_tags() {
  jq -r '
    .target
    | to_entries[]
    | select(.key | startswith("builder-cu"))
    | .key
    | sub("^builder-"; "")
  ' <<<"$MATRIX_JSON"
}

# Emit one python tag (cpXYZ) per line.
# Read from the first builder target's PYTHON_VERSIONS arg,
#  bake guarantees all builder-* targets agree.
all_python_tags() {
  jq -r '
    [.target | to_entries[] | select(.key | startswith("builder-cu"))][0]
    | .value.args.PYTHON_VERSIONS
  ' <<<"$MATRIX_JSON" | tr ',' '\n'
}

validate_cuda_tag() {
  local t=$1
  local found=0 avail
  while IFS= read -r avail; do
    [[ "$t" == "$avail" ]] && found=1
  done < <(all_cuda_tags)
  (( found )) || die "unsupported CUDA tag '$t' (available: $(all_cuda_tags | paste -sd,))"
}

validate_python_tag() {
  local t=$1
  local found=0 avail
  while IFS= read -r avail; do
    [[ "$t" == "$avail" ]] && found=1
  done < <(all_python_tags)
  (( found )) || die "unsupported python tag '$t' (available: $(all_python_tags | paste -sd,))"
}

# Target selection has 3 cases:
#   --cuda set   : one CUDA x (--python or all).
#   pr target    : cp311 x cu124 (the one PR tuple).
#   release      : full sweep from bake.

resolve_cuda_tags() {
  if [[ -n "$CUDA_TAG" ]]; then
    printf '%s\n' "$CUDA_TAG"
    return
  fi
  case "$TARGET_SET" in
    pr)      printf 'cu124\n' ;;
    release) all_cuda_tags ;;
    *)       die "unknown target set: $TARGET_SET (expected pr|release)" ;;
  esac
}

resolve_python_tags() {
  if [[ -n "$PYTHON_TAG" ]]; then
    printf '%s\n' "$PYTHON_TAG"
    return
  fi
  if [[ -n "$CUDA_TAG" ]]; then
    # explicit --cuda without --python means all interpreters for that cuda
    all_python_tags
    return
  fi
  case "$TARGET_SET" in
    pr)      printf 'cp311\n' ;;
    release) all_python_tags ;;
  esac
}

ensure_images() {
  local -a targets=("$@")
  local bake_file="$SRC/scripts/release/container/docker-bake.hcl"
  log "ensuring builder images: ${targets[*]}"
  docker buildx bake -f "$bake_file" --load "${targets[@]}"
}

current_container=""

cleanup_container() {
  if [[ -n "$current_container" ]]; then
    docker rm -f "$current_container" >/dev/null 2>&1 || true
    current_container=""
  fi
}
trap cleanup_container EXIT

# Build wheels for a single cuda tag, sweeping python tags in
#  one container invocation, since cuda version is tied to image
#  but supports any interpreter.
#
# Source is copied into the container (not the image and not
#  bind-mounted) to avoid host leakage.
build_one_cuda() {
  local cuda_tag=$1; shift
  local -a pys=("$@")
  STAGE="build $cuda_tag"

  local image="${IMAGE_PREFIX}:${cuda_tag}-${LIBC}-${ARCH}"
  log "building wheels for python=${pys[*]} in $image"

  local cmd="set -euo pipefail; mkdir -p /out; "
  cmd+="bash scripts/release/build-wheel.sh --cuda-tag '$cuda_tag' --out /out"
  local py
  for py in "${pys[@]}"; do
    cmd+=" --python '$py'"
  done

  current_container=$(docker create "$image" bash -lc "$cmd")
  # Pipe a tar stream with exclusions instead of bare `docker cp $SRC/.` to
  # avoid dragging stale build artifacts or bytecode into the container.
  # /work exists in the image (WORKDIR in Dockerfile).
  tar -cf - -C "$SRC" \
    --exclude='./.*' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='docs' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='result' \
    --exclude='img' \
    --exclude='*.png' \
    . | docker cp - "$current_container:/work"
  docker start -a "$current_container"
  docker cp "$current_container:/out/." "$OUT_DIR/"
  docker rm "$current_container" >/dev/null
  current_container=""

  STAGE="build-wheels"
}

smoke_wheels() {
  local smoke_script="$SCRIPT_DIR/run-install-smoke.sh"
  [[ -x "$smoke_script" ]] || die "smoke script missing or not executable: $smoke_script"

  shopt -s nullglob
  local wheels=("$OUT_DIR"/*.whl)
  shopt -u nullglob
  (( ${#wheels[@]} > 0 )) || die "no wheels found for smoke testing in $OUT_DIR"

  local whl base cuda_tag py_tag
  for whl in "${wheels[@]}"; do
    base=$(basename "$whl")
    [[ "$base" =~ \+cu([0-9]{3})- ]] || die "cannot infer CUDA tag from $base"
    cuda_tag="cu${BASH_REMATCH[1]}"
    [[ "$base" =~ -cp([0-9]{3})-cp[0-9]{3}- ]] || die "cannot infer python tag from $base"
    py_tag="cp${BASH_REMATCH[1]}"

    STAGE="smoke $cuda_tag $py_tag"
    log "smoking $base"
    "$smoke_script" --wheel "$whl" --cuda "$cuda_tag" --python "$py_tag"
    STAGE="build-wheels"
  done
}

main() {
  parse_args "$@"
  mkdir -p "$OUT_DIR"
  read_bake_matrix

  mapfile -t CUDA_TAGS < <(resolve_cuda_tags)
  mapfile -t PYTHON_TAGS < <(resolve_python_tags)
  (( ${#CUDA_TAGS[@]} > 0 )) || die "no CUDA tags selected"
  (( ${#PYTHON_TAGS[@]} > 0 )) || die "no python tags selected"

  local t
  for t in "${CUDA_TAGS[@]}"; do validate_cuda_tag "$t"; done
  for t in "${PYTHON_TAGS[@]}"; do validate_python_tag "$t"; done

  log "matrix: cuda=[${CUDA_TAGS[*]}] python=[${PYTHON_TAGS[*]}]"

  local -a bake_targets=()
  for t in "${CUDA_TAGS[@]}"; do bake_targets+=("builder-${t}"); done
  ensure_images "${bake_targets[@]}"

  local cuda
  for cuda in "${CUDA_TAGS[@]}"; do
    build_one_cuda "$cuda" "${PYTHON_TAGS[@]}"
  done

  if (( SMOKE )); then
    smoke_wheels
  fi

  log "done: wheels in $OUT_DIR"
}

main "$@"
