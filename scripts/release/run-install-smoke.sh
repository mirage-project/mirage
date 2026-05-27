#!/usr/bin/env bash
#
# run-install-smoke.sh — install a wheel in a container and validate it.
#
# Modes:
#   (default)   pip install + import mirage + print version
#   --inspect   also print DT_NEEDED, RPATH, and bundled .libs/ for each .so
#   --shell     drop into an interactive shell with the wheel installed
#
# By default, uses the mpk-wheel-builder image matching --cuda/--python.
# Use --image to test against a different base (e.g., a user-like runtime image).

set -euo pipefail

# ---- logging / helpers ------------------------------------------------------

log() { printf '[smoke] %s\n' "$*" >&2; }
die() { log "error: $*"; exit 1; }

# ---- argument parsing -------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  run-install-smoke.sh --wheel <path.whl> [options]

Required:
  --wheel <path>       Path to wheel file.

Image selection (pick one):
  --image <img>        Docker image to test in (any image, e.g. nvidia/cuda:12.4.1-runtime-ubuntu22.04).
  --cuda <cuXXX>       Use mpk-wheel-builder:cuXXX-manylinux_2_28-x86_64.
                       Defaults to inferring from wheel filename if omitted.

Options:
  --python <cpXYZ>     Interpreter tag (default: cp311). Ignored with --image
                       unless the image has /opt/python/cpXYZ-cpXYZ/bin/python.
  --shell              Drop into an interactive shell with the wheel installed.
  --inspect            After import, print DT_NEEDED, RPATH, and .libs/ contents.
  --no-gpu             Skip GPU passthrough (import-only, no CUDA runtime).

Environment:
  WHEEL_GPU_ARGS       Override GPU passthrough flags.
                       Default: "--device nvidia.com/gpu=all" (CDI syntax).
                       Set to "--gpus all" for legacy nvidia-docker, or "" to disable.
EOF
}

WHEEL=""
IMAGE=""
CUDA_TAG=""
PYTHON_TAG="cp311"
MODE="smoke"       # smoke | inspect | shell
GPU=1

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --wheel)   WHEEL=${2:-}; shift 2 ;;
      --image)   IMAGE=${2:-}; shift 2 ;;
      --cuda)    CUDA_TAG=${2:-}; shift 2 ;;
      --python)  PYTHON_TAG=${2:-}; shift 2 ;;
      --shell)   MODE=shell; shift ;;
      --inspect) MODE=inspect; shift ;;
      --no-gpu)  GPU=0; shift ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; die "unknown argument: $1" ;;
    esac
  done

  [[ -n "$WHEEL" ]] || { usage >&2; die "--wheel is required"; }
  [[ -f "$WHEEL" ]] || die "wheel not found: $WHEEL"
}

# ---- image resolution -------------------------------------------------------

# Accept "cu124" or "12.4" and normalize to "cu124".
normalize_cuda_tag() {
  local input=$1
  case "$input" in
    [0-9]*.[0-9]*) printf 'cu%s\n' "$(echo "$input" | tr -d '.')" ;;
    cu*)           printf '%s\n' "$input" ;;
    *)             die "cannot parse CUDA tag: $input" ;;
  esac
}

# Try to infer CUDA tag from wheel filename (e.g., mirage-0.2.4+cu124-cp311-...).
infer_cuda_tag() {
  local base=$1
  if [[ "$base" =~ \+cu([0-9]{3})- ]]; then
    printf 'cu%s\n' "${BASH_REMATCH[1]}"
  fi
}

resolve_image() {
  if [[ -n "$IMAGE" ]]; then
    return
  fi

  local base
  base=$(basename "$WHEEL")

  if [[ -z "$CUDA_TAG" ]]; then
    CUDA_TAG=$(infer_cuda_tag "$base") || true
    [[ -n "$CUDA_TAG" ]] || die "cannot infer CUDA tag from $base; pass --cuda or --image"
    log "inferred CUDA tag from filename: $CUDA_TAG"
  else
    CUDA_TAG=$(normalize_cuda_tag "$CUDA_TAG")
  fi

  IMAGE="mpk-wheel-builder:${CUDA_TAG}-manylinux_2_28-x86_64"
}

# ---- docker run helpers -----------------------------------------------------

resolve_gpu_args() {
  if (( ! GPU )); then
    WHEEL_GPU_ARGS=""
    return
  fi
  WHEEL_GPU_ARGS="${WHEEL_GPU_ARGS---device nvidia.com/gpu=all}"
}

resolve_python_bin() {
  # In the builder image, interpreters live under /opt/python/cpXYZ-cpXYZ/bin/.
  # In other images, fall back to python3 on PATH.
  local py_bin="/opt/python/${PYTHON_TAG}-${PYTHON_TAG}/bin/python"
  printf '%s' "$py_bin"
}

docker_run() {
  local interactive=$1; shift
  local wheel_dir wheel_basename py_bin
  wheel_dir=$(cd "$(dirname "$WHEEL")" && pwd)
  wheel_basename=$(basename "$WHEEL")
  py_bin=$(resolve_python_bin)
  resolve_gpu_args

  local -a docker_args=(
    docker run --rm
    -v "$wheel_dir":/wheelhouse:ro
  )

  if (( interactive )); then
    docker_args+=(-it)
  fi

  # shellcheck disable=SC2086  # intentional word-splitting on WHEEL_GPU_ARGS
  "${docker_args[@]}" $WHEEL_GPU_ARGS "$IMAGE" "$@"
}

# ---- modes ------------------------------------------------------------------

run_smoke() {
  local py_bin wheel_basename
  py_bin=$(resolve_python_bin)
  wheel_basename=$(basename "$WHEEL")
  log "smoke: $IMAGE (python=$PYTHON_TAG)"

  local script
  script=$(cat <<EOF
set -euo pipefail
$py_bin -m pip install --quiet "/wheelhouse/$wheel_basename"
$py_bin -c 'import mirage; print("mirage", mirage.__version__, mirage.__file__)'
EOF
)

  docker_run 0 bash -lc "$script"
  log "passed"
}

run_inspect() {
  local py_bin wheel_basename
  py_bin=$(resolve_python_bin)
  wheel_basename=$(basename "$WHEEL")
  log "inspect: $IMAGE (python=$PYTHON_TAG)"

  local script
  script=$(cat <<EOF
set -euo pipefail
$py_bin -m pip install --quiet "/wheelhouse/$wheel_basename"
$py_bin -c 'import mirage; print("mirage", mirage.__version__, mirage.__file__)'

echo ""
echo "=== DT_NEEDED / RPATH ==="
site=\$($py_bin -c 'print(__import__("site").getsitepackages()[0])')
for so in "\$site"/mirage/*.so "\$site"/mirage/**/*.so; do
  [ -f "\$so" ] || continue
  echo ""
  echo "--- \$(basename "\$so") ---"
  if command -v patchelf >/dev/null 2>&1; then
    echo "NEEDED: \$(patchelf --print-needed "\$so" | tr '\n' ' ')"
    echo "RPATH:  \$(patchelf --print-rpath "\$so")"
  elif command -v readelf >/dev/null 2>&1; then
    readelf -d "\$so" 2>/dev/null | grep -E 'NEEDED|RPATH|RUNPATH' || true
  else
    echo "(no patchelf or readelf available)"
  fi
done

echo ""
echo "=== Bundled .libs/ ==="
libs_dir="\$site/mirage_project.libs"
if [ -d "\$libs_dir" ]; then
  ls -lh "\$libs_dir"/
else
  echo "(no mirage_project.libs/ directory)"
fi
EOF
)

  docker_run 0 bash -lc "$script"
  log "done"
}

run_shell() {
  local py_bin
  py_bin=$(resolve_python_bin)
  log "shell: $IMAGE (python=$PYTHON_TAG)"
  log "wheel will be at /wheelhouse/$(basename "$WHEEL")"
  log "install with: $py_bin -m pip install /wheelhouse/$(basename "$WHEEL")"

  docker_run 1 bash -l
}

# ---- main -------------------------------------------------------------------

main() {
  parse_args "$@"
  resolve_image

  case "$MODE" in
    smoke)   run_smoke ;;
    inspect) run_inspect ;;
    shell)   run_shell ;;
  esac
}

main "$@"
