#!/usr/bin/env bash
# Run inside the builder container, builds a wheel for each requested
#  interpreter, post processes the wheel to bundle libs as-needed,
#  and injects the `+cuXXX` local-version segment into the final wheel
#  filename.

set -euo pipefail

STAGE="build-wheel"

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

if [[ "${DEBUG:-0}" == 1 ]]; then
  set -x
fi

usage() {
  cat <<'EOF'
Usage: build-wheel.sh --cuda-tag <cu121|cu124|cu126|cu128> --out <dir> [--python <cp310|cp311|cp312>]...
EOF
}

CUDA_TAG=""
OUT_DIR=""
declare -a PYTHON_FILTERS=()

# scratch dir for repaired wheels before renaming and copied out
tmp_dir=""
cleanup_tmp_dir() {
  [[ -n "$tmp_dir" && -d "$tmp_dir" ]] && rm -rf "$tmp_dir"
}
trap cleanup_tmp_dir EXIT

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --cuda-tag) CUDA_TAG=${2:-}; shift 2 ;;
      --out) OUT_DIR=${2:-}; shift 2 ;;
      --python) PYTHON_FILTERS+=("${2:-}"); shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) usage >&2; die "unknown argument: $1" ;;
    esac
  done

  [[ -n "$CUDA_TAG" ]] || { usage >&2; die "--cuda-tag is required"; }
  [[ -n "$OUT_DIR" ]] || { usage >&2; die "--out is required"; }

  case "$CUDA_TAG" in
    cu121|cu124|cu126|cu128) ;;
    *) die "unsupported CUDA tag: $CUDA_TAG" ;;
  esac

  local py
  for py in "${PYTHON_FILTERS[@]}"; do
    case "$py" in
      cp310|cp311|cp312) ;;
      *) die "unsupported Python filter: $py" ;;
    esac
  done
}

set_cuda_env() {
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
    :
  elif [[ -d "/usr/local/cuda" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  else
    die "CUDA toolkit path not found (set CUDA_HOME)"
  fi

  if [[ -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
    export PATH="${CUDA_HOME}/bin:${PATH}"
  fi
}

# Realistically, we only use things included in manylinux images
#  but putting this here as a best-practice in case someone needs
#  to add a dependency later.
require_tools() {
  command -v auditwheel >/dev/null 2>&1 || die "missing auditwheel"
  command -v find >/dev/null 2>&1 || die "missing find"
}

should_build_interpreter() {
  local short_tag=$1
  (( ${#PYTHON_FILTERS[@]} == 0 )) && return 0
  local py
  for py in "${PYTHON_FILTERS[@]}"; do
    [[ "$short_tag" == "$py" ]] && return 0
  done
  return 1
}

# `auditwheel repair` follows DT_NEEDED and only bundles libraries that are
#  actually linked. To resolve those libs at repair time we need search paths
#  covering every shared object the built wheel depends on but may not have
#  been properly bundled or was dynamically linked against a python-distributed
#  library. Rather than hardcode per-package pip installs (which drifts when
#  pyproject.toml moves) and to avoid mistakes we install the wheel itself
#  into a scratch venv,. pyproject.toml then drives the full runtime dep set
#  and discover `.so` directories underneath site-packages.
#
# This is a heuristic: every dir containing at least one `*.so*`. Strictly
#  broader than filtering on `-name lib`, catches packages that ship .so's at
#  arbitrary layouts and requires no updates when a new native dep lands in
#  pyproject.toml. Again, this is sensible because auditwheel only copies libs
#  it actually needs, the only risk is licensing but thats not a build concern.
discover_repair_lib_path() {
  local site=$1
  find "$site" -name '*.so*' -printf '%h\n' 2>/dev/null \
    | sort -u \
    | paste -sd ':'
}

resolve_venv_site_packages() {
  local venv_py=$1
  local all_sites site venv_root
  all_sites=$("$venv_py" -c 'import json, site; print(json.dumps(site.getsitepackages()))')
  log "site-packages candidates: $all_sites"
  venv_root=$(cd "$(dirname "$venv_py")/.." && pwd)
  # picks the first entry under the venv root
  site=$("$venv_py" -c "
import site, os
venv_root = os.path.realpath('$venv_root')
for s in site.getsitepackages():
    if os.path.realpath(s).startswith(venv_root):
        print(s); break
else:
    # fallback to first entry
    print(site.getsitepackages()[0])
")
  log "using site-packages: $site"
  printf '%s\n' "$site"
}

repair_wheel() {
  local pybin=$1 whl=$2 scratch=$3

  rm -rf venv
  "$pybin" -m venv venv
  local venv_py="$PWD/venv/bin/python"
  "$venv_py" -m pip install --quiet "$whl"
  local site
  site=$(resolve_venv_site_packages "$venv_py")

  local repair_libs
  repair_libs=$(discover_repair_lib_path "$site")

  LD_LIBRARY_PATH="${repair_libs}:${LD_LIBRARY_PATH:-}" \
    auditwheel repair "$whl" \
    --plat manylinux_2_28_x86_64 \
    --exclude libcuda.so.1 \
    --exclude libnvidia-ml.so.1 \
    -w "$scratch/repaired"

  rm -rf venv
}

# Inject the CUDA local-version segment into the wheel filename, e.g.
#   mirage-0.2.4-cp311-cp311-manylinux_2_28_x86_64.whl
#     -> mirage-0.2.4+cu124-cp311-cp311-manylinux_2_28_x86_64.whl
rename_with_cuda_suffix() {
  local whl=$1
  local base new_name
  base=$(basename "$whl")
  new_name=$(echo "$base" | sed -E "s/-(cp[0-9]+-cp[0-9]+)-/+${CUDA_TAG}-\1-/")
  [[ "$new_name" != "$base" ]] || die "could not inject CUDA tag into $base"
  local final_whl="$(dirname "$whl")/$new_name"
  mv "$whl" "$final_whl"
  printf '%s\n' "$final_whl"
}

build_for_interpreter() {
  local pybin=$1 short_tag=$2 scratch=$3
  STAGE="build-wheel $short_tag $CUDA_TAG"

  log "building wheel with $pybin"
  rm -rf build dist "$scratch/repaired"
  mkdir -p "$scratch/repaired"

  "$pybin" -m build --wheel

  shopt -s nullglob
  local built_wheels=(dist/*.whl)
  shopt -u nullglob
  (( ${#built_wheels[@]} > 0 )) || die "python -m build --wheel produced no output for $pybin"

  local whl final_whl
  for whl in "${built_wheels[@]}"; do
    log "repairing $(basename "$whl")"
    repair_wheel "$pybin" "$whl" "$scratch"
  done

  shopt -s nullglob
  local repaired_wheels=("$scratch"/repaired/*.whl)
  shopt -u nullglob
  (( ${#repaired_wheels[@]} > 0 )) || die "auditwheel repair produced no output for $short_tag; check repair logs above"

  for whl in "${repaired_wheels[@]}"; do
    final_whl=$(rename_with_cuda_suffix "$whl")
    cp "$final_whl" "$OUT_DIR/"
    auditwheel show "$final_whl" > "$OUT_DIR/auditwheel-${short_tag}-${CUDA_TAG}.txt"
    log "produced $(basename "$final_whl")"
  done

  # Post-repair structural check. We can't `import mirage` here because
  # core.so DT_NEEDs libcuda.so.1 (driver API) which isn't available in
  # the build container without GPU passthrough. Instead we verify that:
  #   1. The wheel installs cleanly
  #   2. The expected package files exist
  #   3. Every DT_NEEDED library in core.so is either bundled in the wheel,
  #      excluded by policy (libcuda), or provided by the system (libc, etc.)
  # Full import validation happens in run-install-smoke.sh with --device GPU.
  rm -rf verify_venv
  "$pybin" -m venv verify_venv
  local verify_py="$PWD/verify_venv/bin/python"
  "$verify_py" -m pip install --quiet "$OUT_DIR/$(basename "$final_whl")"

  local verify_script
  verify_script=$(cat <<'PYEOF'
import importlib.metadata, pathlib, subprocess, sys

dist = importlib.metadata.distribution("mirage-project")
pkg_dir = pathlib.Path(dist.locate_file("mirage"))
assert pkg_dir.is_dir(), f"mirage package dir missing: {pkg_dir}"

# Check core extension exists
cores = list(pkg_dir.glob("core*.so"))
assert cores, f"no core*.so in {pkg_dir}"
core_so = cores[0]

# Check bundled Rust libs
for lib in ("libabstract_subexpr.so", "libformal_verifier.so"):
    p = pkg_dir / "lib" / lib
    assert p.is_file(), f"missing bundled lib: {p}"

# Check .libs/ directory (auditwheel output)
libs_dir = pkg_dir.parent / "mirage_project.libs"
if libs_dir.is_dir():
    bundled = {f.name for f in libs_dir.iterdir()}
else:
    bundled = set()

# Verify DT_NEEDED: every needed lib must be bundled, excluded, or system-provided
try:
    out = subprocess.check_output(["patchelf", "--print-needed", str(core_so)], text=True)
except FileNotFoundError:
    print("patchelf not available, skipping DT_NEEDED check")
    sys.exit(0)

SYSTEM_LIBS = {"libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libpthread.so.0",
               "libc.so.6", "ld-linux-x86-64.so.2", "librt.so.1", "libdl.so.2"}
EXCLUDED = {"libcuda.so.1", "libnvidia-ml.so.1"}
INTERNAL = {"libabstract_subexpr.so", "libformal_verifier.so"}

missing = []
for lib in out.strip().splitlines():
    lib = lib.strip()
    if lib in SYSTEM_LIBS or lib in EXCLUDED or lib in INTERNAL:
        continue
    # Check if auditwheel bundled it (hash-suffixed name)
    if any(b.startswith(lib.split(".so")[0]) for b in bundled):
        continue
    missing.append(lib)

if missing:
    print(f"ERROR: DT_NEEDED libs not bundled or accounted for: {missing}", file=sys.stderr)
    sys.exit(1)

print(f"verify: {core_so.name} — all {len(out.strip().splitlines())} DT_NEEDED entries accounted for")
PYEOF
)

  "$verify_py" -c "$verify_script"
  log "post-repair structural check passed"
  rm -rf verify_venv
}

main() {
  parse_args "$@"
  set_cuda_env
  require_tools

  mkdir -p "$OUT_DIR"
  tmp_dir=$(mktemp -d)

  local found=0 pybin py_tag short_tag
  for pybin in /opt/python/*/bin/python; do
    [[ -x "$pybin" ]] || continue
    py_tag=$(basename "$(dirname "$(dirname "$pybin")")")
    short_tag=${py_tag%%-*}

    should_build_interpreter "$short_tag" || continue
    found=1
    build_for_interpreter "$pybin" "$short_tag" "$tmp_dir"
  done

  (( found )) || die "no matching interpreters found under /opt/python"
}

main "$@"
