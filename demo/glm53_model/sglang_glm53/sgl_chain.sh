#!/usr/bin/env bash
# GLM-5.3 end to end on one node of 8 GPUs, measured as demo/glm53_flash measures GLM-5.3-Flash (sglang_dsa/sgl_chain.py
# and nsys_e2e.py): per configuration and prompt length one engine (sgl_bench.py, TP 8, bs 8 decode, 256 new tokens)
# under nsys, whole CUDA graphs traced (--cuda-graph-trace=graph: per-node records break the kernels' PDL overlap); the
# period of the decode graph's replays (median) from the trace, plus the wall clock per output token and the greedy outputs.
#   base         SGLang alone
#   ours         the plugin (sglang_glm53): every layer's attention and FFN on the static megakernel, the all-reduce and the
#                norm inside the kernels, PDL (the next kernel launches at residual_add)
#   ours_nopdl   ours without PDL (MIRAGE_STATIC_PDL=0)
#   ours_engine  the plugin with MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=0: SGLang's all-reduce and fused add + RMSNorm between
#                the kernels
# Results: OUT/<config>_p<len>.log (wall clock, outputs), OUT/<config>_p<len>.sqlite, OUT/table.txt.
#
# MODEL=/path/to/GLM-5.3 [OUT=...] [CONFIGS="base ours ours_nopdl ours_engine"] [LENS="256 4000"]
#   [MIRAGE_STATIC_PLAN=hand] bash sgl_chain.sh
# Each configuration runs without the plugin's switches of the calling shell (SGLANG_EXTERNAL_MODEL_PACKAGE, STATIC_*,
# MIRAGE_STATIC_* except MIRAGE_STATIC_PLAN and MIRAGE_STATIC_BUILD_CACHE): only the configuration sets them.
# Needs: sglang 0.5.21, mirage installed from this repository (pip install -e .: the core registers row_rmsnorm), CUDA 13
# nvcc, nsys, 8 GPUs with NVLink multicast (NVLS: the kernels' all-reduce; ours_engine runs without).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
MODEL=${MODEL:?set MODEL to the local GLM-5.3 snapshot}
OUT=${OUT:-$PWD/glm53_tp8_e2e}
CONFIGS=${CONFIGS:-"base ours ours_nopdl ours_engine"}
LENS=${LENS:-"256 4000"}
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$OUT"
NSYS=(nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx
      --cuda-graph-trace=graph --trace-fork-before-exec=true)
PLUGIN=(PYTHONPATH="$REPO/python:$REPO:$REPO/demo/glm53_model" SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_glm53
        MIRAGE_STATIC_PLAN="${MIRAGE_STATIC_PLAN:-hand}")
CLEAR=()
for v in $(compgen -e); do
  case $v in
    MIRAGE_STATIC_PLAN|MIRAGE_STATIC_BUILD_CACHE) ;;
    SGLANG_EXTERNAL_MODEL_PACKAGE|STATIC_*|MIRAGE_STATIC_*) CLEAR+=(-u "$v") ;;
  esac
done
: > "$OUT/table.txt"
for p in $LENS; do
  for cfg in $CONFIGS; do
    case $cfg in
      base) E=() ;;
      ours) E=("${PLUGIN[@]}") ;;
      ours_nopdl) E=("${PLUGIN[@]}" MIRAGE_STATIC_PDL=0) ;;
      ours_engine) E=("${PLUGIN[@]}" MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=0) ;;
      *) echo "unknown configuration $cfg" >&2; exit 1 ;;
    esac
    name=${cfg}_p$p
    echo "=== $name $(date +%H:%M:%S)"
    rm -f "$OUT/$name.log" "$OUT/$name.nsys-rep" "$OUT/$name.sqlite"   # no result of an earlier run in the table
    env "${CLEAR[@]}" "${E[@]}" "${NSYS[@]}" -o "$OUT/$name" python "$HERE/sgl_bench.py" --model "$MODEL" --tp 8 --bs 8 \
      --prompt-tokens "$p" --reps 3 --profile --profile-steps 64 --profile-tokens 256 > "$OUT/$name.log" 2>&1
    # SGLang serves its own model when the plugin does not load. Every rank of a plugin configuration prints that the
    # plugin is on, and no rank of base.
    n=$(grep -c "mirage static megakernel decode: on" "$OUT/$name.log" || true)
    want=8
    [ "$cfg" = base ] && want=0
    [ "$n" -eq "$want" ] || echo "  WARNING: the plugin is on in $n of 8 ranks, not $want (see $OUT/$name.log)"
    grep -aE "us per output token|^output|Error|Traceback" "$OUT/$name.log" | cut -c1-200
    if [ -f "$OUT/$name.nsys-rep" ]; then
      nsys export --type sqlite --force-overwrite true -o "$OUT/$name.sqlite" "$OUT/$name.nsys-rep" > /dev/null 2>&1
      line=$(python "$REPO/demo/glm53_flash/sglang_dsa/nsys_e2e.py" --one "$OUT/$name.sqlite" | tail -1)
      echo "  $line"
      echo "$name: $line" >> "$OUT/table.txt"
    fi
  done
done
echo "=== decode graph period per configuration (nsys_e2e.py --one, GPU 0)"
cat "$OUT/table.txt"
