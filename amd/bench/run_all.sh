#!/bin/bash
# Build and run the AMD microbenchmarks on an MI355X box; every result line goes to $OUT/results.jsonl.
#
#   bash amd/bench/run_all.sh [OUT=~/amd_bench_<time>]       (run it detached: nohup bash run_all.sh > run.log 2>&1 &)
#
# Records first: the machine (rocm-smi, hipcc version, GPU names), the md5 of every source and binary (the proof of which code
# ran). Each benchmark then runs with a time limit; its full output goes to $OUT/<name>.txt and its last line (the JSON summary)
# to results.jsonl. A benchmark that fails or times out is reported and the next one runs. Order: single-GPU checks first,
# the all-GPU exchange last. Nothing else may use the GPUs meanwhile.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HOME/amd_bench_$(date +%m%d_%H%M)}
mkdir -p "$OUT"
ARCH=${ARCH:-gfx950}
echo "out $OUT, arch $ARCH, $(date)"

{
  echo "== date"; date
  echo "== hipcc"; hipcc --version
  echo "== rocm-smi"; rocm-smi --showproductname --showmeminfo vram --showclocks 2>&1
  echo "== processes on the GPUs"; rocm-smi --showpids 2>&1
  echo "== rocminfo gfx"; rocminfo 2>/dev/null | grep -E "Marketing Name|gfx[0-9]+" | sort | uniq -c
} > "$OUT/machine.txt" 2>&1

BENCHES="mfma_layout_check sync_latency handoff_stress l2_writeback launch_gap read_bw mfma_rate xgmi_exchange"
for b in $BENCHES; do
  if ! hipcc -std=c++17 --offload-arch=$ARCH -O3 "$HERE/$b.hip" -o "$OUT/$b" > "$OUT/$b.build.txt" 2>&1; then
    echo "BUILD FAILED: $b (see $OUT/$b.build.txt)"
  fi
done

# the MFMA operand layout first: the expert unit kernels (expert_unit.hip: W13 + quantization, W2) are built for the layout
# that matched (fp4 weights x fp8 tokens)
if [ -x "$OUT/mfma_layout_check" ] && timeout 60 "$OUT/mfma_layout_check" > "$OUT/mfma_layout_check.txt" 2>&1; then
  m=$(grep "fp4 x fp8" "$OUT/mfma_layout_check.txt" | grep MATCH | head -n 1)
  echo "layout: ${m:-NO MATCH (see $OUT/mfma_layout_check.txt)}"
  FP8_LIN=$(echo "$m" | grep -q fp8-lin && echo 1 || echo 0)
  FP4_HI=$(echo "$m" | grep -q fp4-hi && echo 1 || echo 0)
  if [ -z "$m" ] || echo "$m" | grep -q d-cols; then
    echo "the expert unit kernel assumes D = d-rows and a matching fp8/fp4 layout: not built"
  else
    for st in 4 7; do
      for hint in 0 1; do
        v="expert_unit_s${st}_h${hint}"
        hipcc -std=c++17 --offload-arch=$ARCH -O3 -DSTAGES=$st -DHINT=$hint -DFP8_LIN=$FP8_LIN -DFP4_HI=$FP4_HI \
          "$HERE/expert_unit.hip" -o "$OUT/$v" > "$OUT/$v.build.txt" 2>&1 || echo "BUILD FAILED: $v"
        BENCHES="$BENCHES $v"
      done
    done
  fi
fi
(cd "$HERE" && md5sum common.h *.hip run_all.sh; cd "$OUT" && md5sum $BENCHES 2>/dev/null) > "$OUT/md5.txt"
echo "md5 of sources and binaries: $OUT/md5.txt"

declare -A LIMIT=([mfma_layout_check]=60 [sync_latency]=300 [handoff_stress]=300 [l2_writeback]=300 [launch_gap]=120 [read_bw]=600 [mfma_rate]=300 [xgmi_exchange]=600)
for b in $BENCHES; do
  [ -x "$OUT/$b" ] || continue
  t0=$(date +%s)
  timeout "${LIMIT[$b]:-300}" "$OUT/$b" > "$OUT/$b.txt" 2>&1
  rc=$?
  t1=$(date +%s)
  last=$(tail -n 1 "$OUT/$b.txt")
  if [ $rc -eq 0 ] && [ "${last:0:1}" = "{" ]; then
    echo "$last" >> "$OUT/results.jsonl"
    echo "DONE $b in $((t1 - t0)) s"
  else
    echo "FAILED $b (exit $rc after $((t1 - t0)) s): $last"
  fi
  grep -v '^{' "$OUT/$b.txt" | tail -n 40
done
echo "ALL-DONE $(date)"
