#!/usr/bin/env bash
# M4-I8 post-sweep gates, in the order their evidence is needed:
#   1. ptxas -v + SASS on ONE TU compiled four ways -- proves each arm's -D
#      reached the compile and that neither arm costs registers (HEAD is at 255
#      with a 4 B spill, so a register cost would tax every stage). No GPU.
#   2. the profiler-overhead control -- the SAME msl=897 geometry as the profiled
#      capture, --no-profiler, so the share of the measured dispatch-latency
#      terms that is the INSTRUMENT rather than the runtime is measured instead
#      of quoted from M4-I5's different geometry. Needs the GPU.
#   3. the counterfactual re-derivation of the exact floors. No GPU.
set -uo pipefail
S=$HOME/mpk-qwen35/mirage-m4i8/demo/qwen3_5/accept/opt/m4i8/scripts
M=/var/tmp/m4i8_prof
PY=$HOME/mpk-qwen35/venv-rm/bin/python

echo "########## STAGE 1: ptxas + SASS $(date -Is) ##########"
BS=1 bash "$S/mk_ptxas_m4i8.sh"; echo "PTXAS_RC=$?"

echo; echo "########## STAGE 2: profiler-overhead control $(date -Is) ##########"
bash "$S/gpu_guard_m4i8.sh" 1,0,3,6,2,5 -- bash "$S/prof_overhead_m4i8.sh"
echo "POVH_RC=$?"

echo; echo "########## STAGE 3: floor counterfactuals $(date -Is) ##########"
for CELL in 1:288,384 8:365,461 16:720,733; do
  BS="${CELL%%:*}"; WIN="${CELL#*:}"
  timeout 7200 "$PY" -u "$S/sched_gap.py" \
      "$M/prof/raw_bs${BS}_rep0.npz" "$M/prof/meta_bs${BS}_rep0.json" \
      "$M/prof/task_names.json" --graph "$M/kernel_bs${BS}/task_graph_rank0.json" \
      --window "$WIN" --iters 1 --sim --out "$M/stage/gap_bs${BS}.json" 2>&1 | tail -6
  "$PY" -c "
import json;d=json.load(open('$M/stage/gap_bs${BS}.json'))
f=d['floors']
print('  bs${BS} cp_exact=%.1f work_bound=%.1f'%(f['cp_exact_us'],f['work_bound_us']))
print('  %-34s %10s %10s %10s'%('zero this type ->','cp_exact','d_cp','work_bnd'))
for r in d['floor_counterfactuals'][:12]:
    print('  %-34s %10.1f %10.1f %10.1f'%(r['name'][:34],r['cp_exact_us'],r['cp_delta_us'],r['work_bound_us']))"
done
echo "M4I8_GATES_DONE $(date -Is)"
