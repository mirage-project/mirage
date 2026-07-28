#!/usr/bin/env bash
# M3-I6a phase 2 body: oracle re-run at the candidate pass size, then the FULL
# AC-3 sweep + per-case byte diff.  Runs under one GPU claim.
set -uo pipefail
M=$HOME/mpk-qwen35/i6a
bash "$M/gate_oracle2.sh" 2>&1 | tee "$M/logs/gate_oracle2.log" | tail -40
PHASES=ac3 bash "$M/gate_all.sh"
