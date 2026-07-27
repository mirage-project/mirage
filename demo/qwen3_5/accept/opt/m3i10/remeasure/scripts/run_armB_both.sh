#!/usr/bin/env bash
cd ~/mpk-qwen35/m3i10-remeasure
bash gpu_guard_m3i10rm.sh 1,2,3,4,5,6 -- bash run_m3i10rm.sh B noprof
bash gpu_guard_m3i10rm.sh 1,2,3,4,5,6 -- bash run_m3i10rm.sh B prof
