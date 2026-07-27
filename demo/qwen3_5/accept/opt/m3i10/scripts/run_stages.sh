#!/bin/bash
set -uo pipefail
R="$HOME/mpk-qwen35/m3i10-profile"
exec > "$R/logs/stages.log" 2>&1
cd "$R"
python3 scripts/stage_map.py traces/main/decode_bs1_win*.json \
   --anchor nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT --label bs1 --out-dir out/main
python3 scripts/stage_map.py traces/main/decode_bs16_win*.json \
   --anchor nvjet_sm100_tst_192x16_64x8_2x1_2cta_v_bz_TNT --label bs16 --out-dir out/main
A8=$(python3 - <<'PY'
import json
d=json.load(open("out/main/bs8_kernels.json"))
print(d["per_trace"][0]["anchor_kernel"])
PY
)
echo "bs8 anchor=$A8"
python3 scripts/stage_map.py traces/main/decode_bs8_win*.json \
   --anchor "$A8" --label bs8 --out-dir out/main
echo "=== STAGES_DONE ==="
