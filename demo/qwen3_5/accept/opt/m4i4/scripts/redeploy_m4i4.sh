#!/usr/bin/env bash
# M4-I4: update the isolated clone's accept/ Python IN PLACE, with provenance.
#
# Why in place rather than a fresh clone: the extension .so does not depend on
# these files and the compiled kernel dirs are expensive, so re-cloning would
# discard ~700 MB of valid per-arm kernels for a change none of the measurement
# paths of geomA/geomM read. The sha256 of every file is printed BEFORE and AFTER
# so the phase boundary is auditable -- a campaign whose tree changed silently
# mid-run is not a campaign.
set -uo pipefail
D=$HOME/mpk-qwen35/mirage-m4i4
ACC=$D/demo/qwen3_5/accept
FILES=${FILES:-"admission_policy.py mpk_engine_run.py harness/gate_ac3_stable.py harness/gate_ac3_stable.sh"}
STAGE=${STAGE:?directory holding the replacement files, same relative layout}

echo "=== BEFORE $(date -Is) ==="
for f in $FILES; do [ -f "$ACC/$f" ] && sha256sum "$ACC/$f"; done
for f in $FILES; do
  [ -f "$STAGE/$f" ] || { echo "missing in stage: $f" >&2; exit 2; }
  cp -f "$STAGE/$f" "$ACC/$f"
done
echo "=== AFTER $(date -Is) ==="
for f in $FILES; do sha256sum "$ACC/$f"; done
echo "=== clone git state (dirty by design; the diff is the redeploy) ==="
git -C "$D" rev-parse HEAD
git -C "$D" status --short -- demo/qwen3_5/accept | head
echo "=== the policy this clone now ships ==="
$HOME/mpk-qwen35/venv-rm/bin/python "$ACC/admission_policy.py"
PYTHONPATH=$D/python $HOME/mpk-qwen35/venv-rm/bin/python \
  "$ACC/harness/tests/test_admission_policy.py" 2>&1 | tail -3
