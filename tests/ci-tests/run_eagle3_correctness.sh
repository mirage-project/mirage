#!/usr/bin/env bash
#
# EAGLE3 output correctness harness: compare the EAGLE3 megakernel token stream
# against a greedy PyTorch Qwen3-30B-A3B reference, for a given K.
#
# Usage:
#   K=1 bash tests/ci-tests/run_eagle3_correctness.sh
#   K=2 bash tests/ci-tests/run_eagle3_correctness.sh
#
# Env overrides:
#   K        draft steps (default 1); mbt = K+1
#   N        generated tokens to compare (default 50)
#   MODEL    HF model id / local path (default Qwen/Qwen3-30B-A3B)
#   PYTHON   python interpreter (default: the mirage00 conda env python)
#   MIRAGE_HOME  repo root (default: auto from this script)
#
# Steps:
#   1. Generate the greedy torch reference ONCE (reused across all K).
#   2. Ensure the SM100 attention MAX_TOKENS baked literal is >= mbt; if not,
#      edit src/kernel/task_register.cc and rebuild core.so (pip install -e .).
#      A trap restores task_register.cc on EXIT.
#   3. Run the EAGLE3 demo at this K with --save-tokens.
#   4. AC-6 fail-fast: assert the artifact's max_tokens_compiled >= mbt BEFORE
#      comparison (MPK metadata validation does NOT check the source MAX_TOKENS).
#   5. pytest the exact-match comparison.
#
# Skips cleanly (exit 0, no artifacts) when CUDA is unavailable.
set -euo pipefail

ROOT="${MIRAGE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export MIRAGE_HOME="$ROOT"
cd "$ROOT"

K="${K:-1}"
MBT=$((K + 1))
N="${N:-50}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
PYTHON="${PYTHON:-/home/letianr/miniconda3/envs/mirage00/bin/python}"

# The demo imports mpi4py (even for world_size=1); ensure libmpi is loadable.
MPI_LIB_DIR="${MPI_LIB_DIR:-/usr/mpi/gcc/openmpi-4.1.9a1/lib}"
if [[ -d "$MPI_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

OUT_DIR="outputs/qwen3_30b_a3b"
REF_JSON="$OUT_DIR/torch_reference.json"
MPK_JSON="$OUT_DIR/mpk_eagle3_k${K}.json"
TASK_REGISTER="src/kernel/task_register.cc"
# Prompt length for Qwen3 chat template is ~24 tokens; pad max-seq-length so the
# demo has room for prompt + N generated tokens.
MAX_SEQ_LEN=$((256 + N))

echo "=== EAGLE3 correctness runner: K=$K mbt=$MBT N=$N ==="
echo "MIRAGE_HOME=$ROOT  PYTHON=$PYTHON  MODEL=$MODEL"

# --- CUDA availability probe (skip cleanly without GPU) ---
if ! "$PYTHON" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA unavailable; skipping (no artifacts produced)."
    exit 0
fi

mkdir -p "$OUT_DIR"

# --- Step 1: greedy torch reference (once) ---
if [[ ! -f "$REF_JSON" ]]; then
    echo "--- Generating greedy torch reference -> $REF_JSON ---"
    "$PYTHON" demo/qwen3/eagle3_correctness/gen_torch_reference.py \
        --model "$MODEL" --num-tokens "$N" --out "$REF_JSON"
else
    echo "--- Reusing existing reference $REF_JSON ---"
fi

# --- Step 2: ensure baked SM100 attention MAX_TOKENS >= mbt ---
# The codegen literal lives in task_register.cc as ", $MAX_TOKENS, $, $>(" inside
# the multitoken_paged_attention_sm100_task_impl instantiation string.
CURRENT_MT="$(grep -oP '"\$, \$, \$, \$, \K[0-9]+(?=, \$, \$>\(")' "$TASK_REGISTER" | head -1 || true)"
if [[ -z "$CURRENT_MT" ]]; then
    echo "WARNING: could not detect current baked MAX_TOKENS literal in $TASK_REGISTER."
    echo "         Proceeding; the AC-6 gate below will still verify the artifact."
else
    echo "--- Current baked SM100 MAX_TOKENS literal = $CURRENT_MT (need >= $MBT) ---"
fi

REBUILT=0
if [[ -n "$CURRENT_MT" && "$CURRENT_MT" -lt "$MBT" ]]; then
    echo "--- MAX_TOKENS ($CURRENT_MT) < mbt ($MBT): editing $TASK_REGISTER and rebuilding core.so ---"
    cp "$TASK_REGISTER" "$TASK_REGISTER.bak_correctness"
    # Restore the source on EXIT (success or failure).
    trap 'mv -f "$TASK_REGISTER.bak_correctness" "$TASK_REGISTER" 2>/dev/null || true' EXIT
    # Replace the single SM100 attention literal: ", $, $, $, <N>, $, $>(" .
    sed -i -E 's/("\$, \$, \$, \$, )[0-9]+(, \$, \$>\(")/\1'"$MBT"'\2/' "$TASK_REGISTER"
    NEW_MT="$(grep -oP '"\$, \$, \$, \$, \K[0-9]+(?=, \$, \$>\(")' "$TASK_REGISTER" | head -1 || true)"
    echo "    new literal = $NEW_MT"
    echo "--- pip install -e . (rebuild core.so; BL-20260601-mpk-stale-core-so) ---"
    "$PYTHON" -m pip install -e . -v >/tmp/eagle3_rebuild_k${K}.log 2>&1 \
        || { echo "REBUILD FAILED; see /tmp/eagle3_rebuild_k${K}.log"; tail -30 /tmp/eagle3_rebuild_k${K}.log; exit 3; }
    REBUILT=1
fi

# --- Step 3: run the EAGLE3 demo with --save-tokens ---
echo "--- Running EAGLE3 demo (K=$K) -> $MPK_JSON ---"
EAGLE3_DRAFT="${EAGLE3_DRAFT:-/raid/catalyst/models/models--lmsys--SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex/snapshots/d1ac703a537d2b8a5b748d4f5f8ca7e97efe9214}"
"$PYTHON" demo/qwen3/demo_30B_A3B_eagle3.py \
    --use-mirage --eagle3 \
    --model "$MODEL" \
    --eagle3-draft-path "$EAGLE3_DRAFT" \
    --max-num-batched-tokens "$MBT" \
    --num-draft-steps "$K" \
    --max-seq-length "$MAX_SEQ_LEN" \
    --save-tokens "$MPK_JSON"

# --- Step 4: AC-6 fail-fast (compiled MAX_TOKENS >= mbt) BEFORE comparison ---
echo "--- AC-6 gate: compiled MAX_TOKENS >= mbt ($MBT)? ---"
"$PYTHON" - "$MPK_JSON" "$MBT" <<'PYEOF'
import json, sys
path, mbt = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    meta = json.load(f)
mtc = meta.get("max_tokens_compiled")
if mtc is None:
    print(f"AC-6 ABORT: {path} has no max_tokens_compiled"); sys.exit(4)
if mtc < mbt:
    print(f"AC-6 ABORT: compiled MAX_TOKENS ({mtc}) < mbt ({mbt}). "
          f"Kernel undersized; refusing to compare (would be a silent wrong pass).")
    sys.exit(4)
print(f"AC-6 OK: compiled MAX_TOKENS ({mtc}) >= mbt ({mbt}).")
PYEOF

# --- Step 5: pytest the comparison ---
echo "--- pytest comparison vs reference ---"
EAGLE3_MPK_OUTPUT="$MPK_JSON" "$PYTHON" -m pytest -q tests/ci-tests/test_eagle3_correctness.py
echo "=== run complete (K=$K, rebuilt=$REBUILT) ==="
