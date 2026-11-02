"""Answers of two server runs (e2e_client.py's answers.json), prompt by prompt: same or not, and the first new token that differs.
python compare_answers.py RUN_DIR_A RUN_DIR_B"""
import json, sys

A, B = (json.load(open(f"{d}/answers.json")) for d in sys.argv[1:3])
same = 0
for i, (a, b) in enumerate(zip(A, B)):
    x, y = a["ids"] or list(a["text"]), b["ids"] or list(b["text"]); unit = "token" if a["ids"] else "char"
    k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), None)
    if k is None and len(x) == len(y):
        same += 1; print(f"  prompt {i}: same ({len(x)} {unit}s)")
    else:
        k = min(len(x), len(y)) if k is None else k
        print(f"  prompt {i}: differs from {unit} {k} of {len(x)} | A {json.dumps(a['text'][:70])} | B {json.dumps(b['text'][:70])}")
print(f"COMPARE {sys.argv[1].rstrip('/').split('/')[-1]} vs {sys.argv[2].rstrip('/').split('/')[-1]}: {same} of {len(A)} answers the same")
