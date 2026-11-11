"""Batch-size client for one SGLang server run (plain decode, no speculative decoding): BS requests at once (one thread
each), every request a different prompt, greedy, NTOK new tokens each with the end-of-sequence token ignored, so the server
decodes a full batch of BS for most of the run. Prints the wall time, the output tokens per second and the median time per
token per request; the answers go to OUT/answers_bs<BS>.json (text + token ids).
PROFILE=N: also N decode steps of a torch-profiler trace, started DELAY s (default 2) after the requests, into OUT/trace_bs<BS>.
    python bs_client.py OUT BS [NTOK] [PORT]"""
import json
import os
import statistics as st
import sys
import threading
import time
import urllib.request

OUT, BS = sys.argv[1], int(sys.argv[2])
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PORT = sys.argv[4] if len(sys.argv) > 4 else "30000"
BASE = ["The history of the Roman Empire begins with", "def quicksort(arr):", "Explain how a transformer language model works.",
        "Write a short poem about the ocean at night.", "The three laws of thermodynamics are",
        "Translate to French: The weather is nice today and we are going to the park.",
        "Q: What is 17 times 23? A:", "Once upon a time in a small village in the mountains,"]
PROMPTS = [f"Request {i}. {BASE[i % len(BASE)]}" for i in range(BS)]
# PLEN=N (env): each prompt starts with ~N tokens of fixed filler text (decode at a long context, e.g. GSM8K few-shot ~700-1200)
if int(os.environ.get("PLEN", "0")) > 0:
    FILL = "The quick brown fox jumps over the lazy dog near the river bank. "   # (~14 tokens)
    PROMPTS = [FILL * (int(os.environ["PLEN"]) // 14) + p for p in PROMPTS]


def post(path, body, timeout=900):
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(rq, timeout=timeout).read()


def gen(i, res):
    t0 = time.perf_counter()
    d = json.loads(post("/generate", {"text": PROMPTS[i], "sampling_params": {"max_new_tokens": NTOK, "temperature": 0,
                                                                               "ignore_eos": True}}))
    res[i] = (d, (time.perf_counter() - t0) * 1e3)


for i in range(min(BS, 2)):   # warm-up (this batch size's graph is captured at server start)
    gen(i, {})
res = {}
th = [threading.Thread(target=gen, args=(i, res)) for i in range(BS)]
t0 = time.perf_counter()
for t in th:
    t.start()
prof = int(os.environ.get("PROFILE", "0"))
if prof:
    time.sleep(float(os.environ.get("DELAY", "2")))
    d = os.path.join(OUT, f"trace_bs{BS}")
    os.makedirs(d, exist_ok=True)
    post("/start_profile", {"output_dir": d, "activities": ["CPU", "GPU"], "num_steps": prof, "record_shapes": False}, 60)
for t in th:
    t.join()
wall = (time.perf_counter() - t0) * 1e3
rows = []
for i in range(BS):
    d, ms = res[i]
    m = d.get("meta_info", {})
    rows.append(dict(prompt=PROMPTS[i], text=d.get("text", ""), ids=d.get("output_ids"), ntok=m.get("completion_tokens"), ms=ms))
json.dump(rows, open(os.path.join(OUT, f"answers_bs{BS}.json"), "w"), indent=1)
tok = sum(r["ntok"] or 0 for r in rows)
print(f"BS {BS}: {tok} tokens in {wall:.0f} ms = {tok / wall * 1e3:.1f} tok/s | per request median "
      f"{st.median(r['ms'] / max(1, r['ntok'] or 1) for r in rows):.3f} ms/token{' | profiled' if prof else ''}", flush=True)
