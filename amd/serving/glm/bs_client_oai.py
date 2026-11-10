"""bs_client.py for an OpenAI-API server (vLLM): the same batch-size run - BS requests at once (one thread each), the same
prompts, greedy, NTOK new tokens each with the end-of-sequence token ignored - through POST /v1/completions. Prints the same
line (wall time, output tokens per second, median time per token per request); answers to OUT/answers_bs<BS>.json.
PROFILE=N: also a torch-profiler trace started DELAY s (default 2) after the requests and stopped PROFSEC s later (default 0.5;
vLLM's /start_profile and /stop_profile, traces under the server's --profiler-config torch_profiler_dir).
    python bs_client_oai.py OUT BS [NTOK] [PORT]      env: MODEL_NAME (default glm)"""
import json
import os
import statistics as st
import sys
import threading
import time
import urllib.request

OUT, BS = sys.argv[1], int(sys.argv[2])
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
PORT = sys.argv[4] if len(sys.argv) > 4 else "30001"
NAME = os.environ.get("MODEL_NAME", "glm")
BASE = ["The history of the Roman Empire begins with", "def quicksort(arr):", "Explain how a transformer language model works.",
        "Write a short poem about the ocean at night.", "The three laws of thermodynamics are",
        "Translate to French: The weather is nice today and we are going to the park.",
        "Q: What is 17 times 23? A:", "Once upon a time in a small village in the mountains,"]
PROMPTS = [f"Request {i}. {BASE[i % len(BASE)]}" for i in range(BS)]


def post(path, body, timeout=900):
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(rq, timeout=timeout).read()


def gen(i, res):
    t0 = time.perf_counter()
    d = json.loads(post("/v1/completions", {"model": NAME, "prompt": PROMPTS[i], "max_tokens": NTOK, "temperature": 0,
                                            "ignore_eos": True}))
    res[i] = (d, (time.perf_counter() - t0) * 1e3)


for i in range(min(BS, 2)):   # warm-up
    gen(i, {})
res = {}
th = [threading.Thread(target=gen, args=(i, res)) for i in range(BS)]
t0 = time.perf_counter()
for t in th:
    t.start()
if int(os.environ.get("PROFILE", "0")):
    time.sleep(float(os.environ.get("DELAY", "2")))
    post("/start_profile", {}, 120)
    time.sleep(float(os.environ.get("PROFSEC", "0.5")))
    post("/stop_profile", {}, 600)
for t in th:
    t.join()
wall = (time.perf_counter() - t0) * 1e3
rows = []
for i in range(BS):
    d, ms = res[i]
    ntok = d.get("usage", {}).get("completion_tokens")
    rows.append(dict(prompt=PROMPTS[i], text=d.get("choices", [{}])[0].get("text", ""), ntok=ntok, ms=ms))
json.dump(rows, open(os.path.join(OUT, f"answers_bs{BS}.json"), "w"), indent=1)
tok = sum(r["ntok"] or 0 for r in rows)
print(f"BS {BS}: {tok} tokens in {wall:.0f} ms = {tok / wall * 1e3:.1f} tok/s | per request median "
      f"{st.median(r['ms'] / max(1, r['ntok'] or 1) for r in rows):.3f} ms/token"
      f"{' | profiled' if os.environ.get('PROFILE', '0') != '0' else ''}", flush=True)
