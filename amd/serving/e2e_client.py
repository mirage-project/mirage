"""End-to-end client for one SGLang server run: 8 prompts, one at a time (batch 1), greedy, NTOK new tokens each. Per prompt: the
wall time from the request to the full answer, the new tokens, the speculative accept length; the answers (text + token ids) go to
OUT/answers.json for compare_answers.py. python e2e_client.py OUT_DIR [NTOK] [PORT]"""
import json, statistics as st, sys, time, urllib.request

OUT = sys.argv[1]; NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 128; PORT = sys.argv[3] if len(sys.argv) > 3 else "30000"
PROMPTS = ["The history of the Roman Empire begins with", "def quicksort(arr):", "Explain how a transformer language model works.",
           "Write a short poem about the ocean at night.", "The three laws of thermodynamics are",
           "Translate to French: The weather is nice today and we are going to the park.",
           "Q: What is 17 times 23? A:", "Once upon a time in a small village in the mountains,"]


def gen(text, n):
    body = json.dumps({"text": text, "sampling_params": {"max_new_tokens": n, "temperature": 0}}).encode()
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); d = json.load(urllib.request.urlopen(rq, timeout=900)); return d, (time.perf_counter() - t0) * 1e3


for p in PROMPTS[:2]:   # warm-up
    gen(p, 16)
rows = []
for i, p in enumerate(PROMPTS):
    d, ms = gen(p, NTOK); m = d.get("meta_info", {})
    rows.append(dict(prompt=p, text=d.get("text", ""), ids=d.get("output_ids"), ntok=m.get("completion_tokens"),
                     accept=m.get("spec_accept_length"), ms=ms))
    print(f"E2E prompt {i}: {ms:7.1f} ms for {m.get('completion_tokens')} tokens = {ms / max(1, m.get('completion_tokens') or 1):.3f} "
          f"ms/token | accept {m.get('spec_accept_length')} | {json.dumps(d.get('text', '')[:60])}", flush=True)
json.dump(rows, open(f"{OUT}/answers.json", "w"), indent=1)
tok = sum(r["ntok"] for r in rows); ms = sum(r["ms"] for r in rows)
print(f"E2E total: {ms:.0f} ms for {tok} tokens = {ms / tok:.3f} ms/token | median per prompt {st.median(r['ms'] for r in rows):.1f} ms "
      f"| accept median {st.median(r['accept'] or 0 for r in rows):.3f}", flush=True)
