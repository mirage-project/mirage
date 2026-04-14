"""Random test matrix: 50 configs of random layer combos ± MTP on single GPU."""
import subprocess, random, sys, time, os

random.seed(42)
num_layers = 61  # DeepSeek V3
max_test_layers = 10
num_tests = 50
results = []
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "5")

for i in range(num_tests):
    # Random number of layers (1-10)
    n = random.randint(1, max_test_layers)
    # Random layer IDs from 0..60
    layers = sorted(random.sample(range(num_layers), n))
    layer_str = ",".join(str(l) for l in layers)
    # Random MTP on/off
    use_mtp = random.choice([True, False])
    # Random spec tokens (1-4) if MTP
    spec = random.randint(1, 4) if use_mtp else 0
    # Max seq length needs to fit prompt(14) + gen
    max_seq = 15 + (spec if use_mtp else 0)

    desc = f"layers=[{layer_str}] mtp={use_mtp}" + (f" spec={spec}" if use_mtp else "")
    
    cmd = [
        sys.executable, "-u",
        "/home/muhengl/mirage/demo/deepseek_v3/demo.py",
        "--model-path", "/raid/catalyst/models/DeepSeek-V3",
        "--use-mirage", "--correctness",
        "--layers", layer_str,
        "--max-seq-length", str(max_seq),
        "--max-num-pages", "4",
        "--max-num-batched-tokens", "1",
    ]
    if use_mtp:
        cmd += ["--mtp", "--num-speculative-tokens", str(spec)]
    
    env = dict(os.environ, MPK_NO_RESIDUAL="1", CUDA_VISIBLE_DEVICES=gpu)
    
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd="/dev/shm/mh_jit")
        elapsed = time.time() - t0
        # Parse result
        passed = "PASS" in out.stdout
        # Find cosine
        cosine = "?"
        for line in out.stdout.split("\n"):
            if "cosine=" in line or "cosine=+" in line:
                import re
                m = re.search(r'cosine=\+?([-\d.]+)', line)
                if m:
                    cosine = m.group(1)
        status = "PASS" if passed else "FAIL"
        if out.returncode != 0 and not passed:
            # Check for specific errors
            if "No space left" in out.stderr:
                status = "DISK_FULL"
            elif "already been declared" in out.stderr:
                status = "NVCC_DUP"
            elif "illegal" in out.stderr.lower():
                status = "CUDA_ERR"
            elif out.returncode == -9 or out.returncode == 137:
                status = "OOM"
            else:
                status = f"ERR({out.returncode})"
    except subprocess.TimeoutExpired:
        elapsed = 600
        status = "TIMEOUT"
        cosine = "?"
    
    results.append((i+1, desc, status, cosine, f"{elapsed:.1f}s"))
    print(f"[{i+1:2d}/50] {status:10s} cos={cosine:>8s} {elapsed:6.1f}s  {desc}", flush=True)

# Summary
print(f"\n{'='*60}")
passed = sum(1 for r in results if r[2] == "PASS")
failed = sum(1 for r in results if r[2] == "FAIL")
errors = sum(1 for r in results if r[2] not in ("PASS", "FAIL"))
print(f"PASS: {passed}  FAIL: {failed}  ERROR: {errors}  TOTAL: {len(results)}")
for r in results:
    if r[2] != "PASS":
        print(f"  [{r[0]:2d}] {r[2]:10s} cos={r[3]:>8s}  {r[1]}")
