import json
import os
import random
import ollama
import re
import ast

# ---- Prompt templates ----

SYSTEM_TEMPLATE = """
You are an expert in GPU kernel optimization and performance analysis.

TASK: Given a single tensor program JSON, predict its execution time (in seconds) after Mirage superoptimization.

CONTEXT (may be long, for reasoning only):
- Hardware: NVIDIA RTX A5000 (24GB), 27.8 TFLOPS, 222.2 TFLOPS Tensor, 64KB/SM.
- Analysis factors: compute intensity, bandwidth, parallelism, fusion, access patterns, cache efficiency.
- Mirage source (search over thread/block/kernel configs):
{mirage_code}
- Calibrating examples (original → optimized → measured time):
{examples}

STRICT OUTPUT CONTRACT:
- Return a **single JSON object** with exactly one key: "execution_time".
- The value MUST be a number (float). No strings, no text, no units.
- DO NOT include any other keys, text, code fences, or explanations.
- If uncertain, output your best numeric estimate

FORMAT:
{{'execution_time': <float>}}
"""

USER_TEMPLATE = """
Predict execution time (in seconds) after Mirage superoptimization for this tensor program JSON:

{subgraph_json}

Return ONLY: {{'execution_time': <float>}}
"""

# ---- Helpers ----

def read_repo_to_string(root_path, include_ext=None, exclude_dirs=None):
    include_ext = include_ext or {'.cc', '.h', '.cu', '.md', '.txt', '.json'}
    exclude_dirs = set(exclude_dirs or {'.git', '__pycache__'})
    repo_text = []

    for dirpath, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext.lower() in include_ext:
                file_path = os.path.join(dirpath, fname)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    rel_path = os.path.relpath(file_path, root_path)
                    repo_text.append(f"\n--- FILE: {rel_path} ---\n{content}")
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
    return "\n".join(repo_text)

def train_test_split(root, scale=0.1, train=0.8, shuffle=True):
    hash_to_time = {}
    performance_files = [f for f in os.listdir(root) if "performance" in f]
    for f in performance_files:
        with open(os.path.join(root, f), 'r') as pf:
            hash_to_time |= json.load(pf)

    all_orig_files = [f for f in os.listdir(root) if f.endswith('.json') and f.startswith('original_')]
    original_files, optimized_files, execution_times = [], [], []
    for orig_file in all_orig_files:
        h = orig_file[len('original_'):-len('.json')]
        if h in hash_to_time:
            original_files.append(orig_file)
            optimized_files.append(os.path.join(root, 'optimized_' + str(h) + '.json'))
            execution_times.append(hash_to_time[h])

    combined = list(zip(original_files, optimized_files, execution_times))
    if shuffle:
        random.shuffle(combined)

    scale_idx = round(len(combined) * scale)
    combined = combined[:scale_idx]

    split_idx = round(len(combined) * train)
    train_set = combined[:split_idx]
    test_set = combined[split_idx:]
    return train_set, test_set

def construct_prompt(train_set, dataset_root, mirage_root):
    """Builds the big SYSTEM prompt content (Mirage code + examples)."""
    mirage_code = read_repo_to_string(mirage_root)

    examples_list = []
    for i, data in enumerate(train_set):
        original, optimized, exec_time = data
        with open(os.path.join(dataset_root, original), 'r', encoding='utf-8') as f:
            examples_list.append(f"\nORIGINAL EXAMPLE {i}\n{f.read()}")
        with open(os.path.join(dataset_root, optimized), 'r', encoding='utf-8') as f:
            examples_list.append(f"\nOPTIMIZED EXAMPLE {i}\n{f.read()}")
        examples_list.append(f"EXAMPLE {i} EXECUTION TIME: {exec_time}")

    examples = '\n'.join(examples_list)
    system_prompt = SYSTEM_TEMPLATE.format(mirage_code=mirage_code, examples=examples)
    return system_prompt

def parse_execution_time(content: str) -> float:
    """
    Extracts execution_time from a messy LLM response.
    Strategy:
      1) Find the last {...} block that looks like it contains "execution_time".
      2) Try json.loads; if that fails:
         - fix common issues (single quotes -> double quotes)
         - strip trailing ] } garbage
         - try json again
      3) As a last resort, regex out the number after the key.
    """
    # 1) Grab candidate object slice(s)
    #    This regex finds the smallest {...} chunk that contains execution_time
    #    and prefers the last occurrence (often the correct one).
    candidates = list(re.finditer(r"\{[^{}]*execution_time[^{}]*\}", content, flags=re.IGNORECASE))
    obj_text = candidates[-1].group(0) if candidates else None

    if obj_text:
        # 2a) Strict JSON first
        try:
            obj = json.loads(obj_text)
            if isinstance(obj, dict) and "execution_time" in obj and isinstance(obj["execution_time"], (int, float)):
                return float(obj["execution_time"])
        except json.JSONDecodeError:
            pass

        # 2b) Fix common issues and retry
        fixed = obj_text

        # replace single quotes with double quotes when they look like JSON keys/strings
        fixed = re.sub(r"'", '"', fixed)

        # remove trailing commas before closing braces
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

        try:
            obj = json.loads(fixed)
            if isinstance(obj, dict) and "execution_time" in obj and isinstance(obj["execution_time"], (int, float)):
                return float(obj["execution_time"])
        except json.JSONDecodeError:
            # 2c) As another fallback, try Python literal_eval (tolerates single quotes)
            try:
                obj = ast.literal_eval(obj_text)
                if isinstance(obj, dict) and "execution_time" in obj and isinstance(obj["execution_time"], (int, float)):
                    return float(obj["execution_time"])
            except Exception:
                pass

    # 3) Final fallback: regex a float after the key anywhere in content
    m = re.search(
        r'execution_time\s*["\']?\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        content,
        flags=re.IGNORECASE
    )
    if m:
        exec_time = float(m.group(1))
        
        # if output in milliseconds, convert to seconds
        if exec_time > 1.0:
            exec_time /= 1000
        
        return exec_time


    raise ValueError("Could not parse execution_time from model output")

def get_prediction(system_prompt, test_original, dataset_root,
                   model_name):
    with open(os.path.join(dataset_root, test_original), "r", encoding="utf-8") as f:
        subgraph_json = f.read()

    user_msg = USER_TEMPLATE.format(subgraph_json=subgraph_json)

    # Ollama chat call
    # We try JSON mode if supported by the model (`format='json'`).
    # If the model ignores it, we still parse robustly with _safe_extract_execution_time.
    resp = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        options={
            "temperature": 0.0,
            "top_k": 1,
            "seed": 7,
        },
        format="json"  # ask for JSON output; some models enforce it strictly
    )

    print(resp)

    # Ollama returns a dict with shape:
    # {"model": ..., "created_at": ..., "message": {"role": "assistant", "content": "..."} , ...}
    content = resp.get("message", {}).get("content", "")
    exec_time = parse_execution_time(content)
    return {"execution_time": exec_time}

def run_pipeline(dataset_root, mirage_root, model, scale=1.0, train=0.8, shuffle=True):
    train_set, test_set = train_test_split(dataset_root, scale, train, shuffle)
    system_prompt = construct_prompt(train_set, dataset_root, mirage_root)

    print(f"Training Set Size: {len(train_set)}")
    print(f"Testing Set Size: {len(test_set)}")

    test_results = []
    for idx, data in enumerate(test_set):
        print(f'testing {idx}')
        test_original, _, exec_time = data
        pred_exec_time = None
        for attempt in range(10):
            try:
                output = get_prediction(system_prompt, test_original, dataset_root, model_name=model_name)
                pred_exec_time = output["execution_time"]
                break
            except Exception as e:
                print(f'attempt {attempt} failed with {e}')

        if pred_exec_time is None:
            continue
        test_results.append({'predicted': pred_exec_time, 'actual': exec_time})

    # calculate pairwise difference score
    num_corrects = 0
    n = len(test_set)
    if n >= 2:
        for i, a in enumerate(test_results):
            for j, b in enumerate(test_results):
                if i == j:
                    continue
                a_gr_b_pred = (a['predicted'] > b['predicted'])
                a_gr_b_actual = (a['actual'] > b['actual'])
                if a_gr_b_pred == a_gr_b_actual:
                    num_corrects += 1
        print(f"Pairwise Comparison Accuracy: {num_corrects / ((n**2) - n)}")
    else:
        print("Not enough test samples for pairwise comparison accuracy.")

# ---- Paths & run ----
dataset_root = "/home/kitao/projects/mirage/dataset/09_29_25"
mirage_root = "/home/kitao/projects/mirage/src"

if __name__ == '__main__':
    run_pipeline(dataset_root, mirage_root, scale=0.5, train=0.5, model_name="gpt-oss:120b")