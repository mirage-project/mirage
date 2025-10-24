import openai
import json
import os
import random
from pydantic import BaseModel

client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://ai-gateway.andrew.cmu.edu/"  # LiteLLM Proxy is OpenAI compatible
)

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
- If uncertain, output your best numeric estimate.

FORMAT:
'execution_time': <float>
"""

USER_TEMPLATE = """
Predict execution time (in seconds) after Mirage superoptimization for this tensor program JSON:

{subgraph_json}

Return ONLY: 'execution_time': <float>
"""

# ---- Helpers ----

def read_repo_to_string(root_path, include_ext=None, exclude_dirs=None):
    include_ext = include_ext or {'.cc', '.h', '.cu', '.md', '.txt', '.json'}
    exclude_dirs = set(exclude_dirs or {'.git', '__pycache__'})
    repo_text = []

    for dirpath, dirs, files in os.walk(root_path):
        # Skip excluded directories
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
        hash_to_time |= json.load(open(os.path.join(root, f), 'r'))
    all_orig_files = [f for f in os.listdir(root) if f.endswith('.json') and f.startswith('original_')]
    original_files = []
    optimized_files = []
    execution_times = []
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
        examples_list.append(f"EXAMPLE {i} EXECUTION TIME: {exec_time}")  # <-- now inside loop

    examples = '\n'.join(examples_list)
    system_prompt = SYSTEM_TEMPLATE.format(mirage_code=mirage_code, examples=examples)
    return system_prompt

def get_prediction(system_prompt, test_original, dataset_root):
    with open(os.path.join(dataset_root, test_original), "r", encoding="utf-8") as f:
        subgraph_json = f.read()

    user_msg = USER_TEMPLATE.format(subgraph_json=subgraph_json)

    response = client.chat.completions.parse(
        model='llama3-2-11b-instruct',
        messages=[
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_msg
            },
        ],
        temperature=0,
        # top_p=1,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "execution_time_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "execution_time": {"type": "number"}
                    },
                    "required": ["execution_time"],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
        seed=7,
    )
    response_data = json.loads(response.choices[0].message.content)

    return response_data

def run_pipeline(dataset_root, mirage_root, scale=1.0, train=0.8, shuffle=True):
    train_set, test_set = train_test_split(dataset_root, scale, train, shuffle)
    system_prompt = construct_prompt(train_set, dataset_root, mirage_root)

    print(f"Training Set Size: {len(train_set)}")
    print(f"Testing Set Size: {len(test_set)}")

    test_results = []
    for idx, data in enumerate(test_set):
        print(f'testing {idx}')
        test_original, _, exec_time = data
        pred_exec_time = None
        for i in range(100):
            try:
                output = get_prediction(system_prompt, test_original, dataset_root)
                pred_exec_time = output["execution_time"]
                break
            except Exception as e:
                print(f'attempt {i} failed with {e}')

        if pred_exec_time is None:
            # Fallback to a sentinel if all attempts fail, to keep the script running
            pred_exec_time = float('nan')
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
mirage_root = "/home/kitao/projects/mirage/src/search/"

run_pipeline(dataset_root, mirage_root, scale=0.5, train=0.5)
