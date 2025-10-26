import json
import os
import random
import ollama
import re
import ast
import sys

from collections import Counter, deque
from typing import List, Tuple, Dict, Iterable, Any

try:
    from cost_model.in_ctx_exec_time import read_repo_to_string, train_test_split
except ImportError:
    from in_ctx_exec_time import read_repo_to_string, train_test_split

SYSTEM_TEMPLATE = """
You are an in GPU kernel optimization and performance analysis.

TASK: Given a computational graph defined by list of operators and connectivities, return a partitioning of the graph into sections consisting of 1-{max_nodes_per_partition} operators such that after we perform Mirage superoptimization on each of the sections, the overall execution time of the computational graph is minimized.

PARTITIONING CRITERIA:
- Each section must have between 1 and {max_nodes_per_partition} operators.
- all operators in a section must be connected.

COMPUTATIONAL GRAPH FORMAT:
The computational graph is in the form of a list of operators, for each operator, the following fields are provided:
- NAME: the name of the operator (unique).
- TYPE: the type of the operator.
- IN_OPS: the names of the operators whose outputs make up the inputs to the operator.
- OUT_OPS: the names of the operators that uses the outputs of the operator.
- IN_TENSORS: the input tensors represented as a tuple where the first element is the size of the tensor and the second element is the tensor ID (unique to each tensor).
- OUT_TENSOR: the output tensors represented in the same way as the input tensors.
From this format, you should reconstruct the computational graph and evaluate it before any further analysis.

CONTEXT (may be long, for reasoning only):
- Hardware: NVIDIA RTX A5000 (24GB), 27.8 TFLOPS, 222.2 TFLOPS Tensor, 64KB/SM.
- Analysis factors: compute intensity, bandwidth, parallelism, fusion, access patterns, cache efficiency.
- Mirage source (search over thread/block/kernel configs):
{mirage_code}
- Calibrating examples showing the original subgraphs that were fed into Mirage, :
{examples}

STRICT OUTPUT CONTRACT:
- Return a **single JSON object** with exactly one key: 'sections' which contains a **list of tuples of strings** where each tuple represents a section and elements in the tuple are the names of the nodes in that tuple.
- The output must include ALL nodes. Please ensure this by counting the number of nodes in the output before returning the result.
- The output must not contain duplicates. Please ensure this by checking that no node name appears more than once in the output before returning the result.
- DO NOT include any other text, code fences, or explanations.
- If uncertain, output your best estimate.

FORMAT:
{{'sections': [('node_OPNAME_ID1', 'node_OPNAME_ID2'), ('node_OPNAME_ID1', 'node_OPNAME_ID2', 'node_OPNAME_ID3'), ...]}}
"""

USER_TEMPLATE = """
Partition the following computational graph into sections of 1 to {max_nodes_per_partition} connected operators such that after after we perform Mirage superoptimization on each of the sections, the overall execution time of the computational graph is minimized.

{comp_graph}

Return ONLY: {{'sections': [('node_OPNAME_ID1', 'node_OPNAME_ID2'), ('node_OPNAME_ID1', 'node_OPNAME_ID2', 'node_OPNAME_ID3'), ...]}}
"""

def parse_sections(resp):
    # 1) get the content string
    if isinstance(resp, dict):
        content = resp.get("message", {}).get("content", "")
    else:
        # e.g., a dataclass/object with .message.content
        msg = getattr(resp, "message", None)
        content = getattr(msg, "content", "") if msg else ""

    # 2) sometimes content is already parsed (rare)
    if isinstance(content, (dict, list)):
        data = content
    else:
        s = content.strip()
        # remove code fences if present
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_+-]*\n|\n```$", "", s)
        # 3) parse JSON
        data = json.loads(s)

    # 4) normalize sections to tuples (optional)
    if "sections" in data and isinstance(data["sections"], list):
        data["sections"] = [tuple(section) for section in data["sections"]]
    return data

def _op_names(lst: Iterable):
    """Return operator names whether the list contains Operator objects or strings."""
    names = []
    for x in lst:
        if isinstance(x, str):
            names.append(x)
        else:
            # assume Operator-like object with .name
            names.append(x.name)
    return names

def verify_partition(operators: List[Any],
                     sections: List[Tuple[str, ...]],
                     raise_on_error: bool = False):
    """
    Verify:
      1) each operator appears exactly once across all sections
      2) within each section, the operators form a connected subgraph

    Returns:
      (ok: bool, errors: List[str])
    """
    errors: List[str] = []

    # Map names -> Operator
    name_to_op: Dict[str, Operator] = {op.name: op for op in operators}
    all_names = set(name_to_op.keys())

    # ---- Rule 1: each operator exactly once ----
    flat = [name for sec in sections for name in sec]
    counts = Counter(flat)
    seen = set(counts.keys())

    missing = sorted(all_names - seen)
    extra = sorted(seen - all_names)
    dups  = sorted([n for n, c in counts.items() if c > 1])

    if missing:
        errors.append(f"Missing operators (not present in any section): {missing}")
    if extra:
        errors.append(f"Unknown operators (not in graph): {extra}")
    if dups:
        errors.append(f"Duplicate operators (appear more than once): {dups}")

    # If there are already fatal coverage issues, we can still continue to report connectivity,
    # but connectivity results might be noisy for unknown nodes.

    # ---- Precompute undirected neighbors from the graph ----
    # neighbors[name] = set of adjacent operator names
    neighbors: Dict[str, set] = {n: set() for n in all_names}
    for op in operators:
        in_names  = _op_names(op.input_ops)
        out_names = _op_names(op.output_ops)
        for nb in in_names + out_names:
            if nb in all_names:
                neighbors[op.name].add(nb)
                neighbors[nb].add(op.name)  # undirected

    # ---- Rule 2: each section forms a connected subgraph ----
    for i, sec in enumerate(sections):
        sec_set = set(sec)

        # Skip connectivity if section contains unknown nodes (already reported above)
        if not sec_set.issubset(all_names):
            continue

        if len(sec_set) <= 1:
            # singletons are trivially connected
            continue

        # BFS/DFS on induced subgraph
        start = next(iter(sec_set))
        visited = set([start])
        q = deque([start])

        while q:
            cur = q.popleft()
            for nb in neighbors.get(cur, ()):
                if nb in sec_set and nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        if visited != sec_set:
            unreachable = sorted(sec_set - visited)
            errors.append(
                f"Section {i} is not connected. Unreached nodes from '{start}': {unreachable}. "
                f"Section: {tuple(sec_set)}"
            )

    ok = len(errors) == 0
    if not ok and raise_on_error:
        raise ValueError("Partition verification failed:\n- " + "\n- ".join(errors))
    return ok, errors

def construct_prompt(train_set, dataset_root, mirage_root, max_nodes_per_partition=4):
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
    system_prompt = SYSTEM_TEMPLATE.format(mirage_code=mirage_code, examples=examples, max_nodes_per_partition=max_nodes_per_partition)
    return system_prompt

def get_sections_in_ctx(system_prompt, comp_graph, model_name, max_nodes_per_partition=4):
    if isinstance(comp_graph, dict):
        comp_graph = list(comp_graph.keys())

    user_msg = USER_TEMPLATE.format(comp_graph=comp_graph, max_nodes_per_partition=max_nodes_per_partition)

    print("Querying model for partitioning...")
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

    print("Model response received, parsing...")
    output = parse_sections(resp)

    print("Verifying partition correctness...")
    # verify correctness
    if not (verify_partition(comp_graph, output)):
        raise ValueError("The returned partitioning is invalid.")
    print("Partition correctness verified.")
    return output

def run_pipeline(sections, dataset_root, mirage_root, model_name, scale=1.0):
    train_set, _ = train_test_split(dataset_root, train=1.0, scale=scale)
    system_prompt = construct_prompt(train_set, dataset_root, mirage_root)

    sections_partitions = []
    for idx, section in enumerate(sections):
        print(f"testing {idx} with {len(section)} operators")
        section_partitions = None
        for attempt in range(1):
            try:
                output = get_sections_in_ctx(system_prompt, section, model_name)
                print("verifying correctness...")
                if not verify_partition(list(section.keys()), output):
                    print(f"attempt {attempt} gave an invalid partitioning, retrying")
                    continue
                print("correctness verified")
                
                section_partitions = output["sections"]
                break
            except Exception as e:
                print(f"attempt {attempt} failed with {e}")
        if sections_partitions is not None:
            sections_partitions.append(section_partitions)
    
    return sections_partitions

class InCtxPartitioner:
    def __init__(self, dataset_root, mirage_root, model_name, max_nodes_per_partition, scale):
        self.dataset_root = dataset_root
        self.mirage_root = mirage_root
        self.model_name = model_name
        self.max_nodes_per_partition = max_nodes_per_partition
        self.scale = scale
        self.train_set, _ = train_test_split(dataset_root, train=1.0, scale=scale)
        self.system_prompt = construct_prompt(self.train_set, dataset_root, mirage_root, max_nodes_per_partition=max_nodes_per_partition)
    def partition(self, comp_graph):
        return get_sections_in_ctx(self.system_prompt, comp_graph, self.model_name, self.max_nodes_per_partition)["sections"]

if __name__ == "__main__":
    import torch
    from torch import nn
    from torchtune.models import qwen2_5

    # Get the absolute path of the parent directory (where file3.py is)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Add it to sys.path if not already there
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from build_computation_graph import get_computation_graph
    from graph_splitter import process_operator_graph
    from op import Operator

    model = qwen2_5.qwen2_5_0_5b()

    dummy_input_tokens = torch.ones(2, 2, dtype=int)

    UNSUPPORTED_OPS = set(['Abs', 'Concat', 'Equal', 'Expand', 'Gather', 'Reshape', 'Shape', 'Slice', 'Transpose', 'Trilu', 'Unsqueeze', 'Where', 'Tanh'])
    IGNORE_OPS = set(['Cast', 'CastLike', 'Constant', 'Identity', 'Dropout'])

    class ExportWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids):
            return self.model(input_ids)

    # Use the wrapper for export
    wrapped_model = ExportWrapper(model)
    unique_operators = {}
    operators = get_computation_graph(model, dummy_input_tokens, unique_operators, "onnx")
    sections, deps, sorted_ops = process_operator_graph(operators, IGNORE_OPS, UNSUPPORTED_OPS)

    all_mirage_sections = []
    for section, section_type in sections:
        if section_type == "mirage":
            all_mirage_sections.append(section)

    dataset_root = "/home/kitao/projects/mirage/dataset/09_29_25"
    mirage_root = "/home/kitao/projects/mirage/src/search/"
    run_pipeline(all_mirage_sections, dataset_root, mirage_root, model_name="gpt-oss:120b")