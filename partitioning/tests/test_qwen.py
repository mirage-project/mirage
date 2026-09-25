import torch
from torch import nn
import torch.nn.functional as F
from torchtune.models import qwen2_5
from ..partition_graph import partition_graph, to_kernel_graph, generate_all_augmented_kernels, partition_graph_with_sampling
import mirage as mi
import time

# graph = mi.new_kernel_graph()
# graph.from_json("/home/kitao/projects/mirage/dataset/qwen/original_1382668400313214164.json")
# graph.superoptimize()

model = qwen2_5.qwen2_5_0_5b()

dummy_input_tokens = torch.ones(2, 2, dtype=int)

UNSUPPORTED_OPS=set(
[
'Abs',
#  'Add',
#  'Cast',
#  'CastLike',
 'Concat',
#  'Constant',
#  'Div',
#  'Dropout',
 'Equal',
 'Expand',
 'Gather',
#  'Identity',
#  'MatMul',
#  'Mul',
#  'Neg',
#  'Pow',
#  'Reciprocal',
#  'ReduceMean',
 'Reshape',
 'Shape',
#  'Sigmoid',
 'Slice',
#  'Softmax',
#  'Sqrt',
 'Transpose',
 'Trilu',
 'Unsqueeze',
 'Where',
 'Tanh',
]
)

COMPOSITE_OPS = {
    "Sigmoid": 4,
    "Softmax": 3,
    "ReduceMean": 2
}

IGNORE_OPS = set(
[
"Cast",
"CastLike",
"Constant",
"Identity",
"Dropout"
]
)

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids):
        return self.model(input_ids)


# Use the wrapper for export
wrapped_model = ExportWrapper(model)

# input_configs = [
#     (1, 4),
#     (2, 16),
#     (4, 16),
#     (8, 8),
#     (1, 8),    # Small batch, short sequence
#     (2, 16),   # Medium
#     (4, 32),   # Larger
#     (1, 64),   # Long sequence
#     (8, 8),    
# ]

input_configs = [
    # Small safe configs
    (1, 8), (1, 16), (1, 32), (1, 64), (1, 128),
    (2, 8), (2, 16), (2, 32), (2, 64),
    (4, 8), (4, 16), (4, 32),
    (8, 8), (8, 16), (8, 32),
    (16, 8), (16, 16),
    (32, 8),
    
    # Edge cases
    (1, 4), (1, 256),
    
    # Non-power-of-2
    (3, 12), (5, 20), (7, 24),
]
generate_all_augmented_kernels(
    input_configs=input_configs,
    model=wrapped_model,
    root_dir="dataset/",
    dataset_name="qwen",
    min_num_ops=2,
    max_num_ops=4,
    aug_factor=5,
    UNSUPPORTED_OPS=UNSUPPORTED_OPS,
    COMPOSITE_OPS=COMPOSITE_OPS,
    IGNORE_OPS=IGNORE_OPS
)

# all_hashes = set()
# all_unique_subgraphs = []

# for batch_size, seq_len in input_configs:
#     print(f"\nProcessing config: batch={batch_size}, seq={seq_len}")
#     dummy_input = torch.ones(batch_size, seq_len, dtype=int)
    
#     subgraphs, _ = partition_graph_with_sampling(
#         wrapped_model, dummy_input,
#         min_num_ops=2, max_num_ops=4,
#         augmentation_factor=5,
#         UNSUPPORTED_OPS=UNSUPPORTED_OPS,
#         COMPOSITE_OPS=COMPOSITE_OPS,
#         IGNORE_OPS=IGNORE_OPS
#     )
    
#     for subgraph in subgraphs:
#         try:
#             kernel_graph, _ = to_kernel_graph(subgraph)
#             graph_hash = kernel_graph.get_owner_independent_hash()
#             if graph_hash not in all_hashes:
#                 all_hashes.add(graph_hash)
#                 all_unique_subgraphs.append(subgraph)
#         except NotImplementedError:
#             continue
    
#     print(f"Total unique subgraphs so far: {len(all_hashes)}")
# generate_all_kernels(wrapped_model, dummy_input_tokens,
#                      "/home/kitao/projects/mirage/dataset/10_16_25",
#                      "qwen",
#                      min_num_ops=2,
#                      max_num_ops=4, 
#                      UNSUPPORTED_OPS=UNSUPPORTED_OPS,
#                      COMPOSITE_OPS=COMPOSITE_OPS,
#                      IGNORE_OPS=IGNORE_OPS)
