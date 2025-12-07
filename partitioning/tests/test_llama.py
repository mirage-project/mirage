import torch
from torch import nn
import torch.nn.functional as F
from torchtune.models import llama3_2
from ..partition_graph import generate_all_augmented_kernels
import mirage as mi
import time

model = llama3_2.llama3_2_1b()

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
#     (1, 8),   
#     (2, 16),   
#     (4, 32),   
#     (4, 64),   
#     (8, 32),
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
    
    (1, 4), (1, 256),
    
    (3, 12), (5, 20), (7, 24),
]

generate_all_augmented_kernels(
    input_configs=input_configs,
    model=wrapped_model,
    root_dir="dataset/",
    dataset_name="llama",
    min_num_ops=2,
    max_num_ops=4,
    aug_factor=5,
    UNSUPPORTED_OPS=UNSUPPORTED_OPS,
    COMPOSITE_OPS=COMPOSITE_OPS,
    IGNORE_OPS=IGNORE_OPS,
    timeout_minutes=1
)