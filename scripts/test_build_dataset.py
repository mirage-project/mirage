import torch
from torch import nn
import torch.nn.functional as F
from torchtune.models import gemma
from partition_graph import partition_graph, to_kernel_graph
from build_computation_graph import get_computation_graph
from build_dataset import partition_graph_with_sampling

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = gemma.gemma_2b().to(device)

dummy_input_tokens = torch.ones(1, 1, dtype=int).to(device)

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
 'Where'
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
# operators = get_computation_graph(wrapped_model, dummy_input_tokens, unique_operators, "onnx")
all_subgraphs, unique_operators = partition_graph_with_sampling(wrapped_model, dummy_input_tokens, 
                    min_num_ops=2,
                    max_num_ops=4, 
                    UNSUPPORTED_OPS=UNSUPPORTED_OPS,
                    COMPOSITE_OPS=COMPOSITE_OPS,
                    IGNORE_OPS=IGNORE_OPS)


# for i, subgraph in enumerate(all_subgraphs):
#     print(f"transcribing subgraph {i}")
#     to_kernel_graph(subgraph)

print(f"Total Subgraphs Collected: {len(all_subgraphs)}")
