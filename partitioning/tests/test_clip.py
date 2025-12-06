import torch
from torch import nn
from PIL import Image
import requests
import torchvision.transforms as transforms
from ..partition_graph import partition_graph, generate_all_augmented_kernels

# Control which model to use
MODEL_TYPE = "transformers_clip_text"  # Options: "transformers_clip", "transformers_clip_text"   

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define unsupported and ignore operations
UNSUPPORTED_OPS = set([
    'Abs',
    'Concat',
    'Equal',
    'Expand',
    'Gather',
    'Pow', 
    'Reshape',
    'Shape',
    'Slice',
    'Sqrt',
    'Transpose', 
    'Trilu',
    'Unsqueeze',
    'Where',
    'Tanh',
])

COMPOSITE_OPS = {
    "Sigmoid": 4,
    "Softmax": 3,
    "ReduceMean": 2
}

IGNORE_OPS = set([
    "Cast",
    "CastLike",
    "Constant",
    "Identity",
    "Dropout"
])

class ExportWrapper(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        self.model = model
        self.model_type = model_type
    
    def forward(self, *args):
        if self.model_type == "text" and len(args) == 2:
            # Text model with input_ids and attention_mask
            return self.model(args[0], args[1])
        else:
            # Vision model with pixel_values
            return self.model(args[0])

def prepare_transformers_clip():
    """Prepare CLIP vision model from transformers"""
    from transformers import CLIPModel
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # model.to(device)
    
    vision_model = model.vision_model
    # vision_model.to(device)
    
    wrapped_model = ExportWrapper(vision_model, "vision")
    
    # Input configs: (batch_size, channels, height, width) 
    # CLIP vision expects (batch_size, 3, 224, 224)
    input_configs = [
        (1,),
        (2,),
        (4,),
        (8,),
        (16,),
    ]
    
    return wrapped_model, input_configs, "clip_vision"

def prepare_transformers_clip_text():
    """Prepare CLIP text model from transformers"""
    from transformers import CLIPModel
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # model.to(device)
    
    text_model = model.text_model
    # text_model.to(device)
    
    wrapped_model = ExportWrapper(text_model, "text")
    
    # Input configs: (batch_size, sequence_length)
    input_configs = [
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 77),  # Max sequence length for CLIP
        (2, 16),
        (2, 32),
        (2, 77),
        (4, 16),
        (4, 32),
        (8, 16),
    ]
    
    return wrapped_model, input_configs, "clip_text"

# Main execution
try:
    if MODEL_TYPE == "transformers_clip":
        wrapped_model, input_configs, dataset_name = prepare_transformers_clip()
    elif MODEL_TYPE == "transformers_clip_text":
        wrapped_model, input_configs, dataset_name = prepare_transformers_clip_text()
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    # Generate augmented kernels
    generate_all_augmented_kernels(
        input_configs=input_configs,
        model=wrapped_model,
        root_dir="/home/ayushkum/ayushkum/mirage/scripts/dataset",
        dataset_name=dataset_name,
        min_num_ops=2,
        max_num_ops=4,
        aug_factor=5,
        UNSUPPORTED_OPS=UNSUPPORTED_OPS,
        COMPOSITE_OPS=COMPOSITE_OPS,
        IGNORE_OPS=IGNORE_OPS
    )
    
    print(f"Successfully generated kernels for {dataset_name}")
    
except Exception as e:
    print(f"Error processing {MODEL_TYPE}: {e}")
    import traceback
    traceback.print_exc()