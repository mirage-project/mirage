import torch
import onnx
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, ViTForImageClassification
from torchvision.models import vit_b_16, swin_v2_b, convnext_base
from build_computation_graph import parse_onnx_model


# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def test_vision_transformer():
    """Test a Vision Transformer (ViT) model"""
    print("\n=== Testing Vision Transformer ===")
    
    model = vit_b_16(pretrained=True)
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    onnx_path = "vit_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=15
    )
    
    print(f"ViT model exported to {onnx_path}")
    
    onnx_model = onnx.load(onnx_path)
    parse_onnx_model(onnx_model)


def test_convnext():
    """Test a ConvNeXt model"""
    print("\n=== Testing ConvNeXt ===")
    
    model = convnext_base(pretrained=True)
    model.eval()
    model.to(device)
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    onnx_path = "convnext_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=15
    )
    
    print(f"ConvNeXt model exported to {onnx_path}")
    
    onnx_model = onnx.load(onnx_path)
    parse_onnx_model(onnx_model)


# def test_stable_diffusion_unet():
#     """Test a Stable Diffusion UNet component"""
#     print("\n=== Testing Stable Diffusion UNet ===")
    
#     try:
#         model = UNet2DConditionModel.from_pretrained(
#             "CompVis/stable-diffusion-v1-4", 
#             subfolder="unet"
#         )
#         model.eval()
#         model.to(device)
        
#         # Create dummy inputs
#         batch_size = 1
#         height = 64  # Using smaller dimensions for testing
#         width = 64
#         channels = 4
        
#         sample = torch.randn(batch_size, channels, height, width, device=device)
#         timestep = torch.tensor([999], device=device)
#         encoder_hidden_states = torch.randn(batch_size, 77, 768, device=device)
        
#         # Export to ONNX
#         onnx_path = "sd_unet_model.onnx"
#         torch.onnx.export(
#             model,
#             (sample, timestep, encoder_hidden_states),
#             onnx_path,
#             input_names=["sample", "timestep", "encoder_hidden_states"],
#             output_names=["output"],
#             dynamic_axes={
#                 "sample": {0: "batch_size"},
#                 "encoder_hidden_states": {0: "batch_size"},
#                 "output": {0: "batch_size"}
#             },
#             opset_version=15
#         )
        
#         print(f"SD UNet model exported to {onnx_path}")
        
#         # Load and analyze the model
#         onnx_model = onnx.load(onnx_path)
#         parse_onnx_model(onnx_model)
#     except Exception as e:
#         print(f"Error testing Stable Diffusion UNet: {e}")


if __name__ == "__main__":
    # Run tests for each model
    try:
        test_vision_transformer()
    except Exception as e:
        print(f"Error in ViT test: {e}")
    
    
    try:
        test_convnext()
    except Exception as e:
        print(f"Error in ConvNeXt test: {e}")
    
