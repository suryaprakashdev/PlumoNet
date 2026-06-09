import os
import torch
from unet3d import UNet3D
from resnet3d import ResNet3D10

def export_unet():
    print("Exporting UNet3D to ONNX...")
    device = torch.device("cpu")
    model = UNet3D().to(device)
    model.load_state_dict(torch.load("checkpoints/unet3d_best.pth", map_location=device, weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 1, 64, 64, 64, device=device)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        "checkpoints/unet3d.onnx", 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Saved checkpoints/unet3d.onnx")

def export_resnet():
    print("Exporting ResNet3D10 to ONNX...")
    device = torch.device("cpu")
    model = ResNet3D10().to(device)
    model.load_state_dict(torch.load("checkpoints/resnet3d_calibrated.pth", map_location=device, weights_only=True))
    model.eval()

    # The forward pass of ResNet3D10 gives unscaled logits, but the original code has `forward_scaled(x) -> logits / self.temperature`.
    # Let's export the forward_scaled directly so ONNX output is scaled automatically.
    
    # We can create a wrapper module
    class ResNetWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            return self.model.forward_scaled(x)
            
    wrapper = ResNetWrapper(model)
    wrapper.eval()
    
    dummy_input = torch.randn(1, 1, 64, 64, 64, device=device)
    
    torch.onnx.export(
        wrapper, 
        dummy_input, 
        "checkpoints/resnet3d.onnx", 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Saved checkpoints/resnet3d.onnx")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    export_unet()
    export_resnet()
