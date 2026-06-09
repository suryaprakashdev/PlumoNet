# save_models.py
import bentoml
import torch
from unet3d import UNet3D
from resnet3d import ResNet3D10

# --- Save U-Net ---
unet = UNet3D()
unet.load_state_dict(torch.load("content/checkpoints/unet3d_best.pth", map_location="cpu"))
unet.eval()

bentoml.picklable_model.save_model(
    "lung_unet3d",
    unet,
    metadata={"task": "segmentation", "input_shape": "(1,1,64,64,64)"}
)

# --- Save ResNet (calibrated) ---
resnet = ResNet3D10()
resnet.load_state_dict(torch.load("content/checkpoints/resnet3d_calibrated.pth", map_location="cpu"))
resnet.eval()

bentoml.picklable_model.save_model(
    "lung_resnet3d",
    resnet,
    metadata={"task": "classification", "gradcam_target": "layer4"}
)

print("Done. Run: bentoml models list")