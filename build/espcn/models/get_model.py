import brevitas_examples.super_resolution.models as models
import brevitas_examples.super_resolution.utils as utils
import os
import torch
import numpy as np
from brevitas.export import export_qonnx

model = models.get_model_by_name('quant_espcn_x2_w4a4_base', True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
_, testloader = utils.get_bsd300_dataloaders(
        "../data",
        num_workers=0,
        batch_size=1,
        upscale_factor=model.upscale_factor,
        crop_size=256,
        download=True)

os.makedirs("../quant_espcn_x2_w4a4_base", exist_ok=True)

inp = testloader.dataset[0][0].unsqueeze(0)  # NCHW
inp = torch.round(inp.to(device)*255)
model = model.to(device)
with open(f"../quant_espcn_x2_w4a4_base/input.npy", "wb") as f:
        np.save(f, inp.cpu().numpy())
with open(f"../quant_espcn_x2_w4a4_base/output.npy", "wb") as f:
        np.save(f, model(inp).detach().cpu().numpy())
print(f"Saved I/O to ../quant_espcn_x2_w4a4_base as numpy arrays")

export_qonnx(
    model.cpu(),
    input_t=inp.cpu(),
    export_path=f"../quant_espcn_x2_w4a4_base/qonnx_model.onnx",
    opset_version=13)
print(f"Saved QONNX model to ../quant_espcn_x2_w4a4_base/qonnx_model.onnx")
