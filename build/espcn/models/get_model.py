# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import brevitas_examples.super_resolution.models as models
import brevitas_examples.super_resolution.utils as utils
import os
import torch
import numpy as np
from brevitas.export import export_qonnx

# model = models.get_model_by_name("quant_espcn_x2_w4a4_base", True)
model = models.quant_espcn_nnrc(upscale_factor=2, weight_bit_width=4, act_bit_width=4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
_, testloader = utils.get_bsd300_dataloaders(
    "../data",
    num_workers=0,
    batch_size=1,
    upscale_factor=model.upscale_factor,
    crop_size=256,
    download=True,
)

FINN_ROOT = os.getenv("FINN_ROOT")
path = os.path.realpath(FINN_ROOT + "/../espcn/quant_espcn_x2_w4a4_base/")
os.makedirs(path, exist_ok=True)

inp = testloader.dataset[0][0].unsqueeze(0).to(device)  # NCHW
model = model.to(device)
with open(path + "/input.npy", "wb") as f:
    np.save(f, torch.round(inp * 255).cpu().numpy())
with open(path + "/output.npy", "wb") as f:
    np.save(f, model(inp).detach().cpu().numpy())
print("Saved I/O to " + path + " as numpy arrays")
export_qonnx(
    model.cpu(), input_t=inp.cpu(), export_path=path + "/qonnx_model.onnx", opset_version=13
)
print("Saved QONNX model to" + path)
