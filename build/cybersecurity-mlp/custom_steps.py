import numpy as np
import pkg_resources as pk
import os

from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity
import torch
import torch.nn as nn
import brevitas.onnx as bo
from brevitas.quant_tensor import QuantTensor

# Define export wrapper
class CybSecMLPForExport(nn.Module):
    def __init__(self, my_pretrained_model):
        super(CybSecMLPForExport, self).__init__()
        self.pretrained = my_pretrained_model
        self.qnt_output = QuantIdentity(
            quant_type='binary',
            scaling_impl_type='const',
            bit_width=1,
            min_val=-1.0,
            max_val=1.0
        )

    def forward(self, x):
        # assume x contains bipolar {-1,1} elems
        # shift from {-1,1} -> {0,1} since that is the
        # input range for the trained network
        x = (x + torch.tensor([1.0]).to("cpu")) / 2.0
        out_original = self.pretrained(x)
        out_final = self.qnt_output(out_original) # output as {-1, 1}
        return out_final

def custom_step_mlp_export(model_name):
    # Define model parameters
    input_size = 593
    hidden1 = 64
    hidden2 = 64
    hidden3 = 64
    weight_bit_width = 2
    act_bit_width = 2
    num_classes = 1

    # Create model definition from Brevitas library
    model = nn.Sequential(
        QuantLinear(input_size, hidden1, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden1),
        nn.Dropout(0.5),
        QuantReLU(act_bit_width=act_bit_width),
        QuantLinear(hidden1, hidden2, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden2),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden2, hidden3, bias=True, weight_bit_width=weight_bit_width),
        nn.BatchNorm1d(hidden3),
        nn.Dropout(0.5),
        QuantReLU(bit_width=act_bit_width),
        QuantLinear(hidden3, num_classes, bias=True, weight_bit_width=weight_bit_width)
    )

    # Load pre-trained weights
    assets_dir = pk.resource_filename("finn.qnn-data", "cybsec-mlp/")
    trained_state_dict = torch.load(assets_dir+"/state_dict.pth")["models_state_dict"][0]
    model.load_state_dict(trained_state_dict, strict=False)

    # Network surgery: pad input size from 593 to 600 and convert bipolar to binary
    W_orig = model[0].weight.data.detach().numpy()
    W_new = np.pad(W_orig, [(0, 0), (0, 7)])
    model[0].weight.data = torch.from_numpy(W_new)

    model_for_export = CybSecMLPForExport(model)

    # Create directory to save model
    os.makedirs("models", exist_ok=True)
    ready_model_filename = "models/%s.onnx" % model_name
    input_shape = (1, 600)

    # create a QuantTensor instance to mark input as bipolar during export
    input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
    input_a = 2 * input_a - 1
    scale = 1.0
    input_t = torch.from_numpy(input_a * scale)
    input_qt = QuantTensor(
        input_t, scale=torch.tensor(scale), bit_width=torch.tensor(1.0), signed=True
    )

    # Export to ONNX
    bo.export_finn_onnx(
        model_for_export, export_path=ready_model_filename, input_t=input_qt
    )

    return ready_model_filename

    