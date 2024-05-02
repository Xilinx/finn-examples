import json
import numpy as np
import pkg_resources as pk
import os

from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity
import torch
import torch.nn as nn
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.util.cleanup import qonnx_cleanup
from brevitas.export import export_qonnx


# Define export wrapper
class ExportModel(nn.Module):
    def __init__(self, my_pretrained_model):
        super(ExportModel, self).__init__()
        self.pretrained = my_pretrained_model
        self.qnt_output = QuantIdentity(
            quant_type="binary", scaling_impl_type="const", bit_width=1, min_val=-1.0, max_val=1.0
        )

    def forward(self, x):
        # assume x contains bipolar {-1,1} elems
        # shift from {-1,1} -> {0,1} since that is the
        # input range for the trained network
        x = (x + torch.tensor([1.0]).to(x.device)) / 2.0
        out_original = self.pretrained(x)
        out_final = self.qnt_output(out_original)  # output as {-1,1}
        return out_final


def custom_step_mlp_export(model_name):
    # load trained model assets
    assets_dir = pk.resource_filename("finn.qnn-data", "ddos-anomaly-detector-mlp/")
    trained_state_dict = torch.load(assets_dir + "/state_dict.pth")["models_state_dict"][0]
    with open(assets_dir + "/metadata.json", "r") as fp:
        metadata = json.load(fp)

    # Define model parameters
    input_size = metadata["total_in_bitwidth"]
    hidden1 = 32
    hidden2 = 32
    weight_bit_width = 2
    act_bit_width = 2
    num_classes = metadata["total_out_bitwidth"]

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
        QuantLinear(hidden2, num_classes, bias=True, weight_bit_width=weight_bit_width),
    )

    # Load pre-trained weights
    model.load_state_dict(trained_state_dict, strict=True)

    # Network surgery: convert inputs to bipolar
    model_for_export = ExportModel(model)

    # Create directory to save model
    os.makedirs("models", exist_ok=True)
    ready_model_filename = "models/%s.onnx" % model_name
    input_shape = (1, metadata["total_in_bitwidth"])

    # create bipolar input
    input_a = np.random.randint(0, 1, size=input_shape).astype(np.float32)
    input_a = 2 * input_a - 1
    scale = 1.0
    input_t = torch.from_numpy(input_a * scale)

    # Export to ONNX
    export_qonnx(model_for_export, export_path=ready_model_filename, input_t=input_t)

    # Clean-up
    qonnx_cleanup(ready_model_filename, out_file=ready_model_filename)

    # Setting the input datatype explicitly because it doesn't get derived from the export function
    model_for_export = ModelWrapper(ready_model_filename)
    model_for_export.set_tensor_datatype(model_for_export.graph.input[0].name, DataType["BIPOLAR"])

    # ConvertQONNXtoFINN() transformation
    model_for_export = model_for_export.transform(ConvertQONNXtoFINN())
    model_for_export.save(ready_model_filename)

    return ready_model_filename
