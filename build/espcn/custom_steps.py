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

import torch
from brevitas.export import export_qonnx
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (  # ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.subpixel_to_deconv import SubPixelToDeconvolution
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.infer_data_layouts import InferDataLayouts

import finn.transformation.streamline.absorb as absorb
import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.reorder import MoveScalarMulPastConvTranspose
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.fpgadataflow.infer_pixel_padding_deconv import InferPixelPaddingDeconv

from finn.builder.build_dataflow_config import DataflowBuildConfig, VerificationStepType
from finn.builder.build_dataflow_steps import verify_step
from finn.util.pytorch import ToTensor

def custom_step_qonnx_tidy_up(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InferShapes())
    # QONNX transformations
    model = model.transform(SubPixelToDeconvolution())
    model = model.transform(InferShapes())
    return model


def custom_step_add_pre_proc(model: ModelWrapper, cfg: DataflowBuildConfig):
    # Add preprocessing node
    preproc = ToTensor()
    export_qonnx(preproc, torch.randn(1, 3, 128, 128), "preproc.onnx", opset_version=11)
    preproc_model = ModelWrapper("preproc.onnx")
    # set input finn datatype to UINT8
    preproc_model.set_tensor_datatype(preproc_model.graph.input[0].name, DataType["UINT8"])
    preproc_model = preproc_model.transform(InferShapes())
    preproc_model = preproc_model.transform(FoldConstants())
    preproc_model = preproc_model.transform(GiveUniqueNodeNames())
    preproc_model = preproc_model.transform(GiveUniqueParameterTensors())
    preproc_model = preproc_model.transform(GiveReadableTensorNames())
    preproc_model.save("preproc.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(MergeONNXModels(preproc_model))
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(RemoveUnusedTensors())

    return model

def custom_step_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Run streamlining on given model. Streamlining involves moving floating point
    scale/shift parameters around, collapsing adjacent ones into a single parameter,
    then absorbing the scale/shift into the following `MultiThreshold` node.
    Streamlining requires careful topology design and cannot be applied to all
    topologies.
    """

    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(Streamline())
    # breakpoint()
    model = model.transform(MoveScalarMulPastConvTranspose())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    need_lowering = len(model.get_nodes_by_op_type("Conv")) > 0
    if need_lowering:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
  
    model = model.transform(Streamline())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())

    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    return model


def custom_step_convert_to_hls(model: ModelWrapper, cfg: DataflowBuildConfig):
    """Convert eligible nodes to `HLSCustomOp` subclasses that represent HLS
    layers. Which nodes and particular configurations can be converted to HLS
    is limited, see the source code of the `convert_to_hls` module for more."""


    mem_mode = cfg.default_mem_mode.value
    if cfg.standalone_thresholds:
        # doing this first causes all threshold layers to be standalone
        model = model.transform(to_hls.InferThresholdingLayer())
    need_convtranspose = len(model.get_nodes_by_op_type("ConvTranspose")) > 0
    if need_convtranspose:
        model = model.transform(InferPixelPaddingDeconv())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(RoundAndClipThresholds())
    # needed for non-bipolar MatMul layers
    model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
    # input quantization (if any) as standalone threshold
    model = model.transform(to_hls.InferThresholdingLayer())
    # needed for convolutions -- TODO always exec?
    need_conv = len(model.get_nodes_by_op_type("Im2Col")) > 0
    if need_conv:
        if cfg.force_rtl_conv_inp_gen:
            model = model.transform(to_hls.InferConvInpGen(use_rtl_variant=True))
        else:
            model = model.transform(to_hls.InferConvInpGen())
    # get rid of Tranpose -> Tranpose identity seq
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    return model
