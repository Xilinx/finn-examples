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

from finn.builder.build_dataflow_config import DataflowBuildConfig
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
