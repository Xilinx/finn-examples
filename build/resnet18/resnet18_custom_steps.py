import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    ShellFlowType,
)

# Step: Attach Pre-Processing Model
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.infer_shapes import InferShapes
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.pytorch import ToTensor
from brevitas.export import export_qonnx

# Step: Streamlining
from qonnx.transformation.general import (
    ConvertDivToMul,
)
from finn.transformation.streamline.reorder import (
    MoveOpPastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarMulPastConv,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoConv,
)
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul
from finn.builder.build_dataflow_steps import VerificationStepType, verify_step
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.insert_topk import InsertTopK

# Step: Lowering Convolutions
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from finn.transformation.streamline.absorb import (
    AbsorbTransposeIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    AbsorbTransposeIntoFlatten,
)
from finn.transformation.streamline.reorder import (
    MoveTransposePastFork,
    MoveTransposePastJoinAdd,
)

# Step: Converting to HW Layers
from finn.transformation.fpgadataflow.convert_to_hw_layers import (
    InferAddStreamsLayer,
    InferPool,
    InferQuantizedMatrixVectorActivation,
    InferThresholdingLayer,
    InferConvInpGen,
    InferDuplicateStreamsLayer,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.absorb import AbsorbConsecutiveTransposes
from qonnx.core.datatype import DataType
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.general import (
    ApplyConfig,
    GiveUniqueNodeNames,
    SortGraph,
)
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from finn.transformation.streamline.absorb import AbsorbTransposeIntoFlatten
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten


def step_resnet18_attach_preproc(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # Make sure the input (and every other) node
    # has a shape attribute.
    model = model.transform(InferShapes())

    # Get the input shape of our model in the form of a tuple.
    shape = []
    for d in model.graph.input[0].type.tensor_type.shape.dim:
        shape.append(d.dim_value)
    shape = tuple(shape)

    # Take the Torch representation of our pre-processing model
    # (used to normalise input from 0-255 to 0-1), and convert it
    # to finn-onnx.
    pre_proc = export_qonnx(ToTensor(), input_shape=shape, opset_version=11)

    #  Wrap the pre-processing model in a QONNX ModelWrapper,
    # Then merge to the start of our model.
    pre_proc_qonnx = ModelWrapper(pre_proc)
    model = model.transform(MergeONNXModels(pre_proc_qonnx))
    
    # Clean up the model before returning.
    return cleanup_model(model)

def step_resnet18_streamline(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # A set of pre-existing steps we run to streamline our model.
    streamline_transformations = [
        MoveOpPastFork(['Mul']),
        MoveLinearPastEltwiseAdd(),
        ConvertDivToMul(),
        BatchNormToAffine(),
        MoveScalarMulPastConv(),
        MoveScalarLinearPastInvariants(),
        MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoConv(),
        RemoveIdentityOps(),
    ]

    # Insert a TopK node at the end of the model, in case there
    # are scalar add/mul nodes that can be absorbed there.
    model = model.transform(InsertTopK())

    # Run all streamlining steps.
    for t in streamline_transformations:
        model = model.transform(t)
    
    if VerificationStepType.STREAMLINED_PYTHON in cfg._resolve_verification_steps():
        verify_step(model, cfg, "streamlined_python", need_parent=False)

    # Clean up the model before returning.
    return cleanup_model(model)

def step_resnet18_lower(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # A set of pre-existing steps we run to lower
    # the convolutions our model.
    lower_transformations = [
        LowerConvsToMatMul(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        MoveTransposePastJoinAdd(),
        AbsorbTransposeIntoMultiThreshold(),
        MoveTransposePastFork(),
        AbsorbTransposeIntoFlatten(),
    ]

    # Run all streamlining steps.
    for t in lower_transformations:
        model = model.transform(t)

    # Clean up the model before returning.
    return cleanup_model(model)

# The set of steps we use to convert the layers in our model to HLS layers.
def step_resnet18_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    # A set of pre-existing steps we run to convert
    # all relevant layers in our model to HLS.
    to_hls_transformations = [
        DoubleToSingleFloat(),
        InferDataTypes(),
        SortGraph(),
        InferShapes(),
        InferAddStreamsLayer(),
        InferPool(),
        RoundAndClipThresholds(),
        InferThresholdingLayer(),
        InferQuantizedMatrixVectorActivation(),
        AbsorbConsecutiveTransposes(),
        InferConvInpGen(),
        InferDuplicateStreamsLayer(),
        AbsorbConsecutiveTransposes(),
        AbsorbTransposeIntoFlatten(),
        RemoveCNVtoFCFlatten(),
    ]
    
    # Workaround for an error. If it's not included, the first Im2Col nod
    # is not converted to an (FMPadding_Batch -> ConvolutionInputGenerator)
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    
    # Run all conversion steps.
    for t in to_hls_transformations:
        model = model.transform(InferDataLayouts())
        model = model.transform(t)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())
    
    # Clean up the model before returning.
    return cleanup_model(model)

def step_resnet18_slr_floorplan(model: ModelWrapper, cfg: DataflowBuildConfig):
    if cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        try:
            from finnexperimental.analysis.partitioning import partition

            # apply partitioning of the model, restricting the first and last layers
            # to SLR0
            default_slr = 0
            abs_anchors = [(0, [default_slr]), (-1, [default_slr])]
            # increase resource limits to make partitioning feasible, except for SLR0
            # which also has DDR subsystem
            limits = np.array(
                [
                    [0.75, 0.5, 0.7, 0.6, 0.6],
                    [1, 0.7, 0.9, 0.8, 0.8],
                    [1, 0.7, 0.9, 0.8, 0.8],
                    [1, 0.7, 0.9, 0.8, 0.8],
                ]
            )
            floorplan = partition(
                model,
                cfg.synth_clk_period_ns,
                cfg.board,
                abs_anchors=abs_anchors,
                multivariant=False,
                linear_cuts=True,
                limits=limits,
            )[0]
            # apply floorplan to model
            model = model.transform(ApplyConfig(floorplan))
            print("SLR floorplanning applied")
        except Exception:
            print("No SLR floorplanning applied")
    return model