# Imports.

import os
from datetime import datetime
from resnet18_custom_steps import (
    step_resnet18_attach_preproc,
    step_resnet18_streamline,
    step_resnet18_lower,
    step_resnet18_to_hls,
    step_resnet50_slr_floorplan,
)
from finn.util.basic import (
    pynq_part_map, alveo_part_map, alveo_default_platform
)
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
    VerificationStepType,
)
from finn.builder.build_dataflow import build_dataflow_cfg

#-------------------------------------------#

# The filename our build program will look for a resnet18 model under.
MODEL_FILENAME = "models/resnet18_4w4a.onnx"

# Our target board.
BOARDS = ["U250"]

# Our clock period, in nanoseconds.
SYNTH_CLK_PERIOD_NS = 4.0

# The folding config file we'll apply to the model.
FOLDING_CONFIG_FILE = "folding_config/U250_folding_config_100k.json"

# TODO: RELEVANT COMMENT
VERIFICATION_IN_OUT_PAIR = ("verification/golden_input.npy",
                            "verification/golden_output.npy")

# TODO: RELEVANT COMMENT
USE_RTL_NODES = False

resnet18_build_steps = [
    "step_qonnx_to_finn",
    step_resnet18_attach_preproc,
    "step_tidy_up",
    step_resnet18_streamline,
    step_resnet18_lower,
    step_resnet18_to_hls,
    "step_create_dataflow_partition",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hls_codegen",
    "step_hls_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    # Uncomment if you want RTL simulation reports! 
    # "step_measure_rtlsim_performance",
    step_resnet50_slr_floorplan,
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

for board in BOARDS:

    # If the board we're using is an alveo board, use the platform
    # name as the release name. Otherwise, use the board name. In
    # either case, set the relevant shell flow type and Vitis platform.
    if (board in alveo_part_map):
        release_platform_name = alveo_default_platform[board]
        shell_flow_type = ShellFlowType.VITIS_ALVEO
        vitis_platform = release_platform_name
    elif (board in pynq_part_map):
        release_platform_name = board
        shell_flow_type = ShellFlowType.VIVADO_ZYNQ
        vitis_platform = None
    else:
        raise Exception("resnet18_build_config.BOARD set to unknown platform.\n"
                        "Check the part maps in finn.util.basic for valid boards.")

    # We'll extract the name of or model from the filename provided.
    model_name = "resnet18_4w4a"

    # Set the output directory based on the model name,
    # and release platform name.
    local_output_dir = f"output_{model_name}_{release_platform_name}"

    # Create a DataflowBuildConfig object with all of our settings.
    cfg = DataflowBuildConfig(
        steps= resnet18_build_steps,
        output_dir= local_output_dir,
        synth_clk_period_ns= SYNTH_CLK_PERIOD_NS,
        board= board,
        shell_flow_type= shell_flow_type,
        vitis_platform= vitis_platform,
        folding_config_file= FOLDING_CONFIG_FILE,
        verify_steps= [
            VerificationStepType.STREAMLINED_PYTHON,
            VerificationStepType.FOLDED_HLS_CPPSIM,
            VerificationStepType.STITCHED_IP_RTLSIM
        ],
        stitched_ip_gen_dcp = True,
        force_rtl_conv_inp_gen = USE_RTL_NODES,
        verify_input_npy= VERIFICATION_IN_OUT_PAIR[0],
        verify_expected_output_npy= VERIFICATION_IN_OUT_PAIR[1],
        split_large_fifos= True,
        generate_outputs= [
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.OOC_SYNTH,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE
        ]
    )

    # Run all of the build steps in our config file.
    build_dataflow_cfg(MODEL_FILENAME, cfg)