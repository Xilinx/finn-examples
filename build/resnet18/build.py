# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import os
from datetime import datetime
from resnet18_custom_steps import (
    step_resnet18_attach_preproc,
    step_resnet18_streamline,
    step_resnet18_lower,
    step_resnet18_to_hw,
    step_resnet18_slr_floorplan,
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

# The filename our build program will look for a resnet18 model under.
MODEL_FILENAME = "models/resnet18_4w4a.onnx"

# Our target board.
BOARDS = ["U250"]

# Our clock period, in nanoseconds.
SYNTH_CLK_PERIOD_NS = 4.0

# The folding config file we'll apply to the model.
FOLDING_CONFIG_FILE = "folding_config/U250_folding_config.json"

VERIFICATION_IN_OUT_PAIR = ("verification/golden_input.npy",
                            "verification/golden_output.npy")

resnet18_build_steps = [
    "step_qonnx_to_finn",
    step_resnet18_attach_preproc,
    "step_tidy_up",
    step_resnet18_streamline,
    step_resnet18_lower,
    step_resnet18_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    # "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    # Uncomment if you want RTL simulation reports! 
    # "step_measure_rtlsim_performance",
    step_resnet18_slr_floorplan,
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

for board in BOARDS:
    # If the board we're using is an alveo board, use the platform
    # name as the release name. Otherwise, use the board name. In
    # either case, set the relevant shell flow type and Vitis platform.
    if board in alveo_part_map:
        release_platform_name = alveo_default_platform[board]
        shell_flow_type = ShellFlowType.VITIS_ALVEO
        vitis_platform = release_platform_name
    elif board in pynq_part_map:
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
            # VerificationStepType.FOLDED_HLS_CPPSIM,
            VerificationStepType.STITCHED_IP_RTLSIM
        ],
        stitched_ip_gen_dcp = True,
        verify_input_npy= VERIFICATION_IN_OUT_PAIR[0],
        verify_expected_output_npy= VERIFICATION_IN_OUT_PAIR[1],
        split_large_fifos= True,
        save_intermediate_models=True,
        generate_outputs= [
            DataflowOutputType.STITCHED_IP,
            DataflowOutputType.ESTIMATE_REPORTS,
            DataflowOutputType.OOC_SYNTH,
            DataflowOutputType.BITFILE,
            DataflowOutputType.PYNQ_DRIVER,
            DataflowOutputType.DEPLOYMENT_PACKAGE
        ],
        vitis_floorplan_file="folding_config/floorplan.json",
        start_step="step_synthesize_bitfile",
        stop_step="step_deployment_package"
    )

    # Run all of the build steps in our config file.
    build_dataflow_cfg(MODEL_FILENAME, cfg)