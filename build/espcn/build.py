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

from custom_steps import (
    # custom_step_export_verification,
    custom_step_qonnx_tidy_up,
    custom_step_add_pre_proc,
    custom_step_streamline,
    custom_step_convert_to_hls,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

model_name = "espcn-bsd300-RCNNTDC"


espcn_build_steps = [
    # custom_step_export_verification,
    custom_step_qonnx_tidy_up,
    custom_step_add_pre_proc,
    "step_qonnx_to_finn",
    "step_tidy_up",
    custom_step_streamline,
    custom_step_convert_to_hls,
    "step_minimize_bit_width",
    "step_create_dataflow_partition",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_generate_estimate_reports",
    "step_hls_codegen",
    "step_hls_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

model_file = "quant_espcn_x2_w4a4_base/qonnx_model.onnx"

cfg = build_cfg.DataflowBuildConfig(
    steps=espcn_build_steps,
    output_dir="output_%s_kriasom" % (model_name),
    synth_clk_period_ns=5.0,
    target_fps=30,
    fpga_part="xck26-sfvc784-2LV-c",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    board="KV260_SOM",
    enable_build_pdb_debug=True,
    verbose=True,
    split_large_fifos=True,
    folding_config_file="folding_config_rcnntdc_cap.json",
    auto_fifo_depths=False,
    # auto_fifo_strategy = build_cfg.AutoFIFOSizingMethod.CHARACTERIZE,
    rtlsim_batch_size=100,
    force_rtl_conv_inp_gen=False,
    # start_step="step_hls_ipgen",
    # stop_step = "step_generate_estimate_reports",
    verify_input_npy="quant_espcn_x2_w4a4_base/input.npy",
    verify_expected_output_npy="quant_espcn_x2_w4a4_base/output.npy",
    verify_steps=[
        # build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
        # build_cfg.VerificationStepType.TIDY_UP_PYTHON,
        # build_cfg.VerificationStepType.STREAMLINED_PYTHON,
        # build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,True
    ],
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        build_cfg.DataflowOutputType.BITFILE,
    ],
)

build.build_dataflow_cfg(model_file, cfg)
