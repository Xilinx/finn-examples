# Copyright (c) 2020, Xilinx
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

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
from warnings import warn
# custom steps for resnet50v1.5
from custom_steps import (
    step_resnet50_tidy,
    step_resnet50_streamline,
    step_resnet50_convert_to_hls,
    step_resnet50_set_fifo_depths,
    step_resnet50_slr_floorplan
)

model_name = "resnet50_w1a2"
board = "U250"
vitis_platform = alveo_default_platform[board]
synth_clk_period_ns = 4.0
target_fps = 300

resnet50_build_steps = [
    step_resnet50_tidy,
    step_resnet50_streamline,
    step_resnet50_convert_to_hls,
    "step_create_dataflow_partition",
    "step_apply_folding_config",
    "step_generate_estimate_reports",
    "step_hls_codegen",
    "step_hls_ipgen",
    step_resnet50_set_fifo_depths,
    step_resnet50_slr_floorplan,
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]

try:
    from finn.transformation.fpgadataflow.infer_doublepacked_dsp import InferDoublePackedConv
    folding_config_file="folding_config/U250_folding_config.json"
    print("DoublePackedConv detected")
except:
    warn(" FINN Experimental not available. Using non-packed folded down convolution. This is 16 times slower per MHz ")
    folding_config_file="folding_config/U250_folding_config_no_doublepack_pe_folded_16.json"

cfg = build_cfg.DataflowBuildConfig(
    steps=resnet50_build_steps,
    output_dir="output_%s_%s" % (model_name, board),
    synth_clk_period_ns=synth_clk_period_ns,
    board=board,
    shell_flow_type=build_cfg.ShellFlowType.VITIS_ALVEO,
    vitis_platform=vitis_platform,
    # throughput parameters (auto-folding)
    mvau_wwidth_max = 24,
    target_fps = target_fps,
    folding_config_file =  folding_config_file,
    # enable extra performance optimizations (physopt)
    vitis_opt_strategy=build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
    generate_outputs=[
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
)

model_file = "models/%s_exported.onnx" % model_name
build.build_dataflow_cfg(model_file, cfg)
