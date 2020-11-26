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

import finn.util.build_dataflow as build

# custom steps for mobilenetv1
from custom_steps import (
    step_mobilenet_streamline,
    step_mobilenet_convert_to_hls_layers,
    step_mobilenet_lower_convs,
)

model_name = "mobilenetv1-w4a4"
board = "U250"
synth_clk_period_ns = 3.0

mobilenet_build_steps = [
    step_mobilenet_streamline,
    step_mobilenet_convert_to_hls_layers,
    step_mobilenet_lower_convs,
    "step_create_dataflow_partition",
    "step_apply_folding_config",
    "step_hls_ipgen",
    "step_auto_set_fifo_depths",
    "step_create_stitched_ip",
    "step_make_pynq_driver",
    "step_synthesize_bitfile",
]

cfg = build.DataflowBuildConfig(
    steps=mobilenet_build_steps,
    output_dir="output_%s_%s" % (model_name, board),
    folding_config_file="folding_config/%s_folding_config.json" % board,
    synth_clk_period_ns=synth_clk_period_ns,
    board=board,
    shell_flow_type=build.ShellFlowType.VITIS_ALVEO,
    # folding config comes with FIFO depths already
    auto_fifo_depths=False,
    # use URAM for large FIFOs
    large_fifo_mem_style=build.LargeFIFOMemStyle.URAM,
    # enable extra performance optimizations (physopt)
    vitis_opt_strategy=build.VitisOptStrategyCfg.PERFORMANCE_BEST,
    generate_outputs=[
        build.DataflowOutputType.PYNQ_DRIVER,
        build.DataflowOutputType.STITCHED_IP,
        build.DataflowOutputType.BITFILE,
    ],
)
model_file = "models/%s_pre_post_tidy.onnx" % model_name
build.build_dataflow_cfg(model_file, cfg)
