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

from custom_steps import custom_step_add_pre_proc, custom_step_qonnx_tidy_up

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import step_qonnx_to_finn, step_tidy_up

model_name = "espcn-bsd300"

espcn_build_steps = [
    custom_step_qonnx_tidy_up,
    custom_step_add_pre_proc,
    step_qonnx_to_finn,
    step_tidy_up,
]

model_file = "models/espcn-bsd300.onnx"

cfg = build_cfg.DataflowBuildConfig(
    steps=espcn_build_steps,
    output_dir="output_%s_kriasom" % (model_name),
    synth_clk_period_ns=10.0,
    target_fps=10000,
    fpga_part="xck26-sfvc784-2LV-c",
    enable_build_pdb_debug=True,
    verbose=True,
    verify_steps=[
        build_cfg.VerificationStepType.QONNX_TO_FINN_PYTHON,
        build_cfg.VerificationStepType.TIDY_UP_PYTHON,
    ],
    generate_outputs=[],
)

build.build_dataflow_cfg(model_file, cfg)
