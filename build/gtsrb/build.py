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

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
from finn.builder.build_dataflow_config import default_build_dataflow_steps
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.core.datatype import DataType
import os
import shutil
import numpy as np
from onnx import helper as oh


model_name = "cnv_1w1a_gtsrb"
model_file = "models/%s.onnx" % model_name

verif_en = os.getenv("VERIFICATION_EN", "0")

# which platforms to build the networks for
zynq_platforms = ["Pynq-Z1"]
alveo_platforms = []
platforms_to_build = zynq_platforms + alveo_platforms


def custom_step_add_preproc(model, cfg):
    # GTSRB data with raw uint8 pixels is divided by 255 prior to training
    # reflect this in the inference graph so we can perform inference directly
    # on raw uint8 data
    in_name = model.graph.input[0].name
    new_in_name = model.make_new_valueinfo_name()
    new_param_name = model.make_new_valueinfo_name()
    div_param = np.asarray(255.0, dtype=np.float32)
    new_div = oh.make_node(
        "Div",
        [in_name, new_param_name],
        [new_in_name],
        name="PreprocDiv",
    )
    model.set_initializer(new_param_name, div_param)
    model.graph.node.insert(0, new_div)
    model.graph.node[1].input[0] = new_in_name
    # set input dtype to uint8
    model.set_tensor_datatype(in_name, DataType["UINT8"])

    return model


# Insert TopK node to get predicted Top-1 class
def custom_step_add_postproc(model, cfg):
    model = model.transform(InsertTopK(k=1))
    return model


custom_build_steps = (
    [custom_step_add_preproc]
    + [custom_step_add_postproc]
    + default_build_dataflow_steps
)


# determine which shell flow to use for a given platform
def platform_to_shell(platform):
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    elif platform in alveo_platforms:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


# create a release dir, used for finn-examples release packaging
os.makedirs("release", exist_ok=True)

for platform_name in platforms_to_build:
    shell_flow_type = platform_to_shell(platform_name)
    if shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        vitis_platform = alveo_default_platform[platform_name]
        # for Alveo, use the Vitis platform name as the release name
        # e.g. xilinx_u250_xdma_201830_2
        release_platform_name = vitis_platform
    else:
        vitis_platform = None
        # for Zynq, use the board name as the release name
        # e.g. ZCU104
        release_platform_name = platform_name
    platform_dir = "release/%s" % release_platform_name
    os.makedirs(platform_dir, exist_ok=True)
    # set up the build configuration for this model
    cfg = build_cfg.DataflowBuildConfig(
        output_dir="output_%s_%s" % (model_name, release_platform_name),
        target_fps=3000,
        synth_clk_period_ns=10.0,
        board=platform_name,
        steps=custom_build_steps,
        folding_config_file="folding_config/gtsrb_folding_config.json",
        specialize_layers_config_file="specialize_layers_config/gtsrb_specialize_layers.json",
        shell_flow_type=shell_flow_type,
        vitis_platform=vitis_platform,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
        ],
    )
    # launch FINN compiler to build
    if verif_en == "1":
        # Build the model with verification
        import sys

        sys.path.append(os.path.abspath(os.getenv("FINN_EXAMPLES_ROOT") + "/ci/"))
        from verification_funcs import init_verif, verify_build_output

        cfg.verify_steps, cfg.verify_input_npy, cfg.verify_expected_output_npy = init_verif(
            model_name
        )
        build.build_dataflow_cfg(model_file, cfg)
        verify_build_output(cfg, model_name)
    else:
        # Build the model without verification
        build.build_dataflow_cfg(model_file, cfg)

    # copy bitfiles into release dir if found
    bitfile_gen_dir = cfg.output_dir + "/bitfile"
    files_to_check_and_copy = [
        "finn-accel.bit",
        "finn-accel.hwh",
        "finn-accel.xclbin",
    ]
    for f in files_to_check_and_copy:
        src_file = bitfile_gen_dir + "/" + f
        dst_file = platform_dir + "/" + f.replace("finn-accel", model_name)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)
