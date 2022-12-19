import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
from finn.builder.build_dataflow_config import default_build_dataflow_steps
from qonnx.core.datatype import DataType
import os
import shutil
import numpy as np
from onnx import helper as oh

models = [
    "cnv_gtsrb",
]

# which platforms to build the networks for
zynq_platforms = ["Pynq-Z1"]
platforms_to_build = zynq_platforms

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


custom_build_steps = [custom_step_add_preproc] + default_build_dataflow_steps

# determine which shell flow to use for a given platform
def platform_to_shell(platform):
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


# create a release dir, used for finn-examples release packaging
os.makedirs("release", exist_ok=True)

for platform_name in platforms_to_build:
    shell_flow_type = platform_to_shell(platform_name)
    vitis_platform = None
    # for Zynq, use the board name as the release name
    # e.g. ZCU104
    release_platform_name = platform_name
    platform_dir = "release/%s" % release_platform_name
    os.makedirs(platform_dir, exist_ok=True)
    for model_name in models:
        # set up the build configuration for this model
        cfg = build_cfg.DataflowBuildConfig(
            output_dir="output_%s_%s" % (model_name, release_platform_name),
            folding_config_file="folding_config/%s_folding_config.json" % model_name,
            synth_clk_period_ns=10.0,
            board=platform_name,
            steps=custom_build_steps,
            shell_flow_type=shell_flow_type,
            vitis_platform=vitis_platform,
            generate_outputs=[build_cfg.DataflowOutputType.BITFILE],
            #stop_step="step_create_dataflow_partition",
            save_intermediate_models=True,
        )
        model_file = "models/%s.onnx" % model_name
        # launch FINN compiler to build
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
