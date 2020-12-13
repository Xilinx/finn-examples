import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
import os
import shutil


# the BNN-PYNQ models -- these all come as exported .onnx models
# see models/download_bnn_pynq_models.sh
models = [
    "tfc-w1a1",
    "tfc-w1a2",
    "tfc-w2a2",
    "cnv-w1a1",
    "cnv-w1a2",
    "cnv-w2a2",
]

# which platforms to build the networks for
zynq_platforms = ["Pynq-Z1", "Ultra96", "ZCU104"]
alveo_platforms = ["U250"]
platforms_to_build = zynq_platforms + alveo_platforms


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
    for model_name in models:
        # set up the build configuration for this model
        cfg = build_cfg.DataflowBuildConfig(
            output_dir="output_%s_%s" % (model_name, release_platform_name),
            folding_config_file="folding_config/%s_folding_config.json" % model_name,
            synth_clk_period_ns=10.0,
            board=platform_name,
            shell_flow_type=shell_flow_type,
            vitis_platform=vitis_platform,
            generate_outputs=[build_cfg.DataflowOutputType.BITFILE],
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
    # create zipfile for all examples for this platform
    shutil.make_archive(
        "release/" + release_platform_name,
        "zip",
        root_dir="release",
        base_dir=release_platform_name,
    )
