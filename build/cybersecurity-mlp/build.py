import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import alveo_default_platform
import os
import shutil
from custom_steps import custom_step_mlp_export

# Which platforms to build the networks for
zynq_platforms = ["Pynq-Z1", "Ultra96", "ZCU104"]
alveo_platforms = []

# Note: only zynq platforms currently tested
platforms_to_build = zynq_platforms + alveo_platforms

# determine which shell flow to use for a given platform
def platform_to_shell(platform):
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    elif platform in alveo_platforms:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")

# Define model name
model_name = "unsw_nb15-mlp-w2a2"

# Create a release dir, used for finn-examples release packaging
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
    # Set up the build configuration for this model
    cfg = build_cfg.DataflowBuildConfig(
        output_dir = "output_%s_%s" % (model_name, release_platform_name),
        mvau_wwidth_max = 80,
        target_fps = 1000000,
        synth_clk_period_ns = 10.0,
        board = platform_name,
        shell_flow_type = shell_flow_type,
        vitis_platform = vitis_platform,
        vitis_opt_strategy = build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
        generate_outputs = [
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
        save_intermediate_models=True
    )

    # Export MLP model to FINN-ONNX
    model = custom_step_mlp_export(model_name)
    # Launch FINN compiler to generate bitfile
    build.build_dataflow_cfg(model, cfg)
    # Copy bitfiles into release dir if found
    bitfile_gen_dir = cfg.output_dir + "/bitfile"
    filtes_to_check_and_copy = [
        "finn-accel.bit",
        "finn-accel.hwh",
        "finn-accel.xclbin"
    ]
    for f in filtes_to_check_and_copy:
        src_file = bitfile_gen_dir + "/" + f
        dst_file = platform_dir + "/" + f.replace("finn-accel", model_name)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)
