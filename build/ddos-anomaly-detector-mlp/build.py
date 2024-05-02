import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import os
from custom_steps import custom_step_mlp_export
from finn.util.basic import alveo_part_map

# Which Alveo board to build for
platforms_to_build = ["U250", "U280"]

# Define model name
model_name = "ddos-anomaly-detector-mlp-w2a2"

# Create a release dir, used for finn-examples release packaging
os.makedirs("release", exist_ok=True)

# Path to custom_folding_config.json
custom_folding_config_fpath = "./data/custom_folding_config.json"

for platform_name in platforms_to_build:
    platform_dir = "release/%s" % platform_name
    os.makedirs(platform_dir, exist_ok=True)
    fpga_part = alveo_part_map[platform_name]
    # Set up the build configuration for this model
    cfg = build_cfg.DataflowBuildConfig(
        output_dir="output_%s_%s" % (model_name, platform_name),
        mvau_wwidth_max=300,
        target_fps=250000000,  # 250M inferences/sec
        synth_clk_period_ns=4.0,  # 250MHz (for OpenNIC 250MHz compute box)
        fpga_part=fpga_part,  # OpenNIC-compatible FPGA
        folding_config_file=custom_folding_config_fpath,
        rtlsim_batch_size=4,
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.OOC_SYNTH,
        ],
        save_intermediate_models=True,
    )

    # Export MLP model to FINN-ONNX
    model = custom_step_mlp_export(model_name)
    # Launch FINN compiler to generate stitched IP
    build.build_dataflow_cfg(model, cfg)
