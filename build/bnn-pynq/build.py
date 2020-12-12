import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

models = [
    "tfc-w1a1",
    "tfc-w1a2",
    "tfc-w2a2",
    "cnv-w1a1",
    "cnv-w1a2",
    "cnv-w2a2",
]

platforms = ["Pynq-Z1", "Ultra96", "ZCU104"]

for model_name in models:
    for platform_name in platforms:
        cfg = build_cfg.DataflowBuildConfig(
            output_dir="output_%s_%s" % (model_name, platform_name),
            folding_config_file="folding_config/%s_folding_config.json" % model_name,
            synth_clk_period_ns=10.0,
            board=platform_name,
            shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
            generate_outputs=[
                build_cfg.DataflowOutputType.PYNQ_DRIVER,
                build_cfg.DataflowOutputType.STITCHED_IP,
                build_cfg.DataflowOutputType.BITFILE,
            ],
            save_intermediate_models=True,
        )
        model_file = "models/%s.onnx" % model_name
        build.build_dataflow_cfg(model_file, cfg)
