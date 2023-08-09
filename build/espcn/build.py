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
