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

from custom_steps import (
    custom_step_export_verification,
    custom_step_qonnx_tidy_up,
    custom_step_add_pre_proc,
    custom_step_streamline,
    custom_step_convert_to_hw,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import argparse

espcn_build_steps = [
    # custom_step_export_verification,
    custom_step_qonnx_tidy_up,
    custom_step_add_pre_proc,
    "step_qonnx_to_finn",
    "step_tidy_up",
    custom_step_streamline,
    custom_step_convert_to_hw,
    "step_minimize_bit_width",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_pynq_driver",
    "step_deployment_package",
]



def main(model_file, output_dir, folding_config_file, board):
    cfg = build_cfg.DataflowBuildConfig(
        steps=espcn_build_steps,
        output_dir=output_dir,
        synth_clk_period_ns=5.0,
        target_fps=42,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        board=board,
        split_large_fifos=True,
        folding_config_file=folding_config_file,
        auto_fifo_depths=False,
        auto_fifo_strategy = build_cfg.AutoFIFOSizingMethod.CHARACTERIZE,
        rtlsim_batch_size=100,
        max_multithreshold_bit_width = 9,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            # build_cfg.DataflowOutputType.STITCHED_IP,
            # build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.BITFILE,
        ],
    )
    build.build_dataflow_cfg(model_file, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--model_file', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--folding_config_file', type=str, required=False)
    parser.add_argument('-b', '--board', choices=['KV260_SOM', 'ZCU104'],type=str, required=True)
    args = parser.parse_args()
    main(args.model_file, args.output_dir, args.folding_config_file, args.board)
    
