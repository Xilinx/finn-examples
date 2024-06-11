import finn.builder.build_dataflow_config as build_cfg
import os
import sys
import logging


def create_logger():
    # Create a logger to capture output in both console and log file
    logger = logging.getLogger("verif_logger")
    out_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler('all_verification_output.log', mode='w')
    out_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    out_format = logging.Formatter('%(message)s')
    file_format = logging.Formatter('%(message)s')
    out_handler.setFormatter(out_format)
    file_handler.setFormatter(file_format)
    logger.addHandler(out_handler)
    logger.addHandler(file_handler)


def set_verif_steps():
    # Check if verification is enabled or not (using environment variable)
    if (os.getenv("VERIFICATION_EN")) == "1":
        print("Verification is enabled")
        # Set verification steps
        verif_steps = [
            "finn_onnx_python",
            "initial_python",
            "streamlined_python",
            #"folded_hls_cppsim",
            "stitched_ip_rtlsim",
        ]
    elif (os.getenv("VERIFICATION_EN", "0")) == "0":
        print("Verification is disabled")
        # Don't use any verification steps
        verif_steps = []
    return verif_steps


def verify_build_output(cfg, model_name):
    logger = logging.getLogger("verif_logger")
    if (os.getenv("VERIFICATION_EN")) == "1":
        verif_output_dir = cfg.output_dir + "/verification_output"
        if os.path.isdir(verif_output_dir) == False:
            logger.info("Verification is enabled, but verification output for %s on %s has not been generated. Please run full build with verification enabled.\n" % (model_name, cfg.board))
            return
        logger.info("\n*****************************************************")
        logger.info("Verification Results for %s on %s" % (model_name, cfg.board))
        logger.info("*****************************************************")
        # Verification step QONNX_TO_FINN_PYTHON uses step name different to build_cfg list,
        # so it produces an output .npy/.npz file with different name
        # Change the step name to what is used by the verify_step function,
        # so the produced output file matches the build_cfg list
        if "finn_onnx_python" in cfg.verify_steps:
            cfg.verify_steps = [step.replace("finn_onnx_python", "qonnx_to_finn_python") for step in cfg.verify_steps]

        # Using output verification files, print whether verification was
        # success or failure, by iterating through the step names and
        # the output file names and comparing them
        out_files = os.listdir(verif_output_dir)
        for step_name in cfg.verify_steps:
            for file_name in out_files:
                if step_name in file_name:
                    # Output file will always end in _SUCCESS.npy or _FAIL.npy
                    # (or .npz if verify_save_full_context is enabled),
                    # so check if it is SUCCESS or FAIL
                    if file_name[-8:-4] == "FAIL":
                        logger.info("Verification for step %-22s: FAIL" % step_name)
                    elif file_name[-11:-4] == "SUCCESS":
                        logger.info("Verification for step %-22s: SUCCESS" % step_name)
                    break
            else:
                # File for the step was not found, so assume the step was skipped
                logger.info("Verification for step %-22s: SKIPPED" % step_name)
        logger.info(" ")


