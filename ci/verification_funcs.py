import logging
import os
import sys


def create_logger():
    # Create a logger to capture output in both console and log file
    logger = logging.getLogger("verif_logger")
    out_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler("all_verification_output.log", mode="w")
    out_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    out_format = logging.Formatter("%(message)s")
    file_format = logging.Formatter("%(message)s")
    out_handler.setFormatter(out_format)
    file_handler.setFormatter(file_format)
    logger.addHandler(out_handler)
    logger.addHandler(file_handler)


def set_verif_steps():
    # Set verification steps
    verif_steps = [
        "finn_onnx_python",
        "initial_python",
        "streamlined_python",
        "folded_hls_cppsim",
        "node_by_node_rtlsim",
        "stitched_ip_rtlsim",
    ]
    return verif_steps


def set_finn_onnx_models():
    finn_onnx_models = [
        "tfc-w1a1",
        "tfc-w1a2",
        "tfc-w2a2",
        "cnv-w1a1",
        "cnv-w1a2",
        "cnv-w2a2",
        "radioml_w4a4_small_tidy",
    ]
    return finn_onnx_models


def set_verif_io(model_name):
    io_folder = os.getenv("VERIFICATION_IO")
    # Set the paths of input/expected output files for verification,
    # using the model name
    if "tfc-w" in model_name:
        # All mnist and cifar10 models use the same i/o
        verif_input = "%s/tfc_mnist_input.npy" % io_folder
        verif_output = "%s/tfc_mnist_output.npy" % io_folder
    elif "cnv-w" in model_name:
        verif_input = "%s/cnv_cifar10_input.npy" % io_folder
        verif_output = "%s/cnv_cifar10_output.npy" % io_folder
    else:
        verif_input = "%s/%s_input.npy" % (io_folder, model_name)
        verif_output = "%s/%s_output.npy" % (io_folder, model_name)
    return verif_input, verif_output


def init_verif(model_name):
    if not logging.getLogger("verif_logger").hasHandlers():
        create_logger()
    verif_steps = set_verif_steps()
    verif_input, verif_output = set_verif_io(model_name)
    return verif_steps, verif_input, verif_output


def verify_build_output(cfg, model_name):
    logger = logging.getLogger("verif_logger")
    verif_output_dir = cfg.output_dir + "/verification_output"
    if os.path.isdir(verif_output_dir) is False:
        logger.info(
            "Verification is enabled, "
            "but verification output for %s on %s has not been generated. "
            "Please run full build with verification enabled.\n" % (model_name, cfg.board)
        )
        return
    logger.info("\n*****************************************************")
    logger.info("Verification Results for %s on %s" % (model_name, cfg.board))
    logger.info("*****************************************************")

    # Using output verification files, print whether verification was
    # success or failure, by iterating through the step names and
    # the output file names and comparing them
    out_files = os.listdir(verif_output_dir)
    for step_name in cfg.verify_steps:
        for file_name in out_files:
            if step_name in file_name:
                # Output file will always end in _SUCCESS.npy or _FAIL.npy
                # (or .npz if verify_save_full_context is enabled),
                # so check the last few characters of the filename
                # to see if it is SUCCESS or FAIL
                if file_name[-8:-4] == "FAIL":
                    logger.info("Verification for step %-22s: FAIL" % step_name)
                elif file_name[-11:-4] == "SUCCESS":
                    logger.info("Verification for step %-22s: SUCCESS" % step_name)
                break
        else:
            match step_name:
                case "step_qonnx_to_finn" if model_name in set_finn_onnx_models():
                    # If the model is already not in QONNX form, then the step skips
                    logger.info(
                        "Verification for step %-22s: MODEL ALREADY IN FINN-ONNX - SKIPPED"
                        % step_name
                    )
                case _:
                    # File for the step was not found, so assume the step was skipped
                    logger.info(
                        "Verification for step %-22s: IO FILE NOT FOUND - SKIPPED" % step_name
                    )
    logger.info(" ")
