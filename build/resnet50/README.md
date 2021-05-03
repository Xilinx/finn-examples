# Resnet 50

Implementation of a binary ResNet50 FINN-style dataflow accelerator targeting Alveo boards.

## Build bitfiles for Resnet 50

We use a specialized build script that replaces a few of the standard steps
in FINN with custom ones.

To add support for two MAC per DSP per cycle install [
finn-experimental](https://github.com/Xilinx/finn-experimental). This allows 16x more FPS per MHz (Alveo U250 implementation).

**Resnet 50 is currently only supported on Alveo U250.**

0. Ensure you have performed the *Setup* steps in the top-level README for setting up the FINN requirements and environment variables.

1. Download the pretrained Resnet 50 ONNX model from the releases page, and extract
the zipfile under `resnet50/models`. You should have e.g. `resnet50/models∕resnet50_w1a2_exported.onnx` as a result.
You can use the provided `resnet50/models/download_resnet50.sh` script for this.

2. Launch the build as follows:
```SHELL
# update this according to where you cloned this repo:
FINN_EXAMPLES=/path/to/finn-examples
# cd into finn submodule
cd $FINN_EXAMPLES/build/finn
# launch the build on the resnet50 folder
./run-docker.sh build_custom /path/to/finn-examples/build/resnet50
```

5. The generated outputs will be under `resnet50/output_<topology>_<board>`. You can find a description of the generated files [here](https://finn-dev.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode).

<!-- ## Where did the ONNX model files come from? -->
