# The multilayer perceptron for cybersecurity use-cases
The MLP for the cybersecurity use-case is based on the three-part tutorial for training a quantized MLP and deploying it with FINN that is provided in the FINN [end-to-end example repository](https://github.com/Xilinx/finn/tree/main/notebooks/end2end_example). For more information on training the network, or more details behind what's happening under the hood, these notebooks provide an excellent starting point.

# Build bitfiles for MLP example
0. Ensure you have performed the Setup steps in the top-level README for setting up the FINN requirements and environment variables.

1. Edit the `mlp-cybersecurity/build.py` to restrict the platform variables to the ones that you are interested in, e.g. `zynq_platforms = ["Pynq-Z1"]`. You can also change the other build configuration options, see the [FINN docs](https://finn-dev.readthedocs.io/en/latest/source_code/finn.util.html#finn.util.build_dataflow.DataflowBuildConfig) for a full explanation.

2. Launch the build as follows:
```shell
# update this according to where you cloned this repo:
FINN_EXAMPLES=/path/to/finn-examples
# cd into finn submodule
cd $FINN_EXAMPLES/build/finn
# launch the build on the cybersecurity-mlp folder
./run-docker.sh build_custom /path/to/finn-examples/build/cybersecurity-mlp
```

3. The generated outputs will be under `cybersecurity-mlp/output_<topology>_<platform>`. You can find a description of the generated files [here](https://finn-dev.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode).

# Where did the ONNX model file come from?
The ONNX model is created and exported prior to the build flow is launched. You can find the details of this process in the `mlp-cybersecurity/custom_steps.py` file.