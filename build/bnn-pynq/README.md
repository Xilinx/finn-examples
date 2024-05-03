# The BNN-PYNQ examples

The BNN-PYNQ examples include the `tfc-w*a*, cnv-w*a*` networks, which are
simple linear topologies that work on the MNIST and CIFAR-10 datasets.

## Build bitfiles for BNN-PYNQ examples

Since the cross product of {topologies} x {bitwidths} x {platforms} is
quite large we use a custom Python script to automate the build.
**It's strongly recommended to edit this script before you launch the build,
as building the full cross product can take several days to finish.**

0. Ensure you have performed the *Setup* steps in the top-level README for setting up the FINN requirements and environment variables.

1. Download the pretrained BNN-PYNQ ONNX models from the releases page, and extract
the zipfile under `bnn-pynq/models`. You should have e.g. `bnn-pynq/models∕cnv-w2a2.onnx`.
You can use the provided `bnn-pynq/models/download_bnn_pynq_models.sh` script for this.

2. Edit the `bnn-pynq/build.py` to restrict the `models` and `platforms` variables to
the ones that you are interested in, e.g. `models=["cnv_w2a2"]` and
`platforms=["Pynq-Z1", "U250"]`. You can also change the other build
configuration options, see the [FINN docs](https://finn-dev.readthedocs.io/en/latest/source_code/finn.util.html#finn.util.build_dataflow.DataflowBuildConfig)
for a full explanation.

3. If you want to change the folding (parallelization) settings to control the throughput of the generated accelerator, these
are configured by the `bnn-pynq/_folding_config/<topology>_folding_config.json` files.

4. Launch the build as follows:
```shell
# update this according to where you cloned this repo:
FINN_EXAMPLES=/path/to/finn-examples
# cd into finn submodule
cd $FINN_EXAMPLES/build/finn
# launch the build on the bnn-pynq folder
./run-docker.sh build_custom $FINN_EXAMPLES/build/bnn-pynq
```

5. The generated outputs will be under `bnn-pynq/output_<topology>_<platform>`. You can find a description of the generated files [here](https://finn-dev.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode).

## Where did those ONNX model files come from?

The BNN-PYNQ networks are part of the
[Brevitas examples](https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/bnn_pynq). You can find the details on quantization, accuracy, layers used in the Brevitas repo, as well as the training scripts if you'd like to retrain them yourself.

Subsequently, those trained networks are [exported to ONNX](https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import_via_QONNX.ipynb). In addition, the particular versions
used here have two additions, as described in the "Adding Pre- and Postprocessing" section of [this notebook](https://github.com/Xilinx/finn/blob/master/notebooks/end2end_example/bnn-pynq/tfc_end2end_example.ipynb):

* A divide-by-255 node is added at the input, and the input is marked as 8-bit (to directly accept 8-bit images as input)
* A top-K node with k=1 is added at the output (to return the class index instead of logits)

## Why are these called BNN-PYNQ networks?

BNN-PYNQ was the name of the GitHub repo that the FINN team first
released after the FINN paper was published. Originally, it contained
BNNs (1-bit networks) running on the PYNQ-Z1 platform. Over time the
examples were extended to multi-bit quantization,
but the name was kept unchanged as the repo had many users.
This is why we still refer to these networks as BNN-PYNQ networks.
