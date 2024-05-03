# MobileNet-v1

MobileNet-v1 was [introduced](https://arxiv.org/abs/1704.04861) by Google in 2017 as a lightweight
DNN targeting the ImageNet dataset.
It has a repeated structure of depthwise-separable (dws) convolution building blocks.
Each dws convolution consists
of a depthwise and a pointwise convolution each followed by a batchnorm and ReLU block.
MobileNet-v1 has 13 of these blocks.
Here, we use a reduced-precision implementation of MobileNet-v1 from [Brevitas](https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/imagenet_classification),
where the weights and activations are quantized to 4-bit, except for the first
layer which uses 8-bit weights and inputs.
It requires about 2 MB of weight storage and 1.1 GMACs per inference, yielding
70.4\% top-1 accuracy on ImageNet.

## Build bitfiles for MobileNet-v1

Due to the depthwise separable convolutions in MobileNet-v1,
we use a specialized build script that replaces a few of the standard steps
in FINN with custom ones.
**MobileNet-v1 is currently only supported on Alveo U250.**
We also provide a folding configuration for the **ZCU102**, but there is no pre-built Pynq image available for this board.

0. Ensure you have performed the *Setup* steps in the top-level README for setting up the FINN requirements and environment variables.

1. Download the pretrained MobileNet-v1 ONNX model from the releases page, and extract
the zipfile under `mobilenet-v1/models`. You should have e.g. `mobilenetv1/models∕mobilenetv1-w4a4_pre_post_tidy.onnx` as a result.
You can use the provided `mobilenet-v1/models/download_mobilenet.sh` script for this.

2. Launch the build as follows:
```SHELL
# update this according to where you cloned this repo:
FINN_EXAMPLES=/path/to/finn-examples
# cd into finn submodule
cd $FINN_EXAMPLES/build/finn
# launch the build on the mobilenet-v1 folder
./run-docker.sh build_custom $FINN_EXAMPLES/build/mobilenet-v1
```

3. The generated outputs will be under `mobilenet-v1/output_<topology>_<board>`. You can find a description of the generated files [here](https://finn-dev.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode).

## Where did the ONNX model files come from?

The 4-bit quantized MobileNet-v1 is part of the
[Brevitas examples](https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/imagenet_classification).
Subsequently, the trained networks is [exported to ONNX](https://github.com/Xilinx/finn/blob/main/notebooks/basics/1_brevitas_network_import_via_QONNX.ipynb). In addition, the particular version used here has two additions for pre- and postprocessing:

* A divide-by-255 node is added at the input, and the input is marked as 8-bit (to directly accept 8-bit images as input)
* Normalization is added at the input with `mean = [0.485, 0.456, 0.406]` and `std = 0.226`. Note that the `std` is global and not per-channel to facilitate its removal via the [streamlining transform](https://arxiv.org/pdf/1709.04060).
* A top-K node with k=5 is added at the output (to return the top-5 class indices instead of logits)

These modifications are done as part of the end2end MobileNet-v1 test in FINN.
You can [see more here](https://github.com/Xilinx/finn/blob/41740ed1a953c09dd2f87b03ebfde5f9d8a7d4f0/tests/end2end/test_end2end_mobilenet_v1.py#L91)
for further reference.
