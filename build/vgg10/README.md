# VGG10

This 1-dimensional CNN was [introduced](https://arxiv.org/pdf/1712.04578.pdf) by DeepSig alongside their RadioML 2018 dataset for RF modulation classification.
It consists of 7 1D convolution + maxpooling layers, followed by 2 hidden dense layers and the final dense classification layer. ReLU activations and Batchnorm are applied throughout the network. The input is a frame of 1024 I/Q samples (i.e. shape [1024,2]), the classifier distinguishes 24 classes (i.e. modulation types).

Here, we use a reduced-precision implementation trained on [RadioML 2018.01A](https://www.deepsig.ai/datasets) with Brevitas. The weights are quantized to 4-bit and the activations to 3-bit, except for the first layer which uses 4-bit activations. The pre-trained model reaches 58.4% overall accuracy and 90.9% at the highest SNR (30 dB).

## Build bitfiles for VGG10

Due to the 1-dimensional topology in VGG10,
we use a specialized build script that adds a few custom build steps to the standard steps in FINN.
**We currently provide bitstreams and the corresponding build configuration only for the ZCU104, but plan to extend to other boards shortly.**

0. Ensure you have performed the *Setup* steps in the top-level README for setting up the FINN requirements and environment variables.

1. Download the pretrained VGG10 ONNX model from the releases page, and extract
the zipfile under `vgg10/models`. You should have e.g. `vgg10/models∕radioml-w4a3_tidy.onnx` as a result.
You can use the provided `vgg10/models/download_vgg10.sh` script for this.

2. Launch the build as follows:
```SHELL
# update this according to where you cloned this repo:
FINN_EXAMPLES=/path/to/finn-examples
# cd into finn submodule
cd $FINN_EXAMPLES/build/finn
# launch the build on the vgg10 folder
./run-docker.sh build_custom /path/to/finn-examples/build/vgg10
```

5. The generated outputs will be under `vgg10/output_<topology>_<board>`. You can find a description of the generated files [here](https://finn-dev.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode).

## Where did the ONNX model files come from?

The quantized VGG10 is based on the baseline topology for our problem statement in the ITU AI/ML in 5G Challenge. You can find it in our [sandbox repository](https://github.com/Xilinx/brevitas-radioml-challenge-21).

In addition, the ONNX model has been tidied up by removing the input quantization, which we do in software for this example, and by adding a top-k (k=1) node at the output. Thus, the accelerator returns the top-1 class index instead of logits.
