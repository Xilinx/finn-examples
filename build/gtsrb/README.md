# Brevitas GTSRB example

This is the binarized CNV topology from the paper [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference](https://arxiv.org/abs/1612.07119) which is trained
on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) dataset.

## Build bitfiles for GTSRB

0. Ensure you have performed the *Setup* steps in the top-level README for setting up the FINN requirements and environment variables.

1. Run the `download-model.sh` script under the `models` directory to download the pretrained QONNX model. You should have `gtsrb/models/cnv_1w1a_gtsrb.onnx` as a result.

2. Launch the build as follows:
```SHELL
# update this according to where you cloned this repo:
FINN_EXAMPLES=/path/to/finn-examples
# cd into finn submodule
cd $FINN_EXAMPLES/build/finn
# launch the build on the gtsrb folder
./run-docker.sh build_custom $FINN_EXAMPLES/build/gtsrb
```

3. The generated outputs will be under `gtsrb/output_<topology>_<board>`. You can find a description of the generated files [here](https://finn-dev.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode).

## Where did the ONNX model files come from?

The model is part of the QONNX model zoo and gets directly downloaded from [here](https://github.com/fastmachinelearning/qonnx_model_zoo/tree/feature/gtsrb_cnv/models/GTSRB/Brevitas_CNV1W1A).
