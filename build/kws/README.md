# The KWS example

The KWS example includes an MLP for the Google SpeechCommandsV2 dataset.

## Build bitfiles for BNN-PYNQ examples

The build is currently configured for the PYNQ-Z1 board and a throughput of 200k FPS at a clock frequency of 100 MHz.

1. Download the pretrained MLP ONNX models and pre-processed validation data using the `get-kws-data-model.sh` script.

2. Launch the build as follows:
```shell
# update this according to where you cloned this repo:
FINN_EXAMPLES=/path/to/finn-examples
# cd into finn submodule
cd $FINN_EXAMPLES/build/finn
# launch the build on the bnn-pynq folder
bash run-docker.sh build_custom $FINN_EXAMPLES/build/kws
```

3. The generated outputs will be under `kws/<timestamp>_output_<onnx_file_name>_<platform>`.
You can find a description of the generated files [here](https://finn-dev.readthedocs.io/en/latest/command_line.html#simple-dataflow-build-mode).
The folder will additionally include the quantized inputs for verification (`all_validation_KWS_data_inputs_len_10102.npy`) and the expected outputs (`all_validation_KWS_data_outputs_len_10102.npy`).
When running the network on hardware the validation should achieve an accuracy of 89.78 % with 9070 of the 10102 samples being classified correctly.
