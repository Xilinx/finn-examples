# Tutorial: DNN-Based DDoS Anomaly Detection in the Network Data Plane

This folder contains a 4-part notebook series that shows how a quantized neural network (QNN) can be trained to classify packets as belonging to DDoS (malicious) or regular (benign) network traffic flows. The model is trained with quantized weights and activations, and we use the [Brevitas](https://github.com/Xilinx/brevitas) framework to train the QNN. The model is then converted into an FPGA-friendly RTL implementation for high-throughput inference, which can be integrated with a packet-processing pipeline in the network data plane.

This notebook series is composed of 4 parts. Below is a brief summary of what each part covers.

[Part 1](./1-train.ipynb): How to use Brevitas to train a quantized neural network for our target application, which is classifying packets as belonging to malicious/DDoS or benign/normal network traffic flows. The output trained model at the end of this part is a pure software implementation, i.e. it cannot be converted to a custom RTL FINN model to run on an FPGA just yet.

[Part 2](./2-prepare.ipynb): This notebook focuses on taking the output software model from the previous part and preparing it for hardware-friendly implementation using the FINN framework. The notebook describes the steps taken to "surgery" the software model in order for hardware generation via FINN. We also verify that all the changes made to the software model in this notebook DO NOT affect the output predictions in the "surgeried" model.

[Part 3](./3-build.ipynb): In this notebook, we use the FINN framework to build the custom RTL accelerator for our target model. FINN can generate a variety of RTL accelerators, and this notebook covers some build configuration parameters that influence these outputs.

[Part 4](./4-verify.ipynb): The generated hardware is simulated using cycle-accurate RTL simulation tools, and its outputs are compared against the original software-only model trained in part one. The output model from this step is now ready to be integrated into a larger FPGA design, which in this context is a packet-processing network data plane pipeline designed for identifying anomalous DDoS flows from benign flows.

# Running the notebooks

These tutorial notebooks are written in the same tutorial-style notebooks found on the [FINN GitHub repository](https://github.com/Xilinx/finn). In order to run these interactive Jupyter notebooks in your browser, here are the steps to follow:

1. Clone a copy of the [FINN framework](https://github.com/Xilinx/finn) and checkout the `v0.10` tagged release.
2. Copy entire contents of this folder into `<FINN_ROOT>/notebooks/end2end_example/ddos-anomaly-detector`. You have to use the `ddos-anomaly-detector` folder name as the notebooks use hardcoded relative paths to build outputs in an organized manner. Choosing a different directory name may break the build process.
3. Inside FINN_ROOT, run `./run-docker.sh notebook`, which will build the image and launch a Jupyter Notebook server that you can view in your browser (default on port `8888`). Then, inside your browser, you can simply navigate to `Home` > `end2end_example` > `ddos-anomaly-detector` directory, and start launching the tutorial notebooks.

# Citations

We'd like to thank the Canadian Institute for Cybersecurity for providing open-source network traffic datasets. For this use-case, we use the CIC-IDS2017 dataset which can be [found here](https://www.unb.ca/cic/datasets/ids-2017.html). More details of the dataset and underlying methodologies used to collect the data can be found in their paper:

```
Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion
Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on
Information Systems Security and Privacy (ICISSP), Portugal, January 2018
```

We provide a preprocesed version of their dataset inside the `data/` directory.
The dataset has been split into train and test sets using an 80:20 split. More
information about this preprocessed dataset, along with other details such as
modeling and implementation can also be found in [our demo
paper](https://dl.acm.org/doi/abs/10.1145/3630047.3630191). You can cite our
work using the following bibtex snippet:

```
@inproceedings{siddhartha2023enabling,
  title={Enabling DNN Inference in the Network Data Plane},
  author={Siddhartha and Tan, Justin and Bansal, Rajesh and Chee Cheun, Huang and Tokusashi, Yuta and Yew Kwan, Chong and Javaid, Haris and Baldi, Mario},
  booktitle={Proceedings of the 6th on European P4 Workshop},
  pages={65--68},
  year={2023}
}
```
