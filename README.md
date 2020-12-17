## <img src=https://raw.githubusercontent.com/Xilinx/finn/master/docs/img/finn-logo.png width=128/> Dataflow Accelerator Examples
*for PYNQ on Zynq and Alveo*

<img align="left" src="docs/img/finn-example.png" alt="drawing" style="margin-right: 20px" width="250"/>

This repository contains a variety of customized FPGA neural network accelerator
examples built using
the [FINN compiler](https://github.com/Xilinx/finn), which
targets few-bit quantized neural networks with emphasis on
generating dataflow-style architectures customized for each network.

The examples here come with
pre-built bitfiles, PYNQ Python drivers and Jupyter notebooks to get started,
and you can rebuild them from source.
Both PYNQ on Zynq and Alveo are supported.

## Quickstart


*For Alveo we recommend setting up everything inside a virtualenv as described [here](https://pynq.readthedocs.io/en/v2.6.1/getting_started/alveo_getting_started.html?highlight=alveo#install-conda).*

First, ensure that your `pip` and `setuptools` installations are up-to-date
on your PYNQ board or Alveo server:

```shell
python3 -m pip install --upgrade pip setuptools
```

Install the `finn-examples` package using `pip`:

```shell
# remove previous versions with: pip3 uninstall finn-examples
pip3 install finn-examples
# to install particular git branch:
# pip3 install git+https://github.com/Xilinx/finn-examples.git@dev
```

Retrieve the example Jupyter notebooks using the PYNQ get-notebooks command:

```shell
# on PYNQ boards, first cd /home/xilinx/jupyter_notebooks
pynq get-notebooks --from-package finn-examples -p .
```

You can now navigate the provided Jupyter notebook examples, or just use the
provided accelerators as part of your own Python program:

```python
from finn_examples import models
import numpy as np

# instantiate the accelerator
accel = models.cnv_w2a2_cifar10()
# generate an empty numpy array to use as input
dummy_in = np.empty(accel.ishape_normal, dtype=np.uint8)
# perform inference and get output
dummy_out = accel.execute(dummy_in)
```

##  Example Neural Network Accelerators
| Dataset                                                        | Topology                | Quantization                                               | Supported boards |
|----------------------------------------------------------------|-------------------------|------------------------------------------------------------|------------------|
| <img src="docs/img/cifar-10.png" width="150"/><br/>CIFAR-10     | CNV (VGG-11-like)       | several variants:<br>1/2-bit weights/activations           | all              |
| <img src="docs/img/mnist.jpg" width="150"/><br/><br>MNIST       | 3-layer fully-connected | several variants:<br>1/2-bit weights/activations           | all              |
| <img src="docs/img/imagenet.jpg" width="150"/><br/><br>ImageNet | MobileNet-v1            | 4-bit weights and activations<br>8-bit first layer weights | Alveo U250       |

## Supported Boards

*Note that the larger NNs are only available on Alveo boards.*

`finn-examples` provides pre-built FPGA bitfiles for the following boards:

* **Edge:** Pynq-Z1, Pynq-Z2, Ultra96 and ZCU104
* **Datacenter:** Alveo U250

It's possible to generate Vivado IP for the provided examples to target *any*
modern Xilinx FPGA of sufficient size.
In this case you'll have to manually integrate the generated IP into your design
using Vivado IPI.
You can read more about this [here](build/README.md).

## Rebuilding the bitfiles

All of the examples here are built using the [FINN compiler](https://github.com/Xilinx/finn), and can be re-built or customized.
See the [build/README.md](build/README.md) for more details.
