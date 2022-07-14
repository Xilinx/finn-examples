## <img src=https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-logo.png width=128/> Dataflow Accelerator Examples
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

Need help with a problem in this repo, or got a question? Feel free to ask for help in the [GitHub discussions](https://github.com/Xilinx/finn/discussions).
In the past, we also had a [Gitter channel](https://gitter.im/xilinx-finn/community). Please be aware that this is no longer maintained by us but can still be used to search for questions previous users had.

## Quickstart


*For Alveo we recommend setting up everything inside a virtualenv as described [here](https://pynq.readthedocs.io/en/v2.6.1/getting_started/alveo_getting_started.html?highlight=alveo#install-conda).*
*For PYNQ boards, all commands below must be prefixed with `sudo` or by first going into `sudo su`. We recommend PYNQ version 2.6.1 as some installation issues have been reported for PYNQ version 2.7.*

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
pynq get-notebooks --from-package finn-examples -p . --force
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
| Dataset                                                        | Topology                | Quantization                                               | Supported boards | Supported build flows
|----------------------------------------------------------------|-------------------------|------------------------------------------------------------|------------------|------------------|
| <img src="docs/img/cifar-10.png" width="150"/><br/>CIFAR-10     | CNV (VGG-11-like)       | several variants:<br>1/2-bit weights/activations           | all              | Pynq-Z1<br>ZCU104<br>Ultra96 |
| <img src="docs/img/mnist.jpg" width="150"/><br/><br>MNIST       | 3-layer fully-connected | several variants:<br>1/2-bit weights/activations           | all              | Pynq-Z1<br>ZCU104<br>Ultra96 |
| <img src="docs/img/imagenet.jpg" width="150"/><br/><br>ImageNet | MobileNet-v1            | 4-bit weights and activations<br>8-bit first layer weights | Alveo U250<br>ZCU104       | ZCU104 |
| <img src="docs/img/imagenet.jpg" width="150"/><br/><br>ImageNet | ResNet-50            | 1-bit weights 2-bit activations<br>4-bit residuals<br>8-bit first/last layer weights | Alveo U250       | - |
| <img src="docs/img/radioml.png" width="150"/><br/><br>RadioML 2018 | 1D CNN (VGG10)     |  4-bit weights and activations | ZCU104  | ZCU104 |
| <img src="docs/img/maskedfacenet.jpg" width="150"/><br/><br>MaskedFace-Net | [BinaryCoP](https://arxiv.org/pdf/2102.03456)<br/>*Contributed by TU Munich+BMW*  | 1-bit weights and activations | Pynq-Z1       | Pynq-Z1 |
| <img src="docs/img/keyword-spotting.png" width="150"/><br/><br>Google Speech Commands v2 | 3-layer fully-connected  | 3-bit weights and activations | Pynq-Z1       | Pynq-Z1 |

*Please note that for the non-supported Alveo build flows, you can use the pre-built FPGA bitfiles generated with older versions of the Vitis/Vivado tools. These bitfiles target the following Alveo U250 platform: [xilinx_u250_xdma_201830_2](https://www.xilinx.com/products/boards-and-kits/alveo/package-files-archive/u250-2018-3-1.html).

We welcome community contributions to add more examples to this repo!

## Supported Boards

*Note that the larger NNs are only available on Alveo or selected Zynq boards.*

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
