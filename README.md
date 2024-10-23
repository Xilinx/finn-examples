# <img src=https://raw.githubusercontent.com/Xilinx/finn/github-pages/docs/img/finn-logo.png width=200 style="margin-bottom: -15px; margin-right: 10px"/> Dataflow Accelerator Examples
<p align="center"> <em>for PYNQ on Zynq and Alveo</em> <p>
<p align="left">
    <a>
        <img src="https://img.shields.io/github/v/release/Xilinx/finn-examples?color=%09%23228B22&display_name=tag&label=Release" />
    </a>
    <a href="https://github.com/Xilinx/finn/tree/v0.9">
        <img src="https://img.shields.io/badge/FINN-v0.9.0-blue" />
    </a>
    <a href="https://github.com/Xilinx/PYNQ/tree/v3.0.1">
        <img src="https://img.shields.io/badge/PYNQ-v3.0.1-blue" />
    </a>
    <a href="https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2022-1.html">
        <img src="https://img.shields.io/badge/Vivado%2FVitis-v2022.1-blue" />
    </a>
</p>

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
*We recommend PYNQ version 3.0.1, but older installations of PYNQ should also work. For PYNQ v2.6.1, please refer for set-up instructions to [FINN-examples v0.0.5](https://github.com/Xilinx/finn-examples/tree/v0.0.5).*

### Zynq
*For ZYNQ boards, all commands below must be prefixed with `sudo` or by first going into `sudo su`.*

First, source the PYNQ and XRT virtual environment:

```shell
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
```

Next, ensure that your `pip` and `setuptools` installations are up-to-date
on your PYNQ board:

```shell
python3 -m pip install pip==23.0 setuptools==67.1.0
```

Since we are going to install finn-examples without build-isolation, we need to ensure all dependencies are installed. For that, install `setuptools_scm` as well:

```shell
python3 -m pip install setuptools_scm==7.1.0
```

Install the `finn-examples` package using `pip`:

```shell
# remove previous versions with: pip3 uninstall finn-examples
pip3 install finn-examples --no-build-isolation
# to install particular git branch:
# pip3 install git+https://github.com/Xilinx/finn-examples.git@dev --no-build-isolation
```

Retrieve the example Jupyter notebooks using the PYNQ get-notebooks command. An example of how to run the Jupyter notebook server, assuming we are forwarding port 8888 from the target to some port on our local machine, is also shown below:

```shell
# on PYNQ boards, first cd /home/xilinx/jupyter_notebooks
pynq get-notebooks --from-package finn-examples -p . --force
jupyter-notebook --no-browser --allow-root --port=8888
```

### Alveo
*For Alveo we recommend setting up everything inside a virtualenv as described [here](https://pynq.readthedocs.io/en/v2.6.1/getting_started/alveo_getting_started.html?highlight=alveo#install-conda).*

First, create & source a virtual environment:
```shell
conda create -n <virtual-env> python=3.10
conda activate <virtual-env>
```

Next, ensure that your `pip` and `setuptools` installations are up-to-date:
```shell
python3 -m pip install --upgrade pip==23.0 setuptools==67.2.0
```

Finally, we can now install Pynq, FINN-examples and Jupyter (please note to source the XRT environment before):
```shell
pip3 install pynq==3.0.1
python3 -m pip install setuptools_scm==7.1.0 ipython==8.9.0
pip3 install finn-examples --no-build-isolation
# to install particular git branch:
# pip3 install git+https://github.com/Xilinx/finn-examples.git@dev --no-build-isolation
python3 -m pip install jupyter==1.0.0
```

Retrieve the example Jupyter notebooks using the PYNQ get-notebooks command. An example of how to run the Jupyter notebook server is also shown below:

```shell
pynq get-notebooks --from-package finn-examples -p . --force
jupyter-notebook --no-browser --port=8888
```

***

You can now navigate the provided Jupyter notebook examples, or just use the
provided accelerators as part of your own Python program:

```python
from finn_examples import models
import numpy as np

# instantiate the accelerator
accel = models.cnv_w2a2_cifar10()
# generate an empty numpy array to use as input
dummy_in = np.empty(accel.ishape_normal(), dtype=np.uint8)
# perform inference and get output
dummy_out = accel.execute(dummy_in)
```

##  Example Neural Network Accelerators
| Dataset                                                        | Topology                | Quantization                                               | Supported boards | Supported build flows
|:----------------------------------------------------------------:|:-------------------------:|:------------------------------------------------------------:|:------------------:|:------------------:|
| CIFAR-10     | CNV (VGG-11-like)       | several variants:<br>1/2-bit weights/activations           | Pynq-Z1<br>ZCU104<br>Ultra96<br>U250              | Pynq-Z1<br>ZCU104<br>Ultra96<br>U250 |
| MNIST       | 3-layer fully-connected | several variants:<br>1/2-bit weights/activations           | Pynq-Z1<br>ZCU104<br>Ultra96<br>U250              | Pynq-Z1<br>ZCU104<br>Ultra96<br>U250 |
| ImageNet | MobileNet-v1            | 4-bit weights & activations<br>8-bit first layer weights | Alveo U250       | Alveo U250 |
| ImageNet | ResNet-50            | 1-bit weights 2-bit activations<br>4-bit residuals<br>8-bit first/last layer weights | Alveo U250       | - |
| RadioML 2018 | 1D CNN (VGG10)     |  4-bit weights & activations | ZCU104  | ZCU104 |
| MaskedFace-Net | [BinaryCoP](https://arxiv.org/pdf/2102.03456)<br/>*Contributed by TU Munich+BMW*  | 1-bit weights & activations | Pynq-Z1       | Pynq-Z1 |
| Google Speech Commands v2 | 3-layer fully-connected  | 3-bit weights & activations | Pynq-Z1       | Pynq-Z1 |
| UNSW-NB15 | 4-layer fully-connected  | 2-bit weights & activations | Pynq-Z1 <br> ZCU104 <br> Ultra96       | Pynq-Z1 <br> ZCU104 <br> Ultra96 |

*Please note that the build flow for ResNet-50 for the Alveo U250 has known issues and we're currently working on resolving them. However, you can still execute the associated notebook, as we provide a pre-built FPGA bitfile generated with an older Vivado (/FINN) version targeting the [xilinx_u250_xdma_201830_2](https://www.xilinx.com/products/boards-and-kits/alveo/package-files-archive/u250-2018-3-1.html) platform.* <br>
*Furthermore, please note that you can target other boards (such as the Pynq-Z2 or ZCU102) by changing the build script manually, but these accelerators have not been tested.*

We welcome community contributions to add more examples to this repo!

## Supported Boards

*Note that the larger NNs are only available on Alveo or selected Zynq boards.*

`finn-examples` provides pre-built FPGA bitfiles for the following boards:

* **Edge:** Pynq-Z1, Ultra96 and ZCU104
* **Datacenter:** Alveo U250

It's possible to generate Vivado IP for the provided examples to target *any*
modern Xilinx FPGA of sufficient size.
In this case you'll have to manually integrate the generated IP into your design
using Vivado IPI.
You can read more about this [here](build/README.md).

## Rebuilding the bitfiles

All of the examples here are built using the [FINN compiler](https://github.com/Xilinx/finn), and can be re-built or customized.
See the [build/README.md](build/README.md) for more details.
