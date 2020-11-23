# Rebuilding finn-examples


Although this repo comes with pre-synthesized bitfiles, it's possible to
re-generate those.
This can be useful for customizing the existing examples, or for building
to target new Xilinx FPGA platforms.
Rebuilding uses the [FINN compiler](https://github.com/Xilinx/finn) and must happen on a host x86 PC with Vivado/Vitis installed.

## Setup

1. Make sure you have cloned the `finn-examples` repo using the `--recursive` option,
so that the included git submodules are also pulled at checkout.
In case you haven't, you can check them out at a later stage by typing:

  ```git submodule update --init --recursive```

2. Ensure you have the [requirements](https://finn.readthedocs.io/en/latest/getting_started.html#requirements) for FINN installed, which includes
Docker community edition `docker-ce`.

3. Set up the environment variables to point to your Vivado/Vitis installation, depending on your target platform(s):
    *  For Zynq platforms you'll need to set `VIVADO_PATH`, e.g. `VIVADO_PATH=/opt/xilinx/Vivado/2019.1/`
    * For Alveo platforms you'll need to set `VITIS_PATH`, `PLATFORM_REPO_PATHS` and `XILINX_XRT`

## Build bitfiles

Please see the READMEs under the respective subfolders here for instructions on how to rebuild the bitfiles.

## How do I use my custom bitfile? What about the software/drivers?

All examples in this repo use the same Python PYNQ driver, located under
`finn_examples/driver.py` in the repo. This driver can support any FINN-generated
accelerator that doesn't use external weights, the only thing that needs to be
specified is the configuration for the input and output tensors in the `io_shape_dict`. Have a look at `finn_examples/models.py` to see how this is done for the example models in this repo:

```
_cifar10_cnv_io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : DataType.UINT8,
    "odt" : DataType.UINT8,
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : (1, 32, 32, 3),
    "oshape_normal" : (1, 1),
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler, you can see them in the generated Python driver
    "ishape_folded" : (1, 1, 32, 32, 1, 3),
    "oshape_folded" : (1, 1, 1),
    "ishape_packed" : (1, 1, 32, 32, 1, 3),
    "oshape_packed" : (1, 1, 1),
}
```

Instead of using the `driver.py` provided in the repo, you can also use the generated driver under the output folder.
