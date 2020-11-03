![alt tag](./finn-logo.png)
![alt tag](./pynq-alveo-logo.png)

# Introductory Examples for using PYNQ with Alveo

TODO update
This project includes introductory examples for using 
[PYNQ](http://www.pynq.io/) with Alveo. It requires `pynq` version `2.5.1` and 
above to work.

Refer to the 
[README](https://github.com/Xilinx/Alveo-PYNQ/tree/master/overlays/README.md) 
in the `overlays` folder for more information regarding the used overlays and 
how they are created.

Please notice that not all overlays might be available for all target devices. 
Supported devices are listed in the overlays 
[README](https://github.com/Xilinx/Alveo-PYNQ/tree/master/overlays/README.md). 
There you may also find instructions on how to synthesize and use overlays for 
a different device.

## Quick Start

TODO update
Install the `pynq-alveo-examples` package using `pip`:
   ```bash
   pip install pynq-alveo-examples
   ```

After the package is installed, to get your own copy of all the notebooks 
available run:
   ```bash
   pynq get-notebooks
   ```

You can then try things out by doing:
   ```bash
   cd pynq-notebooks
   jupyter notebook
   ```

There are a number of additional options for the `pynq get-notebooks` command,
you can list them by typing 
   ```bash
   pynq get-notebooks --help
   ```

You can also refer to the official 
[PYNQ documentation](https://pynq.readthedocs.io/) for more information 
regarding the *PYNQ Command Line Interface* and in particular the 
`get-notebooks` command.
