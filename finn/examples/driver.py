# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pynq import DefaultOverlay, allocate
import pkg_resources as pk
import argparse
import os
import numpy as np
import time
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy
)
from pynq.ps import Clocks

class FINNExampleOverlay(DefaultOverlay):
    def __init__(self, bitfile_name, platform, io_shape_dict, batch_size=1, fclk_mhz=100.0, download=None):
        super().__init__(bitfile_name, download)
        self._io_shape_dict = io_shape_dict
        self.platform = platform
        self.batch_size = batch_size
        self.fclk_mhz = fclk_mhz
        if self.platform == "alveo":
            self.idma = self.idma0
            self.odma = self.odma0
        elif self.platform == "zynq-iodma":
            self.idma = self.idma0
            self.odma = self.odma0
            # set the clock frequency as specified by user during transformations
            if self.fclk_mhz > 0:
                Clocks.fclk0_mhz = self.fclk_mhz
        else:
            raise ValueError("Supported platforms are zynq-iodma alveo")

        # allocate a PYNQ buffer for the packed input and buffer
        if self.platform == "alveo":
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8)
        else:
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8, cacheable=True)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8, cacheable=True)

    @property
    def idt(self):
        return self._io_shape_dict["idt"]

    @property
    def odt(self):
        return self._io_shape_dict["odt"]

    @property
    def ishape_normal(self):
        ret = list(self._io_shape_dict["ishape_normal"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def oshape_normal(self):
        ret = list(self._io_shape_dict["oshape_normal"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def ishape_folded(self):
        ret = list(self._io_shape_dict["ishape_folded"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def oshape_folded(self):
        ret = list(self._io_shape_dict["oshape_folded"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def ishape_packed(self):
        ret = list(self._io_shape_dict["ishape_packed"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def oshape_packed(self):
        ret = list(self._io_shape_dict["oshape_packed"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        # free the old buffers
        self.ibuf_packed_device.freebuffer()
        self.obuf_packed_device.freebuffer()
        if self.platform == "alveo":
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8)
        else:
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8, cacheable=True)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8, cacheable=True)

    def fold_input(self, ibuf_normal):
        """Reshapes input in desired shape.
        Gets input data (ibuf_normal), checks if data is in expected normal shape.
        Returns folded input."""
        # ensure that shape is as expected
        assert ibuf_normal.shape == self.ishape_normal
        # convert to folded form
        ibuf_folded = ibuf_normal.reshape(self.ishape_folded)
        return ibuf_folded

    def pack_input(self, ibuf_folded):
        """Packs folded input and reverses both SIMD dim and endianness.
        Gets input data in folded shape and returns packed input data."""
        ibuf_packed = finnpy_to_packed_bytearray(
            ibuf_folded, self.idt, reverse_endian=True, reverse_inner=True
        )
        return ibuf_packed

    def unpack_output(self, obuf_packed):
        """Unpacks the packed output buffer from accelerator.
        Gets packed output and returns output data in folded shape."""
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed, self.odt, self.oshape_folded, reverse_endian=True, reverse_inner=True
        )
        return obuf_folded

    def unfold_output(self, obuf_folded):
        """Unfolds output data to normal shape.
        Gets folded output data and returns output data in normal shape."""
        obuf_normal = obuf_folded.reshape(self.oshape_normal)
        return obuf_normal

    def copy_input_data_to_device(self, data):
        """Copies given input data to PYNQ buffer."""
        np.copyto(self.ibuf_packed_device, data)
        self.ibuf_packed_device.flush()

    def copy_output_data_from_device(self, data):
        """Copies PYNQ output buffer from device."""
        self.obuf_packed_device.invalidate()
        np.copyto(data, self.obuf_packed_device)

    def execute_on_buffers(self):
        """Executes accelerator by setting up the DMA(s) and
        waiting until all transfers/calls complete. Uses only member variables and
        returns nothing."""
        if self.platform == "zynq-iodma":
            # manually launch IODMAs since signatures are missing
            self.idma.write(0x10, self.ibuf_packed_device.device_address)
            self.idma.write(0x1c, self.batch_size)
            self.odma.write(0x10, self.obuf_packed_device.device_address)
            self.odma.write(0x1c, self.batch_size)
            self.idma.write(0x00, 1)
            self.odma.write(0x00, 1)
            # wait until output IODMA is finished
            status = self.odma.read(0x00)
            while status & 0x2 == 0:
                status = self.odma.read(0x00)
        elif self.platform == "alveo":
            idma_handle = self.idma.start_sw(self.ibuf_packed_device, self.batch_size)
            odma_handle = self.odma.start_sw(self.obuf_packed_device, self.batch_size)
            odma_handle.wait()
        else:
            raise Exception("Unrecognized platform: %s" % self.platform)

    def execute(self, input_npy):
        """Given input numpy array, first perform necessary packing and copying
        to device buffers, execute on accelerator, then unpack output and return
        output numpy array from accelerator."""
        ibuf_folded = self.fold_input(ibuf_normal)
        ibuf_packed = self.pack_input(ibuf_folded)
        self.copy_input_data_to_device(ibuf_packed)
        self.execute_on_buffers()
        obuf_packed = np.empty_like(finnDriver.obuf_packed_device)
        self.copy_output_data_from_device(obuf_packed)
        obuf_folded = self.unpack_output(obuf_packed)
        obuf_normal = self.unfold_output(obuf_folded)
        return obuf_normal

    def throughput_test(self):
        "Run accelerator with empty inputs to measure throughput and other metrics."
        # dictionary for results of throughput test
        res={}
        start = time.time()
        self.execute_on_buffers()
        end = time.time()
        runtime = end - start
        res["runtime[ms]"] = runtime*1000
        res["throughput[images/s]"] = self.batch_size / runtime
        res["DRAM_in_bandwidth[Mb/s]"] = np.prod(self.ishape_packed)*0.000001 / runtime
        res["DRAM_out_bandwidth[Mb/s]"] = np.prod(self.oshape_packed)*0.000001 / runtime
        if self.platform != "alveo":
            res["fclk[mhz]"] = Clocks.fclk0_mhz
        else:
            res["fclk[mhz]"] = self.fclk_mhz
        res["batch_size"] = self.batch_size
        return res
