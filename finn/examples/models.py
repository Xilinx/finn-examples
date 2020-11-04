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

from finn.examples.driver import FINNExampleOverlay
from finn.core.datatype import DataType
import pkg_resources as pk

def tfc_w1a1_mnist():
    # TODO make dependent on edge/pcie
    platform = "zynq-iodma"
    model_name = "tfc-w1a1"
    ext = "bit"
    bitfile = "%s.%s" % (model_name, ext)
    filename = pk.resource_filename("finn.examples", "bitfiles/%s" % bitfile)
    io_shape_dict = {
        "idt" : DataType.UINT8,
        "odt" : DataType.UINT8,
        "ishape_normal" : (1, 784),
        "oshape_normal" : (1, 1),
        "ishape_folded" : (1, 1, 784),
        "oshape_folded" : (1, 1, 1),
        "ishape_packed" : (1, 1, 784),
        "oshape_packed" : (1, 1, 1),
    }
    return FINNExampleOverlay(filename, platform, io_shape_dict)
