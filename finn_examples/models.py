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

import pkg_resources as pk

import os
import platform
import pynq

from qonnx.core.datatype import DataType
from finn_examples.driver import FINNExampleOverlay

_mnist_fc_io_shape_dict = {
    "idt": DataType["UINT8"],
    "odt": DataType["UINT8"],
    "ishape_normal": (1, 784),
    "oshape_normal": (1, 1),
    "ishape_folded": (1, 1, 784),
    "oshape_folded": (1, 1, 1),
    "ishape_packed": (1, 1, 784),
    "oshape_packed": (1, 1, 1),
}

_cifar10_cnv_io_shape_dict = {
    "idt": DataType["UINT8"],
    "odt": DataType["UINT8"],
    "ishape_normal": (1, 32, 32, 3),
    "oshape_normal": (1, 1),
    "ishape_folded": (1, 1, 32, 32, 1, 3),
    "oshape_folded": (1, 1, 1),
    "ishape_packed": (1, 1, 32, 32, 1, 3),
    "oshape_packed": (1, 1, 1),
}

_bincop_cnv_io_shape_dict = {
    "idt": DataType["UINT8"],
    "odt": DataType["UINT8"],
    "ishape_normal": (1, 72, 72, 3),
    "oshape_normal": (1, 1),
    "ishape_folded": (1, 1, 72, 72, 1, 3),
    "oshape_folded": (1, 1, 1),
    "ishape_packed": (1, 1, 72, 72, 1, 3),
    "oshape_packed": (1, 1, 1),
}

_imagenet_top5inds_io_shape_dict = {
    "idt": DataType["UINT8"],
    "odt": DataType["UINT16"],
    "ishape_normal": (1, 224, 224, 3),
    "oshape_normal": (1, 1, 1, 5),
    "ishape_folded": (1, 224, 224, 1, 3),
    "oshape_folded": (1, 1, 1, 1, 5),
    "ishape_packed": (1, 224, 224, 1, 3),
    "oshape_packed": (1, 1, 1, 1, 10),
}

# resnet50 uses a different io_shape_dict due to
# external weights for last layer
_imagenet_resnet50_top5inds_io_shape_dict = {
    "idt": DataType["UINT8"],
    "odt": DataType["UINT16"],
    "ishape_normal": (1, 224, 224, 3),
    "oshape_normal": (1, 5),
    "ishape_folded": (1, 224, 224, 3),
    "oshape_folded": (1, 5, 1),
    "ishape_packed": (1, 224, 224, 3),
    "oshape_packed": (1, 5, 2),
    "input_dma_name": "idma1",
    "number_of_external_weights": 1,
}

_radioml_io_shape_dict = {
    "idt": DataType["INT8"],
    "odt": DataType["UINT8"],
    "ishape_normal": (1, 1024, 1, 2),
    "oshape_normal": (1, 1),
    "ishape_folded": (1, 1024, 1, 1, 2),
    "oshape_folded": (1, 1, 1),
    "ishape_packed": (1, 1024, 1, 1, 2),
    "oshape_packed": (1, 1, 1),
}

_gscv2_mlp_io_shape_dict = {
    "idt" : DataType["INT8"],
    "odt" : DataType["UINT8"],
    "ishape_normal" : (1, 490),
    "oshape_normal" : (1, 1),
    "ishape_folded" : (1, 49, 10),
    "oshape_folded" : (1, 1, 1),
    "ishape_packed" : (1, 49, 10),
    "oshape_packed" : (1, 1, 1),
    "input_dma_name" : 'idma0',
}

# from https://github.com/Xilinx/PYNQ-HelloWorld/blob/master/setup.py
# get current platform: either edge or pcie


def get_edge_or_pcie():
    cpu = platform.processor()
    if cpu in ["armv7l", "aarch64"]:
        return "edge"
    elif cpu in ["x86_64"]:
        return "pcie"
    else:
        raise OSError("Platform is not supported.")


def find_bitfile(model_name, target_platform):
    bitfile_exts = {"edge": "bit", "pcie": "xclbin"}
    bitfile_ext = bitfile_exts[get_edge_or_pcie()]
    bitfile_name = "%s.%s" % (model_name, bitfile_ext)
    bitfile_candidates = [
        pk.resource_filename(
            "finn_examples", "bitfiles/%s/%s" % (target_platform, bitfile_name)
        ),
        pk.resource_filename(
            "finn_examples",
            "bitfiles/bitfiles.zip.d/%s/%s" % (target_platform, bitfile_name),
        ),
    ]
    for candidate in bitfile_candidates:
        if os.path.isfile(candidate):
            return candidate
    raise Exception(
        "Bitfile for model = %s target platform = %s not found. Looked in: %s"
        % (model_name, target_platform, str(bitfile_candidates))
    )


def find_runtime_weights(model_name, target_platform):
    weight_dir = "%s_runtime_weights" % (model_name)
    weight_dir_candidates = [
        pk.resource_filename(
            "finn_examples", "bitfiles/%s/%s" % (target_platform, weight_dir)
        ),
        pk.resource_filename(
            "finn_examples",
            "bitfiles/bitfiles.zip.d/%s/%s" % (target_platform, weight_dir),
        ),
    ]
    for candidate in weight_dir_candidates:
        if os.path.isdir(candidate):
            weight_files = os.listdir(candidate)
            if weight_files:
                return candidate
    raise Exception(
        "Runtime weights for model = %s target platform = %s not found. Looked in: %s"
        % (model_name, target_platform, str(weight_dir_candidates))
    )


def get_driver_mode():
    driver_modes = {"edge": "zynq-iodma", "pcie": "alveo"}
    return driver_modes[get_edge_or_pcie()]


def resolve_target_platform(target_platform):
    if target_platform is None:
        return pynq.Device.active_device.name
    else:
        assert target_platform in [x.name for x in pynq.Device.devices]
        return target_platform

def kws_mlp(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "kwsmlp-w3a3"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _gscv2_mlp_io_shape_dict)

def tfc_w1a1_mnist(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "tfc-w1a1"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _mnist_fc_io_shape_dict)


def tfc_w1a2_mnist(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "tfc-w1a2"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _mnist_fc_io_shape_dict)


def tfc_w2a2_mnist(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "tfc-w2a2"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _mnist_fc_io_shape_dict)


def cnv_w1a1_cifar10(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "cnv-w1a1"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _cifar10_cnv_io_shape_dict)


def cnv_w1a2_cifar10(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "cnv-w1a2"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _cifar10_cnv_io_shape_dict)


def cnv_w2a2_cifar10(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "cnv-w2a2"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _cifar10_cnv_io_shape_dict)


def bincop_cnv(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "bincop-cnv"
    filename = find_bitfile(model_name, target_platform)
    return FINNExampleOverlay(filename, driver_mode, _bincop_cnv_io_shape_dict)


def mobilenetv1_w4a4_imagenet(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "mobilenetv1-w4a4"
    filename = find_bitfile(model_name, target_platform)
    if target_platform in ["ZCU104"]:
        runtime_weight_dir = find_runtime_weights(model_name, target_platform)
    else:
        runtime_weight_dir = ""
    # target 185 MHz for Zynq (this is ignored for Alveo)
    fclk_mhz = 185.0
    return FINNExampleOverlay(
        filename,
        driver_mode,
        _imagenet_top5inds_io_shape_dict,
        runtime_weight_dir=runtime_weight_dir,
        fclk_mhz=fclk_mhz,
    )


def resnet50_w1a2_imagenet(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "resnet50-w1a2"
    filename = find_bitfile(model_name, target_platform)
    runtime_weight_dir = find_runtime_weights(model_name, target_platform)
    return FINNExampleOverlay(
        filename,
        driver_mode,
        _imagenet_resnet50_top5inds_io_shape_dict,
        runtime_weight_dir=runtime_weight_dir,
    )


def vgg10_w4a4_radioml(target_platform=None):
    target_platform = resolve_target_platform(target_platform)
    driver_mode = get_driver_mode()
    model_name = "vgg10-radioml-w4a4"
    filename = find_bitfile(model_name, target_platform)
    fclk_mhz = 250.0
    return FINNExampleOverlay(
        filename,
        driver_mode,
        _radioml_io_shape_dict,
        fclk_mhz=fclk_mhz,
    )
