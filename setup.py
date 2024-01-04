# Copyright (c) 2020-2023 Xilinx, Inc.
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

from setuptools import find_packages, setup

import os
import zipfile
from distutils.command.build import build as dist_build
from pynqutils.setup_utils import build_py as _build_py

__author__ = "Yaman Umuroglu"
__copyright__ = "Copyright 2020-2021, Xilinx"
__email__ = "yamanu@xilinx.com"


# global variables
module_name = "finn_examples"
data_files = []


def unzip_to_same_folder(zipfile_path):
    dir_path = os.path.dirname(os.path.realpath(zipfile_path))
    with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
        zip_ref.extractall(dir_path)


class _unzip_overlays(dist_build):
    """Custom distutils command to unzip downloaded overlays."""

    description = "Unzip downloaded overlays"
    user_options = []
    boolean_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cmd = self.get_finalized_command("build_py")
        for package, f, build_dir, _ in cmd.data_files:
            for dirpath, dirnames, filenames in os.walk(build_dir):
                for f in filenames:
                    if f.endswith(".zip"):
                        zip_path = dirpath + "/" + f
                        print("Extracting " + zip_path)
                        unzip_to_same_folder(zip_path)


class build_py(_build_py):
    """Overload the PYNQ 'build_py' command to also call the
    command 'unzip_overlays'.
    """

    def run(self):
        super().run()
        self.run_command("unzip_overlays")


def extend_package(path):
    if os.path.isdir(path):
        data_files.extend(
            [os.path.join("..", root, f) for root, _, files in os.walk(path) for f in files]
        )
    elif os.path.isfile(path):
        data_files.append(os.path.join("..", path))


with open("README.md", encoding="utf-8") as fh:
    readme_lines = fh.readlines()[4:]
long_description = "".join(readme_lines)
extend_package(os.path.join(module_name, "bitfiles"))
extend_package(os.path.join(module_name, "data"))
extend_package(os.path.join(module_name, "notebooks"))

setup(
    name=module_name,
    use_scm_version=True,
    description="FINN Examples on PYNQ for Zynq and Alveo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yaman Umuroglu",
    author_email="yamanu@xilinx.com",
    url="https://github.com/Xilinx/finn-examples",
    packages=find_packages(),
    download_url="https://github.com/Xilinx/finn-examples",
    package_data={
        "": data_files,
    },
    python_requires=">=3.5.2",
    # keeping 'setup_requires' only for readability - relying on
    # pyproject.toml and PEP 517/518
    setup_requires=["pynq>=2.5.1", "setuptools_scm"],
    install_requires=[
        "pynq>=2.5.1",
        "bitstring>=3.1.7",
        "numpy",
        "finn-dataset_loading==0.0.5",  # noqa
    ],
    extras_require={
        ':python_version<"3.6"': ["matplotlib<3.1", "ipython==7.9"],
        ':python_version>="3.6"': ["matplotlib"],
    },
    entry_points={"pynq.notebooks": ["finn_examples = {}.notebooks".format(module_name)]},
    cmdclass={"build_py": build_py, "unzip_overlays": _unzip_overlays},
    license="Apache License 2.0",
)
