#  Copyright (C) 2020 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from setuptools import find_packages, setup

import os
import zipfile
from distutils.command.build import build as dist_build
from pynq.utils import build_py as _build_py

__author__ = "Yaman Umuroglu"
__copyright__ = "Copyright 2020, Xilinx"
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
            for (dirpath, dirnames, filenames) in os.walk(build_dir):
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
            [
                os.path.join("..", root, f)
                for root, _, files in os.walk(path)
                for f in files
            ]
        )
    elif os.path.isfile(path):
        data_files.append(os.path.join("..", path))


with open("README.md", encoding="utf-8") as fh:
    readme_lines = fh.readlines()[4:]
long_description = "".join(readme_lines)
extend_package(os.path.join(module_name, "bitfiles"))
extend_package(os.path.join(module_name, "notebooks"))

setup(
    name=module_name,
    version="0.0.1b",
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
    setup_requires=["pynq>=2.5.1"],
    install_requires=[
        "pynq>=2.5.1",
        "finn-base==0.0.1b0",
        "finn-dataset_loading==0.0.4",  # noqa
    ],
    extras_require={
        ':python_version<"3.6"': ["matplotlib<3.1", "ipython==7.9"],
        ':python_version>="3.6"': ["matplotlib"],
    },
    entry_points={
        "pynq.notebooks": ["finn_examples = {}.notebooks".format(module_name)]
    },
    cmdclass={"build_py": build_py, "unzip_overlays": _unzip_overlays},
    license="Apache License 2.0",
)
