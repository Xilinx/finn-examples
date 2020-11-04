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

from setuptools import setup, find_packages
import os
from pynq.utils import build_py


__author__ = "Yaman Umuroglu"
__copyright__ = "Copyright 2020, Xilinx"
__email__ = "yamanu@xilinx.com"


# global variables
module_name = "finn_examples"
data_files = []


def extend_package(path):
    if os.path.isdir(path):
        data_files.extend(
            [os.path.join("..", root, f)
             for root, _, files in os.walk(path) for f in files]
        )
    elif os.path.isfile(path):
        data_files.append(os.path.join("..", path))

with open("README.md", encoding="utf-8") as fh:
    readme_lines = fh.readlines()[4:]
long_description = ("".join(readme_lines))

extend_package(os.path.join(module_name, "notebooks"))
setup(name=module_name,
      version="1.0.0",
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
      setup_requires=[
          "pynq>=2.5.1"
      ],
      install_requires=[
          "pynq>=2.5.1",
      ],
      extras_require={
          ':python_version<"3.6"': [
              'matplotlib<3.1',
              'ipython==7.9'
          ],
          ':python_version>="3.6"': [
              'matplotlib'
          ]
      },
      cmdclass={"build_py": build_py},
      license="Apache License 2.0"
      )
