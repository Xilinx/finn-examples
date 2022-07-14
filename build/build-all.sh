#!/bin/bash
# Copyright (c) 2022, Xilinx
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
# * Neither the name of FINN nor the names of its
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


# absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
# subdirs for all finn-examples build folders
BUILD_FOLDERS="bnn-pynq kws mobilenet-v1 resnet50 vgg10-radioml"
# all HW platforms we build for
PLATFORMS="Pynq-Z1 Ultra96 ZCU104 U250"

# fetch correct compiler version
cd $SCRIPTPATH
bash get-finn.sh


# fetch all models, continue on error
for BUILD_FOLDER in $BUILD_FOLDERS; do
    cd $SCRIPTPATH/$BUILD_FOLDER/models
    rm -rf *.zip *.onnx *.npz
    ./download-model.sh || true
done

# run all build scripts, continue on error
cd $SCRIPTPATH/finn
for BUILD_FOLDER in $BUILD_FOLDERS; do
    ./run-docker.sh build_custom $SCRIPTPATH/$BUILD_FOLDER || true
done

# gather all release folders, continue on error
RELEASE_TARGET=$SCRIPTPATH/release
mkdir -p $RELEASE_TARGET
for BUILD_FOLDER in $BUILD_FOLDERS; do
    cp -r $SCRIPTPATH/$BUILD_FOLDER/release/* $RELEASE_TARGET || true
done

# create zipfiles for finn-examples upload
cd $SCRIPTPATH/$BUILD_FOLDER/release
rm -rf *.zip
for PLATFORM in $PLATFORMS; do
    zip -r $PLATFORM.zip $PLATFORM/ || true
    MD5SUM=$(md5sum $PLATFORM.zip)
    echo "$PLATFORM.zip : $MD5SUM" >> md5sum.log
done
