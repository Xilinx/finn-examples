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

# collect all local folders, each are considered build folders
LOCAL_BUILD_FOLDERS=$(find . -maxdepth 1 -type d -printf "%P ")

# remove trailing spaces and store the directory names in an array
IFS=' ' read -r -a BUILD_FOLDERS <<< "$LOCAL_BUILD_FOLDERS"

# fetch all models (if there are models to fetch), continue on error
for BUILD_FOLDER in ${BUILD_FOLDERS[@]}; do
    if [ -d "$SCRIPTPATH/$BUILD_FOLDER/models" ]; then
        cd $SCRIPTPATH/$BUILD_FOLDER/models
        rm -rf *.zip *.onnx *.npz
        ./download-model.sh || true
    fi
done

# run all build scripts, continue on error
cd $SCRIPTPATH/finn
for BUILD_FOLDER in ${BUILD_FOLDERS[@]}; do
    ./run-docker.sh build_custom $SCRIPTPATH/$BUILD_FOLDER || true
done

# gather all release folders, continue on error
RELEASE_TARGET=$SCRIPTPATH/release
mkdir -p $RELEASE_TARGET
for BUILD_FOLDER in ${BUILD_FOLDERS[@]}; do
    cp -r $SCRIPTPATH/$BUILD_FOLDER/release/* $RELEASE_TARGET || true
done

# create zip files for finn-examples upload
cd $RELEASE_TARGET
rm -rf *.zip
for dir in */; do
    # remove trailing slash to get the directory name
    dir_name="${dir%/}"

    # check if it is a directory we are zipping
    if [ -d "$dir_name" ]; then
        zip -r "${dir_name}.zip" "$dir_name" || true
    fi
done

# calculate the MD5sum for each of the zip files
zip_files=(*.zip)
for zip_file in "${zip_files[@]}"; do
    md5sum_value=$(md5sum "$zip_file" | awk '{print $1}')
    echo "$zip_file : $md5sum_value" >> md5sum.log
done
