#!/bin/bash
# Copyright (C) 2020-2022, Xilinx
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
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

# URL for git repo to be cloned
REPO_URL=https://github.com/Xilinx/finn
# commit hash for repo
REPO_COMMIT=39fb8859fec0e47276ffadcafe43092d1b10af7e
# directory (under the same folder as this script) to clone to
REPO_DIR=finn


# absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=$(readlink -f "$0")
# absolute path this script is in, thus /home/user/bin
SCRIPTPATH=$(dirname "$SCRIPT")
# absolute path for the repo local copy
CLONE_TO=$SCRIPTPATH/$REPO_DIR

# clone repo if dir not found
if [ ! -d "$CLONE_TO" ]; then
  git clone $REPO_URL $CLONE_TO
fi
git -C $CLONE_TO pull
# checkout the expected commit
git -C $CLONE_TO checkout $REPO_COMMIT
# verify
CURRENT_COMMIT=$(git -C $CLONE_TO rev-parse HEAD)
if [ $CURRENT_COMMIT == $REPO_COMMIT ]; then
  echo "Successfully checked out $REPO_DIR at commit $CURRENT_COMMIT"
else
  echo "Could not check out $REPO_DIR. Check your internet connection and try again."
fi
