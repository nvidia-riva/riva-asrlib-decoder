#!/bin/bash

set -euo pipefail

export CIBW_MANYLINUX_X86_64_IMAGE=cuda_11.2.1_manylinux_x86_64:latest
export CIBW_BUILD_VERBOSITY=3
# export CIBW_ENVIRONMENT='RIVA_ASRLIB_DECODER_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release" CUDAARCHS="52-real;60-real;61-real;70-real;75-real;80-real;86" pythonLocation="$(readlink -f $(which python))/.."'

docker build -t $CIBW_MANYLINUX_X86_64_IMAGE .

# pipx run
cibuildwheel --platform linux
