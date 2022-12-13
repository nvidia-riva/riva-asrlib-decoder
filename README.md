# Riva ASR Library CUDA WFST Decoder

This code repository exposes the CUDA WFST decoder originally
described here https://arxiv.org/abs/1910.10032 as a C++ library and
as a python library (via wrapping the C++ library with pybind11). It
is the same decoder used when you specify --decoder_type=kaldi when
running `riva-build` in NVIDIA Riva.

## Pre-built wheels

Install from [PyPi](https://pypi.org/project/riva-asrlib-decoder/).

```
pip install riva-asrlib-decoder
```

These manylinux2014 wheels, with the exception that they use cuda. You
need cuda 11.2.1 or greater and python 3.6 or greater.

Cuda as old as 10.2 could also be supported, but there is an
[incompatibility](https://forums.developer.nvidia.com/t/nvc-20-9-fails-to-compile-code-instantiating-any-std-tuple-with-gcc-10-2-on-c-17/160987)
with gcc-10 (which manylinux2014 uses) not fixed until cuda 11.2.1.

## Build and Test

To build just the offline binary:

```
mkdir build
cd build
cmake -DRIVA_ASRLIB_BUILD_PYTHON_BINDINGS=NO ..
cmake --build --target all --parallel
# Run build/offline-cuda-decode-binary
```

To build the python bindings from source and optionally run the python
test cases:

```
# Optionally set up a conda environment first.
pip install -e .[testing]
python -m unittest src/riva/asrlib/decoder/test_decoder.py 2>&1 | tee test.log
```


## Bazel

Bazel build is not being used. Please don't expect it to work.