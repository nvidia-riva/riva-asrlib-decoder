#########################################################################################
## Get manylinux image with any neccessary DALI dependencies.
## It is possible to just use defaults and have a pure manylinux2014 with CUDA on top
## DALI is based on "manylinux2014", official page https://github.com/pypa/manylinux
#########################################################################################
ARG CUDA_VER=11.2.2
ARG ARCH=x86_64
ARG FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_x86_64
ARG CUDA_IMAGE=nvidia/cuda:${CUDA_VER}-devel-centos7 # nvidia/dali:cuda${CUDA_VER}_${ARCH}.toolkit
FROM ${CUDA_IMAGE} as cuda
FROM ${FROM_IMAGE_NAME}

ENV PATH=/usr/local/cuda/bin:${PATH}

ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# CUDA
COPY --from=cuda /usr/local/cuda /usr/local/cuda
