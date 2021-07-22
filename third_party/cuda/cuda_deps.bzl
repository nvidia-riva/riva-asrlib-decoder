# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA dependencies for Jarvis"""

def cuda_deps():
    # native.new_local_repository(
    #     name = "cuda_cuda",
    #     build_file = "external/BUILD.cuda_cuda",
    #     path = "/usr/local/cuda-11.3",
    # )

    native.new_local_repository(
        name = "cuda_cudart",
        build_file = "external/BUILD.cuda_cudart",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_cublas",
        build_file = "external/BUILD.cuda_cublas",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_nvcc",
        build_file = "external/BUILD.cuda_nvcc",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_curand",
        build_file = "external/BUILD.cuda_curand",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_nvml",
        build_file = "external/BUILD.cuda_nvml",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_cusparse",
        build_file = "external/BUILD.cuda_cusparse",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_nvtx",
        build_file = "external/BUILD.cuda_nvtx",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_cusolver",
        build_file = "external/BUILD.cuda_cusolver",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_cufft",
        build_file = "external/BUILD.cuda_cufft",
        path = "/usr/local/cuda-11.3",
    )

    native.new_local_repository(
        name = "cuda_nvprof",
        build_file = "external/BUILD.cuda_nvprof",
        path = "/usr/local/cuda-11.3",
    )
