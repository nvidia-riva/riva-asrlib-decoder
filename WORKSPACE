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

workspace(name = "com_nvidia_cuda_wfst_speech_decoder")
load("//third_party/cuda:cuda_deps.bzl", "cuda_deps")
cuda_deps()

local_repository(
    name = "build_bazel_rules_cuda",
    path = "third_party/build_bazel_rules_cuda",
)
load("@build_bazel_rules_cuda//gpus:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
    ],
)

http_archive(
    name = "openfst",
    build_file = "BUILD.openfst",
    sha256 = "21c3758ff8574dedaf22b0a6b88c2829bbf3b2e59fbf04740ce2cf92029a0f3b",
    strip_prefix = "openfst-1.7.2",
    url = "http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "bazel_skylib",
        urls = [
                "https://github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
                        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.1.1/bazel-skylib-1.1.1.tar.gz",
                            ],
                                sha256 = "c6966ec828da198c5d9adbaa94c05e3a1c7f21bd012a0b29ba8ddbccb2c93b0d",
                                )
                                load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
                                bazel_skylib_workspace()
                                