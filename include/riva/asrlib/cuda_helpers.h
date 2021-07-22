/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cuda_runtime_api.h>

namespace riva::asrlib {

bool cudaCheckError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'\n", file, line,
               (int)code, cudaGetErrorString(code), expr);
        return true;
    }
    return false;
}

#define RIVA_ASRLIB_CUDA_CHECK_ERR(...)                                                        \
    do {                                                                         \
        bool err = ::riva::asrlib::cudaCheckError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
    if (err) {                                                               \
        exit(1);                                                        \
    }                                                                        \
    } while (0)

}  // namespace riva::asrlib
