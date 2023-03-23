#!/usr/bin/env python3
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

import itertools
import sys

if __name__ == "__main__":
    with_sefloops = len(sys.argv) <= 2 or sys.argv[2] != "false"

    with open(sys.argv[1], "r") as fread:
        print("0 0 <blk> <eps>")

        disambigs = []
        phones = []
        start = 1
        nodeX = start
        for entry in fread:
            phone = entry.strip().split()[0]
            if phone == "<eps>" or phone == "<blk>":
                continue
            # disambig phones are handled correctly then!
            if phone.startswith("#") and not phone.startswith("##"):
                disambigs.append(phone)
            else:
                phones.append(phone)
                print(f"0 {nodeX} {phone} {phone}")
                if with_sefloops:
                    print(f"{nodeX} {nodeX} {phone} <eps>")
                print(f"{nodeX} 0 <blk> <eps>")
                nodeX += 1

        for i in range(start, nodeX):
            print(i)
            for j, phone in enumerate(phones, start):
                if i != j:
                    print(f"{i} {j} {phone} {phone}")
            for disambig in disambigs:
                print(f"{i} {i} <eps> {disambig}")

        print("0")
