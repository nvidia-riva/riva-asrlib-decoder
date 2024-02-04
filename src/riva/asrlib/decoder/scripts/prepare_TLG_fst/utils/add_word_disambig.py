#!/usr/bin/env python
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

import click
@click.command()
@click.option('--word_symbols', type=click.File('a'),
              help="path to words.txt file")
@click.option('--extra_word_disamb', type=click.File('r'),
               help="path to the file containing extra word disambiguations")
@click.argument('last_idx', type=int)
def main(word_symbols, extra_word_disamb, last_idx):
    word_disam=[line.strip().split(" ")[-1] for line in extra_word_disamb]
    idx=last_idx+1
    for word in word_disam[:-1]:
        word_symbols.write(f'{word} {idx}\n')
        idx += 1
    word_symbols.write(f'{word_disam[-1]} {idx}')


if __name__=="__main__":
    main()