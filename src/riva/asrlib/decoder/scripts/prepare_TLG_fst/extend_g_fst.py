# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script iteratively parses the grammar folder to compose fst for each of the grammars nonterms
# fileformat: <entity_name>.txt and does 'fstrepace' on each of the grammar nonterms (#entitY:<entity_name>)
# in G.fst


import pywrapfst as fst
import click
from pathlib import Path
from expand_grammarfst import expand_grammar


@click.command()
@click.option('--word_symbols', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
              help="path to words.txt file")
@click.argument('g_fst_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True))
              # help="path to G.fst file containing nonterms '#entity:<entity_name>"
@click.argument('grammars', type=click.Path(exists=True, file_okay=False, dir_okay=True))
              # help="path to folder containing grammars in '<entity_name>.txt' files"
def main(word_symbols, g_fst_file, grammars):
    print(word_symbols, g_fst_file, grammars)
    isym = fst.SymbolTable.read_text(word_symbols)
    g_root_fst = fst.VectorFst().read(g_fst_file)
    print(g_root_fst.weight_type())
    p = Path(grammars)
    for grammar_file in p.glob("*.txt"):
        print(str(grammar_file))
        g_root_fst = expand_grammar(isym, g_root_fst, str(grammar_file))
    g_root_fst.rmepsilon()
    g_root_fst.minimize()
    g_root_fst.arcsort()
    g_root_fst.write(g_fst_file+'.expanded')

if __name__=="__main__":
    main()
