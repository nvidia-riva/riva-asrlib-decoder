#!/usr/bin/env python3

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

import torch
import nemo.collections.asr as nemo_asr
import click
from arpa2wordlist import unigram2words
from pathlib import Path
import tqdm


@click.command()
@click.option('--asr_model', type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="path to asr nemo model")
@click.option('--lm_path', type=click.File('r'), help="path to lm arpa file")
@click.option('--grammars', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=False,
              help="path to folder containing grammar text")
@click.argument('units', type=click.File('w'))
@click.argument('lexicon', type=click.File('w'))
@click.argument('extra_disamb', type=click.File('w'), required=False)
def main(asr_model, lm_path, lexicon, units, grammars=None, extra_disamb=None):
    #Ensure  all entries in LM have pronunciations/tokens in  lexicon
    print("Finding words in LM:")
    word_list, entity_list, skipped_list = unigram2words(lm_path)
    p = Path(grammars)

    #Ensure  all entries in grammars have pronunciations/tokens in  lexicon
    for grammar_file in p.glob("*.txt"):
        with open(str(grammar_file),'r') as grm_fp:
            for line in grm_fp:
                words=line.strip().split()
                for word in words:
                    if word not in word_list:
                        if word[0] != '#' and word[0] != '<':
                            word_list.append(word)

    word_list.sort()
    entity_list.sort()

    asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        restore_path=asr_model
    ).to(torch.device("cpu"))

    tokenizer = asr_model.tokenizer


    pbar = tqdm.tqdm(total=len(word_list))

    for word in word_list:
        pbar.update(1)
        entry_lex = " ".join(tokenizer.text_to_tokens(word.strip()))
        lexicon.write(f"{word.strip()} {entry_lex}\n")
    units.write("\n".join(tokenizer.vocab))
    if len(entity_list)>0:
        if extra_disamb is not None:
            for symbol in entity_list:
                extra_disamb.write(f"{symbol}\n")
        else:
            print("ERROR: Entity list is not empty and no disambiguation file specified. This is not normal.")
            print("Outputing to screen")
            print('\n'.join(entity_list))


if __name__=="__main__":
    main()
