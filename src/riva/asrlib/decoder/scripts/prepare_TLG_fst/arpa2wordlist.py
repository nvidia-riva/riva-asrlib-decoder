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

import sys
import parse
import os
import tqdm

## Generate wordlist from the unigrams present in arpa lm file

VOCAB_FILENAME = "words.txt"
ENTITIES_FILENAME = "entities.txt"

UNIGRAM_HDR = "\\1-grams:"
BIGRAM_HDR = "\\2-grams:"
TRIGRAM_HDR = "\\3-grams:"

# arpa format <log_prob>    <n-gram>  <log_backoff>
UNIGRAM_FORMAT = "{log_prob:g}\t{n_gram:S}\t{log_backoff:g}"
ENTITY_PREFIX="#entity:"
EXCLUSIONS=["<s>", "</s>", "<unk>"]

def get_unigram_count(fp):
    NGRAM_COUNT="ngram {ngram:d}={count:d}"
    unigram_cnt=0
    for line in fp:
        ngram_count=parse.parse(NGRAM_COUNT,line.strip())
        if ngram_count is None:
            continue
        else:
            if ngram_count.named["ngram"] != 1:
                print("Missing ungram count")
                exit(1)
            else:
                unigram_cnt = ngram_count.named["count"]
                break
    for line in fp:
        if UNIGRAM_HDR not in line:
            continue
        else:
            break
    return unigram_cnt
def unigram2words(arpa_file: str, vocab_folder:str):
    word_list = []
    entity_list = []
    skipped_list=[]
    with open(arpa_file, 'r') as fp:
        unigram_count=get_unigram_count(fp)
        pbar=tqdm.tqdm(total=unigram_count)
        for line in fp:
            pbar.update(1)
            if (BIGRAM_HDR in line) or (TRIGRAM_HDR in line):
                break

            n = parse.parse(UNIGRAM_FORMAT, line.strip())

            if n is None:
                continue
            unigram=n.named['n_gram']

            if unigram not in word_list:
                if unigram not in EXCLUSIONS:
                    if " " not in unigram:
                        if unigram[0]=="<" and unigram[-1]==">":
                            skipped_list.append(unigram)
                        elif ENTITY_PREFIX in unigram:
                            entity_list.append(unigram)
                        else:
                            word_list.append(unigram)
    os.makedirs(vocab_folder, exist_ok=True)
    with open(f"{vocab_folder}/words.txt",'w') as words_file:
        words_file.write("\n".join(word_list))
    with open(f"{vocab_folder}/entities.txt",'w') as entity_file:
        entity_file.write("\n".join(entity_list))
    with open(f"{vocab_folder}/skipped.txt",'w') as skipped_file:
        skipped_file.write("\n".join(skipped_list))


if __name__ == "__main__":
    arpa_file = sys.argv[1]
    vocab_folder = sys.argv[2]
    unigram2words(arpa_file, vocab_folder)
