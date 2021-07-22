#!/bin/bash

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

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/path.sh || exit 1;

target="words"          # The type of the lexicon's first column. Choices: "words", "wp"
vocab=                  # Word vocabulary
tokenizer_dir=
model_path=
tokenizer_type=         # The type of the tokenizer. Choices: "bpe", "wpe"
do_lowercase=false      # Whether to lowercase vocab.

. parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

out_file=$1


if [ ${target} != "words" ] && [ ${target} != "wp" ]; then
    log "Unsupported target: ${target}" && exit 1;
fi

if [ ${target} = "wp" ]; then
    log "Preparing purely wordpiece lexicon"

    if [ -z ${tokenizer_dir} ]; then
        log "tokenizer_dir must be set" && exit 1;
    fi
    awk '{print $1,$1}' ${tokenizer_dir}/vocab.txt > ${out_file}
else
    log "Preparing wordpiece to words lexicon"

    num_error=0
    if [ -z ${vocab} ]; then
        log "Vocab must be set"
        num_error=$((num_error + 1))
    fi
    if [ ! -f ${vocab} ]; then
        log "Vocab not found: ${vocab}"
        num_error=$((num_error + 1))
    fi
    [ $num_error -gt 0 ] && exit 1;

    if [ ${do_lowercase} = true ]; then
        do_lowercase_cmd1="--do_lowercase"
        do_lowercase_cmd2="tr '[:upper:]' '[:lower:]'"
    else
        do_lowercase_cmd1=""
        do_lowercase_cmd2="cat"
    fi

    if [ ! -z ${tokenizer_dir} ]; then
        subw_tokenize_text.py ${do_lowercase_cmd1} --text_file=${vocab} --tokenizer_dir=${tokenizer_dir} --tokenizer_type=${tokenizer_type} --output_file=${out_file}
        transcript=$(cat ${out_file}) || exit 1;
    elif [ ! -z ${model_path} ]; then
        subw_tokenize_text.py ${do_lowercase_cmd1} --text_file=${vocab} --nemo_model=${model_path} --output_file=${out_file}
        transcript=$(cat ${out_file}) || exit 1;
    else
        log "Either --model_path or --tokenizer_dir and --tokenizer_type must be set"
    fi
    paste -d' ' <(cat ${vocab} | ${do_lowercase_cmd2}) <(echo "${transcript}") > ${out_file}
fi

log "Done"
