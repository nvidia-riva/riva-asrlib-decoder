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

# Copyright 2020 ITMO University (Aleksandr Laptev)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

. $SCRIPT_DIR/path.sh || exit 1;

stage=1                     # Processes starts from the specified stage.
stop_stage=10000            # Processes is stopped at the specified stage.
topo="ctc"                  # The type of neural network topology. Choices: "ctc", "ctc_eesen", "ctc_compact", "identity"
topo_with_sefloops=true     # Whether to build the topology with token self-loops. Used with "ctc", "ctc_eesen", and "ctc_compact" topologies.
blank_first=false           # Whether to put '<blk>' at the beginning in token list or at the end.
use_space=false             # Whether to inclure space token.
space_char="<space>"        # The token you have used to represent spaces. Use "sil" for phonemes.
positional=none             # Choices: none, begin, full
lm_lowercase=false          # Whether to lowercase the LM.
units=                      # Use pre-defined units, if provided.
carpa_lm_path=              # Compile G.carpa, if provided.
const_graph=true            # Whether to convert TLG.fst to constant type.
topo_lm_path=               # Inject an LM into the topology, if provided.

. parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

lexicon_path=$1
lm_path=$2
dest_dir=$3

if [ -z $units ]; then
    log "--units argument must be specified"
    exit 1
fi


mkdir -p ${dest_dir}

suffix=${topo}
if [ ! ${topo_with_sefloops} = true ]; then
    suffix=${suffix}_no_selfloops
fi
t_suffix=
if [ ! -z ${topo_lm_path} ]; then
    t_suffix=_$(basename ${topo_lm_path} .fst)
    suffix=${suffix}${t_suffix}
fi
graphdir=${dest_dir}/graph_${suffix}_$(basename ${lm_path} .arpa.gz)
langdir=${dest_dir}/lang_${suffix}
dict_dir=${dest_dir}/dict_${suffix}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Preparing dict ..."
    prepare_dict_ctc.sh --positional=${positional} --use_sil=${use_space} --sil_token=${space_char} --units=${units} ${lexicon_path} ${dict_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Preparing T.fst and L.fst ..."
    ctc_compile_dict_token.sh --topo=${topo} --topo_with_sefloops=${topo_with_sefloops} --use_space=${use_space} --space_char=${space_char} --blank_first=${blank_first} ${dict_dir} ${langdir}_tmp ${langdir}
    if [ ! -z ${topo_lm_path} ]; then
        log "Injecting topo_lm into T.fst ..."
        fstcompose ${langdir}/T.fst <(add_lm_disambig.py ${topo_lm_path} --disambig_num=$(grep "#[[:digit:]]" ${langdir}/tokens.txt | wc -l) --shift_labels=${blank_first}) > ${langdir}/T${t_suffix}.fst
    fi
    if [ ${blank_first} = true ]; then
        echo "true" > ${langdir}/blank_first
    else
        echo "false" > ${langdir}/blank_first
    fi
fi

if [ ${lm_lowercase} = true ]; then
    lm_command="tr '[:upper:]' '[:lower:]'"
else
    lm_command="cat"
fi

mkdir -p ${graphdir} && cp -r ${langdir}/* ${graphdir}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Preparing G.fst ..."
    gunzip -c ${lm_path} | $lm_command | arpa2fst --disambig-symbol=#0 --read-symbol-table=${graphdir}/words.txt - ${graphdir}/G.fst
fi

if [ ${const_graph} = true ]; then
    post_proc_command="fstconvert --fst_type=const"
else
    post_proc_command="cat"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Building a \"${suffix}\" TLG.fst ..."
    # fstdeterminizestar --use-log=true ${graphdir}/L_disambig.fst | fstminimizeencoded | fstarcsort --sort_type=olabel > ${graphdir}/L_disambig_detmin.fst
    fsttablecompose ${graphdir}/L_disambig.fst ${graphdir}/G.fst | fstdeterminizestar --use-log=true | \
        fstminimizeencoded | fstarcsort --sort_type=ilabel > ${graphdir}/LG.fst
    # TODO: Why is this not determinized?
    # Also, we are not removing the #0 symbol... Or any other disambig symbol?
    # fstrmsymbols $dir/disambig_tid.int | \
    fsttablecompose ${graphdir}/T${t_suffix}.fst ${graphdir}/LG.fst | \
        $post_proc_command > ${graphdir}/TLG.fst
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ ! -z ${carpa_lm_path} ]; then
        if [ -f ${carpa_lm_path} ]; then
            carpa_lm_basename=$(basename ${carpa_lm_path})
            log "Preparing G.carpa from ${carpa_lm_basename} ..."
            bos=`grep "^<s>\s" ${graphdir}/words.txt | awk '{print $2}'`
            eos=`grep "^</s>\s" ${graphdir}/words.txt | awk '{print $2}'`
            if [[ -z $bos || -z $eos ]]; then
                log "<s> and </s> symbols are not in ${graphdir}/words.txt"; exit 1
            fi
            arpa-to-const-arpa --bos-symbol=$bos --eos-symbol=$eos \
                "gunzip -c ${carpa_lm_path} | $lm_command | map_arpa_lm.pl ${graphdir}/words.txt|" \
                ${graphdir}/G_${carpa_lm_basename%.*}.carpa || exit 1;
            ln -s ${graphdir}/G_${carpa_lm_basename%.*}.carpa ${graphdir}/G.carpa
        else
            log "${carpa_lm_path} provided but is not valid"; exit 1
        fi
    else
        log "Preparing G.carpa skipped."
    fi
fi

log "Done"
