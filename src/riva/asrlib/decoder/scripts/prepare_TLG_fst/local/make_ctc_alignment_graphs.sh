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

. ./path.sh || exit 1;

topo_lm_path=               # Inject an LM into the topology, if provided.

. local/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

langdir=$1
src_dir=$2
dest_dir=$3

t_suffix=

if [ ! -z ${topo_lm_path} ]; then
    log "Injecting topo_lm into T.fst ..."
    t_suffix=_$(basename ${topo_lm_path} .fst)
    blank_first=$(cat ${langdir}/blank_first)
    if [ ! -f ${langdir}/T${t_suffix}.fst ]; then
        fstcompose ${langdir}/T.fst <(local/add_lm_disambig.py ${topo_lm_path} --disambig_num=$(grep "#[[:digit:]]" ${langdir}/tokens.txt | wc -l) --shift_labels=${blank_first}) > ${langdir}/T${t_suffix}.fst
    fi
fi

log "Preparing T${t_suffix}L.fst ..."
fsttablecompose ${langdir}/T${t_suffix}.fst <(fstarcsort --sort_type=ilabel ${langdir}/L_disambig.fst) | fstarcsort --sort_type=olabel > ${langdir}/T${t_suffix}L.fst

log "Preparing soft alignment graphs ..."
mkdir -p ${dest_dir}
sym2int.pl --map-oov '<unk>' -f 2- ${langdir}/words.txt ${src_dir}/text | transcripts-to-fsts --left-compose=${langdir}/T${t_suffix}L.fst ark,t:- ark,scp:${dest_dir}/num.fsts,${dest_dir}/num.scp
cp -rf ${langdir}/{T.fst,words.txt,blank_first} ${dest_dir}/
cp -rf ${langdir}/phones ${dest_dir}/phones

log "Done"
