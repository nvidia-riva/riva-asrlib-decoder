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

# Copyright 2015       Yajie Miao        (Carnegie Mellon University)
#           2020       Aleksandr Laptev  (ITMO University)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# This script compiles the lexicon and CTC tokens into FSTs. FST compiling slightly differs between the
# phoneme and character-based lexicons.

topo="ctc"                  # The type of neural network topology. Choices: "ctc", "ctc_eesen", "ctc_compact", "identity"
topo_with_sefloops=true     # Whether to build the topology with token self-loops. Used with "ctc", "ctc_compact" topologies.
blank_first=false           # Whether to put '<blk>' at the beginning in token list or at the end.
use_space=false             # Whether to use optional silence
space_char="<SPACE>"        # The character you have used to represent spaces

. parse_options.sh

set -e
set -u
set -o pipefail

if [ $# -ne 3 ]; then
  echo "usage: local/ctc_compile_dict_token.sh <dict-src-dir> <tmp-dir> <lang-dir>"
  echo "e.g.: local/ctc_compile_dict_token.sh data/local/dict_phn data/local/lang_phn_tmp data/lang_phn"
  echo "<dict-src-dir> should contain the following files:"
  echo "lexicon.txt units.txt"
  echo "options: "
  echo "     --blank-first <true or false>                   # default: false."
  echo "     --use-space <true or false>                     # default: false."
  echo "     --space-char <space character>                  # default: <SPACE>, the character to represent spaces."
  echo "     --topo <topology type>                          # default: ctc, Choices: ctc, ctc_eesen, ctc_compact, identity."
  echo "     --topo-with-sefloops <true or false>            # default: true."
  exit 1;
fi

srcdir=$1
tmpdir=$2
dir=$3
mkdir -p $dir/phones $tmpdir

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
. $SCRIPT_DIR/../path.sh

cp $srcdir/{lexicon.txt,units.txt} $dir

if [ -f $srcdir/lexiconp.txt ]; then
    cp $srcdir/lexiconp.txt $tmpdir/lexiconp.txt
    cp $srcdir/lexiconp.txt $dir/lexiconp.txt
else
    # Add probabilities to lexicon entries. There is in fact no point of doing this here since all the entries have 1.0.
    # But make_lexicon_fst.pl requires a probabilistic version, so we just leave it as it is. 
    perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $srcdir/lexicon.txt > $tmpdir/lexiconp.txt || exit 1;
fi

# First remove pron-probs from the lexicon.
perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' <$tmpdir/lexiconp.txt >$tmpdir/align_lexicon.txt
# TODO: generalize
echo "<eps> <blk>" >> $tmpdir/align_lexicon.txt
cat $tmpdir/align_lexicon.txt | \
  perl -ane '@A = split; print $A[0], " ", join(" ", @A), "\n";' | sort | uniq > $dir/phones/align_lexicon.txt

# Add disambiguation symbols to the lexicon. This is necessary for determinizing the composition of L.fst and G.fst.
# Without these symbols, determinization will fail. 
ndisambig=`add_lex_disambig.pl $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt`
ndisambig=$[$ndisambig+1];

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $tmpdir/disambig.list
cp -rf $tmpdir/disambig.list $dir/phones/disambig.txt

# Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>, the actual labels (e.g.,
# phonemes), and the disambiguation symbols. 
cat $srcdir/units.txt | awk '{print $1}' > $tmpdir/units.list
if [ $blank_first = true ]; then
  (echo '<eps>'; echo '<blk>';) | cat - $tmpdir/units.list $tmpdir/disambig.list | awk '{print $1 " " (NR-1)}' > $dir/tokens.txt
else
  (echo '<eps>'; ) | cat - $tmpdir/units.list <(echo '<blk>'; ) $tmpdir/disambig.list | awk '{print $1 " " (NR-1)}' > $dir/tokens.txt
fi

case $topo in
  ctc)
    token_fst_cmd=ctc_token_fst.py
    ;;
  ctc_eesen)
    token_fst_cmd=ctc_eesen_token_fst.py
    ;;
  ctc_compact)
    token_fst_cmd=ctc_compact_token_fst.py
    ;;
  identity)
    token_fst_cmd=identity_token_fst.py
    ;;
  *) echo "$0: invalid topology $topo" && exit 1;
esac

# Just for debugging purposes
${token_fst_cmd} $dir/tokens.txt $topo_with_sefloops > $dir/T.txt

${token_fst_cmd} $dir/tokens.txt $topo_with_sefloops | \
    fstcompile --isymbols=$dir/tokens.txt --osymbols=$dir/tokens.txt --keep_isymbols=false --keep_osymbols=false | \
    fstarcsort --sort_type=olabel > $dir/T.fst || exit 1;

# Encode the words with indices. Will be used in lexicon and language model FST compiling.
cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | uniq | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' > $dir/words.txt || exit 1;

cat $dir/phones/align_lexicon.txt | sym2int.pl -f 3- $dir/tokens.txt | \
  sym2int.pl -f 1-2 $dir/words.txt > $dir/phones/align_lexicon.int

# Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time.
token_disambig_symbol=`grep \#0 $dir/tokens.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $dir/words.txt | awk '{print $2}'`

if [ $use_space = true ]; then
  silprob=0.5
else
  silprob=0
fi

make_lexicon_fst.pl --pron-probs $tmpdir/lexiconp_disambig.txt $silprob "$space_char" '#'$ndisambig | \
  fstcompile --isymbols=$dir/tokens.txt --osymbols=$dir/words.txt \
  --keep_isymbols=false --keep_osymbols=false |   \
  fstaddselfloops  "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
  fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;

echo "Dict and token FSTs compiling succeeded"
