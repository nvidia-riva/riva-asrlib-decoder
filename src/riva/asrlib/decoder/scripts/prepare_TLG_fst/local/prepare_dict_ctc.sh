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

# Copyright 2014 Vassil Panayotov
#           2020 ITMO University (Aleksandr Laptev)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# Prepares the dictionary and auto-generates the pronunciations for the words,
# that are in our vocabulary but not in CMUdict

units=
positional=none        # none, begin, full
use_sil=false
sil_token="<space>"   # the character you have used to represent spaces

. parse_options.sh || exit 1;
. path.sh || exit 1

# We would like to set this, but a usage of cmp later can return a non-zero
# exit code without being fatal.
# set -e
set -u
set -o pipefail


lexicon=$1
dir=$2


[ ! -f $lexicon ] && echo "$0: lexicon file not found at $lexicon" && exit 1;

mkdir -p $dir || exit 1;
echo $dir

awk '{$1=$1;print}' < $lexicon > $dir/lexicon_no_trailing_whitepsace.txt

lexicon=$dir/lexicon_no_trailing_whitepsace.txt

awk '{if ($2 !~ /[01]\.[0-9]+/) {exit 1;}}' $lexicon && probabilistic=true || probabilistic=false

case $positional in
  full)
    echo "using '$positional' positional marks"
    if [ $probabilistic = true ]; then
        perl -ane '@A=split(" ",$_); $w = shift @A; $p = shift @A; @A>0||die;
             if(@A==1) { print "$w $p $A[0]_S\n"; } else { print "$w $p $A[0]_B ";
             for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
             < $lexicon > $dir/lexicon_pos.tmp || exit 1;
    else
        perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
             if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
             for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
             < $lexicon > $dir/lexicon_pos.tmp || exit 1;
    fi
    vocab=$dir/lexicon_pos.tmp
    ;;
  begin)
    echo "using '$positional' positional marks"
    if [ $probabilistic = true ]; then
        awk '{$3="▁"$3; print}' < $lexicon > $dir/lexicon_pos.tmp || exit 1;
    else
        awk '{$2="▁"$2; print}' < $lexicon > $dir/lexicon_pos.tmp || exit 1;
    fi
    vocab=$dir/lexicon_pos.tmp
    ;;
  none)
    echo "using '$positional' positional marks"
    ;;
  *) echo "$0: invalid positional mark: '$positional'" && exit 1;
esac

if [ -z $units ]; then
    if [ ${use_sil} = true ]; then
        if [ $probabilistic = true ]; then
            cut -d" " -f3- $lexicon | tr ' ' '\n' | sort -u | sed "/^$sil_token$/d" > $dir/units_nosil.txt
        else
            cut -d" " -f2- $lexicon | tr ' ' '\n' | sort -u | sed "/^$sil_token$/d" > $dir/units_nosil.txt
        fi
    else
        if [ $probabilistic = true ]; then
            cut -d" " -f3- $lexicon | tr ' ' '\n' | sort -u > $dir/units_nosil.txt
        else
            cut -d" " -f2- $lexicon | tr ' ' '\n' | sort -u > $dir/units_nosil.txt
        fi
    fi

    #  The complete set of lexicon units, indexed by numbers starting from 1
    if [ ${use_sil} = true ]; then
      (echo '[UNK]'; echo $sil_token; ) | cat - $dir/units_nosil.txt | awk '{print $1}' > $dir/units.txt
    else
      echo '[UNK]' | cat - $dir/units_nosil.txt | awk '{print $1}' > $dir/units.txt
    fi

    units=$dir/units.txt

    echo "units.txt done"
else
    echo "using existing units: '$units'"
    sed 's/<unk>/[UNK]/g' $units > $dir/units.txt
fi

if [ $probabilistic = true ]; then
    (echo '<spoken_noise> 1.0 [UNK]'; echo '<unk> 1.0 [UNK]'; ) | cat - $lexicon | sort -k1,1r -k2,2n | tac | uniq > $dir/lexiconp.txt || exit 1;
    echo "probabilistic lexicon filtering is not supported"
    cut -d" " -f1,3- $dir/lexiconp.txt > $dir/lexicon.txt
else
    # Do not understand. Too hard to read!!!!
    (echo '<spoken_noise> [UNK]'; echo '<unk> [UNK]'; ) | cat - $lexicon | sed 's/\t/ /g' | sort | uniq > $dir/lexicon.tmp || exit 1;
    # It has a problem when the lexicon does not fully cover the units, it seems...
    # But we cannot always guarantee that this is the case anyway...
    cut -d" " -f2- $dir/lexicon.tmp > $dir/1.txt
    cut -d" " -f2- $dir/lexicon.tmp | tr ' ' '\n' > $dir/2.txt
    cut -d" " -f2- $dir/lexicon.tmp | tr ' ' '\n' | sort -u > $dir/units_from_lexicon_sorted.txt
    sort $dir/units.txt > $dir/units_sorted.txt
    cmp $dir/units_sorted.txt $dir/units_from_lexicon_sorted.txt
    if [ $? -ne 0 ]; then
        echo "WARNING: Difference in units.txt and units derived from lexicon!"
        echo "This means that some units will never be output, which may be okay."
    fi
    comm -23 $dir/units_from_lexicon_sorted.txt $dir/units_sorted.txt > $dir/units_in_lexicon_only.txt
    grep -vF --file=$dir/units_in_lexicon_only.txt  $dir/lexicon.tmp > $dir/lexicon.txt
    [ -s $dir/lexicon.txt ] && cmp --silent $dir/lexicon.tmp $dir/lexicon.txt || echo "Something wrong with the final lexicon. Check units for consistency.";
fi

echo "lexicon.txt done"
