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
#!/usr/bin/env python3

import argparse
import sys
import tempfile
from distutils.util import strtobool

PYNINI_AVAILABLE = False
PYWRAPFST_AVAILABLE = False
errors = []

try:
    import pynini as fst

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError) as e1:
    errors.append(e1)
    try:
        import pywrapfst as fst

        PYWRAPFST_AVAILABLE = True
    except (ModuleNotFoundError, ImportError) as e2:
        errors.append(e2)
finally:
    if not (PYNINI_AVAILABLE or PYWRAPFST_AVAILABLE):
        raise Exception(errors)


def str2bool(value: str) -> bool:
    return bool(strtobool(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Add disambig arcs into an input LM.""",
    )
    parser.add_argument(
        "lm_in_path",
        type=str,
        help="LM path",
    )
    parser.add_argument(
        "--lm_out_path",
        type=str,
        default="-",
        help="Out path or stdout",
    )
    parser.add_argument(
        "--disambig_num",
        type=int,
        default=3,
        help="How many disambiguators to add",
    )
    parser.add_argument(
        "--shift_labels",
        type=str2bool,
        default=True,
        help="""Whether to shift labels right by one to match T.fst
                If set to false, only disabiguators are shifted.""",
    )
    args = parser.parse_args()

    lm_in = fst.Fst.read(args.lm_in_path)
    max_label = 0
    if args.shift_labels:
        lm_out = fst.Fst()
        for state in lm_in.states():
            new_state = lm_out.add_state()
            if new_state == lm_in.start():
                lm_out.set_start(lm_in.start())
            lm_out = lm_out.set_final(new_state, lm_in.final(state))
            for arc in lm_in.arcs(state):
                new_ilabel = arc.ilabel + 1 if arc.ilabel > 0 else arc.ilabel
                new_olabel = arc.olabel + 1 if arc.olabel > 0 else arc.olabel
                lm_out.add_arc(new_state, fst.Arc(new_ilabel, new_olabel, arc.weight, arc.nextstate))
                max_label = max(max_label, new_ilabel)
    else:
        lm_out = lm_in.copy()
        for state in lm_in.states():
            if list(lm_in.arcs(state)):
                max_label = max(max_label, *[arc.ilabel for arc in lm_in.arcs(state)])
        # If we are not shifting labels, then we suppose that <blank> is the last token.
        max_label += 1

    for state in lm_in.states():
        for disambig in range(max_label + 1, max_label + args.disambig_num + 1):
            lm_out.add_arc(state, fst.Arc(disambig, disambig, 0.0, state))
    if args.lm_out_path == "-":
        tmpfile = tempfile.NamedTemporaryFile()
        lm_out.write(tmpfile.name)
        with open(tmpfile.name, "rb") as f:
            sys.stdout.buffer.write(f.read())
    else:
        lm_out.write(args.lm_out_path)
