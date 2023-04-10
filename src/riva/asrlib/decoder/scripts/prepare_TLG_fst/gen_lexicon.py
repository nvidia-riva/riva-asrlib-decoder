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
import sys

model_path = "models/speechtotext_en_us_conformer_xl_vtrainable_v4.0/"
asr_model_name = model_path + "Conformer-CTC-XL_spe-128_en-US_Riva-ASR-SET-4.0.nemo"

asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    restore_path=asr_model_name
).to(torch.device("cpu"))

tokenizer = asr_model.tokenizer

words=sys.argv[1]
lexicon=sys.argv[2]

with open(lexicon,'w') as fp:
    for word in open(words,'r'):
        if word.strip()[0]!='#':
            entry_lex=" ".join(tokenizer.text_to_tokens(word.strip()))
            fp.write(f"{word.strip()} {entry_lex}\n")
        else:
            print(f"Skipping class label {word}")
