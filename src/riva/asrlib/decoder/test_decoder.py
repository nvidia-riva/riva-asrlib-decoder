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
import itertools
import os
import pathlib
import unittest

import kaldi_io
import more_itertools
import torch
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedDecoderCudaConfig


class DecoderTest(unittest.TestCase):
    def test_decoder(self):
        test_data_dir = pathlib.Path(__file__).parent.resolve() / "test_data"

        logits_ark = str("ark:" / test_data_dir / "logits.ark")

        _, matrix = next(kaldi_io.read_mat_ark(logits_ark))
        num_tokens_including_blank = matrix.shape[1]

        # TODO: What to do about minimize option?
        config = BatchedMappedDecoderCudaConfig()
        config.n_input_per_chunk = 50
        config.online_opts.lattice_postprocessor_opts.word_boundary_rxfilename = str(
            test_data_dir / "word_boundary.int"
        )
        config.online_opts.decoder_opts.default_beam = 17.0
        config.online_opts.decoder_opts.lattice_beam = 8.0
        config.online_opts.decoder_opts.max_active = 7000
        config.online_opts.determinize_lattice = True
        config.online_opts.max_batch_size = 400
        config.online_opts.num_channels = 800
        config.online_opts.frame_shift_seconds = 0.03
        decoder = BatchedMappedDecoderCuda(
            config, str(test_data_dir / "TLG.fst"), str(test_data_dir / "words.txt"), num_tokens_including_blank
        )

        for batch in more_itertools.chunked(kaldi_io.read_mat_ark(logits_ark), config.online_opts.max_batch_size):
            sequences = []
            sequence_lengths = []
            for key, matrix in batch:
                sequences.append(torch.from_numpy(matrix.copy()))
                sequence_lengths.append(matrix.shape[0])
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).cuda()
            sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
            for result in decoder.decode(padded_sequence, sequence_lengths_tensor):
                print(result)
