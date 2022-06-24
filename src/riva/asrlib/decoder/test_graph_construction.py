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

import gzip
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile

import more_itertools
import nemo.collections.asr as nemo_asr
import riva.asrlib.decoder
import torch
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedDecoderCudaConfig
from tqdm import tqdm


class GraphConstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs("tmp", exist_ok=True)
        cls.temp_dir = os.path.abspath("tmp")

        lm_zip_file = os.path.join(cls.temp_dir, "speechtotext_english_lm_deployable_v1.0.zip")
        if not os.path.exists(lm_zip_file):
            subprocess.check_call(
                f"wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/speechtotext_english_lm/versions/deployable_v1.0/zip -O {lm_zip_file}",
                shell=True,
            )
            with zipfile.ZipFile(lm_zip_file, 'r') as zip_ref:
                zip_ref.extractall(cls.temp_dir)

        am_zip_file = os.path.join(cls.temp_dir, "stt_en_conformer_ctc_small_1.6.0.zip")
        if not os.path.exists(am_zip_file):
            subprocess.check_call(
                f"wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.6.0/zip -O {am_zip_file}",
                shell=True,
            )
            with zipfile.ZipFile(am_zip_file, 'r') as zip_ref:
                zip_ref.extractall(cls.temp_dir)

        # Work around: At the time of writing this test, the words.txt
        # file downloaded from NGC is simply a git lfs stub file, not
        # the actual file itself, so overwrite cls.words_path by
        # exracting the symbol table from the arpa file
        lm_path = os.path.join(cls.temp_dir, "3-gram.pruned.3e-7.arpa")
        cls.words_path = os.path.join(cls.temp_dir, "words.mixed_lm.3-gram.pruned.3e-7.txt")
        temp_words_path = os.path.join(cls.temp_dir, "words_with_ids.txt")
        subprocess.check_call(
            [
                os.path.join(riva.asrlib.decoder.__path__[0], "scripts/prepare_TLG_fst/bin/arpa2fst"),
                f"--write-symbol-table={temp_words_path}",
                lm_path,
                "/dev/null",
            ]
        )
        cls.gzipped_lm_path = lm_path + ".gz"
        with open(lm_path, 'rb') as f_in, gzip.open(cls.gzipped_lm_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        with open(temp_words_path, "r") as words_with_ids_fh, open(cls.words_path, "w") as words_fh:
            for word_with_id in words_with_ids_fh:
                word = word_with_id.split()[0].lower()
                if word in {"<eps>", "<s>", "</s>", "<unk>"}:
                    continue
                words_fh.write(word)
                words_fh.write("\n")

        cls.nemo_model_path = os.path.join(cls.temp_dir, "stt_en_conformer_ctc_small.nemo")

    @classmethod
    def tearDownClass(cls):
        pass
        # shutil.rmtree(cls.temp_dir)

    def test_eesen_ctc_topo(self):
        self.create_TLG("ctc_eesen", os.path.join(self.temp_dir, "ctc_eesen"))

    def test_vanilla_ctc_topo(self):
        self.create_TLG("ctc", os.path.join(self.temp_dir, "ctc"))

    def test_compact_ctc_topo(self):
        self.create_TLG("ctc_compact", os.path.join(self.temp_dir, "ctc_compact"))

    def test_identity_ctc_topo(self):
        self.create_TLG("identity", os.path.join(self.temp_dir, "identity"))

    def create_TLG(self, topo, work_dir):
        (path,) = riva.asrlib.decoder.__path__
        prep_subw_lexicon = os.path.join(path, "scripts/prepare_TLG_fst/prep_subw_lexicon.sh")
        lexicon_path = os.path.join(work_dir, "lexicon.txt")
        subprocess.check_call(
            [
                prep_subw_lexicon,
                "--target",
                "words",
                "--model_path",
                self.nemo_model_path,
                "--vocab",
                self.words_path,
                lexicon_path,
            ]
        )
        mkgraph_ctc = os.path.join(path, "scripts/prepare_TLG_fst/mkgraph_ctc.sh")
        dest_dir = os.path.join(work_dir, "graph")
        subprocess.check_call(
            [
                mkgraph_ctc,
                "--stage",
                "1",
                "--lm_lowercase",
                "true",
                "--topo",
                topo,
                lexicon_path,
                self.gzipped_lm_path,
                dest_dir,
            ]
        )

    def run_decoder(self):
        asr_model = nemo_asr.models.ASRModel.restore_from(self.nemo_model_path, map_location=torch.device("cuda"))
        test_data_dir = pathlib.Path(__file__).parent.resolve() / "test_data"

        logits_ark = str("ark:" / test_data_dir / "logits.ark")

        _, matrix = next(kaldi_io.read_mat_ark(logits_ark))
        num_tokens_including_blank = matrix.shape[1]

        # TODO: What to do about minimize option?
        config = BatchedMappedDecoderCudaConfig()
        config.n_input_per_chunk = 50
        # config.online_opts.lattice_postprocessor_opts.word_boundary_rxfilename = str(
        #     test_data_dir / "word_boundary.int"
        # )
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
            all_logits = asr_model.transcribe(audio_file_paths, batch_size=args.acoustic_batch_size, logprobs=True)

            sequences = []
            sequence_lengths = []
            for key, matrix in batch:
                sequences.append(torch.from_numpy(matrix.copy()))
                sequence_lengths.append(matrix.shape[0])
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True).cuda()
            sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
            for result in decoder.decode(padded_sequence, sequence_lengths_tensor):
                print(result)


def get_logits(asr_model, paths2audio_files, batch_size):
    device = next(asr_model.parameters()).device
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.eval()
    asr_model.encoder.freeze()
    asr_model.decoder.freeze()

    if num_workers is None:
        num_workers = os.cpu_count()

    # Work in tmp directory - will store manifest file there
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in paths2audio_files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                fp.write(json.dumps(entry) + '\n')

            config = {
                'paths2audio_files': paths2audio_files,
                'batch_size': batch_size,
                'temp_dir': tmpdir,
                'num_workers': num_workers,
            }

        temporary_datalayer = self._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
            logits, logits_len, greedy_predictions = self.forward(
                input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
            )
            # dump log probs per file
            for idx in range(logits.shape[0]):
                lg = logits[idx][: logits_len[idx]]
                hypotheses.append(lg.cpu().numpy())
            hypotheses += current_hypotheses

            del greedy_predictions
            del logits
            del test_batch
