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
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import unittest
import zipfile

import more_itertools
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
import torch
from tqdm import tqdm

import riva.asrlib.decoder
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedDecoderCudaConfig

class GraphConstructionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.abspath("tmp_graph_construction")
        os.makedirs(cls.temp_dir, exist_ok=True)

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
        config_yaml = os.path.join(cls.temp_dir, "model_config.yaml")

        yaml = YAML(typ='safe')
        with tarfile.open(cls.nemo_model_path, "r:gz") as tar_fh:
            with tar_fh.extractfile("./model_config.yaml") as fh:
                data = yaml.load(fh)
        cls.units_txt = os.path.join(cls.temp_dir, "units.txt")
        with open(cls.units_txt, "w") as fh:
            for unit in data["decoder"]["vocabulary"]:
                fh.write(f"{unit}\n")

        cls.num_tokens_including_blank = len(data["decoder"]["vocabulary"]) + 1
        assert cls.num_tokens_including_blank == 1025

    @classmethod
    def tearDownClass(cls):
        pass
        # shutil.rmtree(cls.temp_dir)

    def test_eesen_ctc_topo(self):
        self.create_TLG("ctc_eesen", os.path.join(self.temp_dir, "ctc_eesen"))

    def test_vanilla_ctc_topo(self):
        self.create_TLG("ctc", os.path.join(self.temp_dir, "ctc"))
        self.run_decoder(os.path.join(self.temp_dir, "ctc"))

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
                "--units",
                self.units_txt,
                "--topo",
                topo,
                lexicon_path,
                self.gzipped_lm_path,
                dest_dir,
            ]
        )

    def run_decoder(self, graph_path: str):
        asr_model = nemo_asr.models.ASRModel.restore_from(self.nemo_model_path, map_location=torch.device("cuda"))

        config = BatchedMappedDecoderCudaConfig()
        config.n_input_per_chunk = 50
        config.online_opts.decoder_opts.default_beam = 17.0
        config.online_opts.decoder_opts.lattice_beam = 8.0
        config.online_opts.decoder_opts.max_active = 7000
        config.online_opts.determinize_lattice = True
        config.online_opts.max_batch_size = 400
        config.online_opts.num_channels = 800
        config.online_opts.frame_shift_seconds = 0.03
        decoder = BatchedMappedDecoderCuda(
            config,
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            self.num_tokens_including_blank
        )

        manifest = "/mnt/disks/sda_hdd/librispeech/dev_clean.json"
        paths = []
        with open(manifest) as fh:
            for line in fh:
                entry = json.loads(line)
                paths.append(entry["audio_filepath"])

        for path in paths:
            logprobs = asr_model.transcribe([path], batch_size=1, logprobs=True)
            sequences = [torch.from_numpy(logprobs[0]).cuda()]
            sequence_lengths = [logprobs[0].shape[0]]
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
            sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
            for result in decoder.decode(padded_sequence,
                                         sequence_lengths_tensor):
                print(result)
