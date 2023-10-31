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

import glob
import gzip
import itertools
import json
import logging
import multiprocessing
import os
import pathlib
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest
import urllib.request
import zipfile
from contextlib import ExitStack
from enum import Enum

import more_itertools
import nemo.collections.asr as nemo_asr
import numpy as np
import pytest
import torch
import torchaudio
import torchmetrics
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.ctc_beam_decoding import BeamCTCInfer, FlashlightConfig
from ruamel.yaml import YAML
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

import riva.asrlib.decoder
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedOnlineDecoderCuda, BatchedMappedDecoderCudaConfig

from nemo.collections.asr.metrics.wer import word_error_rate

ERROR_MARGIN = 0.0003

# os.environ["TORCH_CUDNN_V8_API_ENABLED"]="1"


# importing this causes pytest never to finish "collecting". No idea why.
# import pywrapfst


logging.getLogger("nemo").setLevel(logging.INFO)


# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

class DecodeType(Enum):
    MBR = 1
    NBEST = 2


# Not as good as it could be. Doesn't support num_workers > 0. And
# that's hard to support because of multiprocessing, unfortunately.
class CacheDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __getitem__(self, n: int):
        if n not in self.cache:
            result = self.dataset[n]
            self.cache[n] = result
        return self.cache[n]

    def __len__(self) -> int:
        return len(self.dataset)


def load_word_symbols(path):
    word_id_to_word_str = {}
    with open(path, "rt") as fh:
        for line in fh:
            word_str, word_id = line.rstrip().split()
            word_id_to_word_str[int(word_id)] = word_str
    return word_id_to_word_str

class TestGraphConstruction:
    temp_dir = None
    lm_path = None
    words_path = None
    gzipped_lm_path = None
    original_dataset_map = None
    dataset_map = None

    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        cls = self.__class__
        cls.temp_dir = os.path.abspath("tmp_graph_construction")
        os.makedirs(cls.temp_dir, exist_ok=True)

        lm_zip_file = os.path.join(cls.temp_dir, "speechtotext_english_lm_deployable_v1.0.zip")
        if not os.path.exists(lm_zip_file):
            urllib.request.urlretrieve("https://api.ngc.nvidia.com/v2/models/nvidia/tao/speechtotext_english_lm/versions/deployable_v1.0/zip", lm_zip_file)
            with zipfile.ZipFile(lm_zip_file, 'r') as zip_ref:
                zip_ref.extractall(cls.temp_dir)

        # Work around: At the time of writing this test, the words.txt
        # file downloaded from NGC is simply a git lfs stub file, not
        # the actual file itself, so overwrite cls.words_path by
        # exracting the symbol table from the arpa file
        cls.lm_path = os.path.join(cls.temp_dir, "3-gram.pruned.3e-7.arpa")
        cls.words_path = os.path.join(cls.temp_dir, "words.mixed_lm.3-gram.pruned.3e-7.txt")
        temp_words_path = os.path.join(cls.temp_dir, "words_with_ids.txt")
        if not os.path.exists(temp_words_path):
            subprocess.check_call(
                [
                    os.path.join(riva.asrlib.decoder.__path__[0], "scripts/prepare_TLG_fst/bin/arpa2fst"),
                    f"--write-symbol-table={temp_words_path}",
                    cls.lm_path,
                    "/dev/null",
                ]
            )
        cls.gzipped_lm_path = cls.lm_path + ".gz"
        with open(cls.lm_path, 'rb') as f_in, gzip.open(cls.gzipped_lm_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        with open(temp_words_path, "r") as words_with_ids_fh, open(cls.words_path, "w") as words_fh:
            for word_with_id in words_with_ids_fh:
                word = word_with_id.split()[0].lower()
                if word in {"<eps>", "<s>", "</s>", "<unk>"}:
                    continue
                words_fh.write(word)
                words_fh.write("\n")

        librispeech_test_clean = torchaudio.datasets.LIBRISPEECH(cls.temp_dir, "test-clean", download=True)
        librispeech_test_other = torchaudio.datasets.LIBRISPEECH(cls.temp_dir, "test-other", download=True)
        librispeech_dev_clean = torchaudio.datasets.LIBRISPEECH(cls.temp_dir, "dev-clean", download=True)
        librispeech_dev_other = torchaudio.datasets.LIBRISPEECH(cls.temp_dir, "dev-other", download=True)

        cls.original_dataset_map = {
            "test-clean": librispeech_test_clean,
            "test-other": librispeech_test_other,
            "dev-clean": librispeech_dev_clean,
            "dev-other": librispeech_dev_other,
        }
        cls.dataset_map = {}

        for key, dataset in cls.original_dataset_map.items():
            dataset = CacheDataset(dataset)
            lengths = []
            for i in range(len(dataset)):
                waveform, *_ = dataset[i]
                lengths.append(waveform.size(1))
            sorted_indices = list(np.argsort(lengths))
            cls.dataset_map[key] = Subset(dataset, sorted_indices)

    def test_eesen_ctc_topo(self):
        self.create_TLG("ctc_eesen", os.path.join(self.temp_dir, "ctc_eesen"), "stt_en_conformer_ctc_small")

    @pytest.mark.parametrize(
        "nemo_model_name, dataset, expected_wer, half_precision",
        [
            ("stt_en_conformer_ctc_small", "test-clean", 0.03581482045039562, False),
            ("stt_en_conformer_ctc_small", "test-clean", 0.03560559951308582, True),
            ("stt_en_conformer_ctc_small", "test-other", 0.07055384674168466, False),
            ("stt_en_conformer_ctc_small", "test-other", 0.07116519878493781, True),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.034392117936840556, False),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.034355354582552115, True),
            ("stt_en_conformer_ctc_small", "dev-other", 0.06969851613409751, False),
            ("stt_en_conformer_ctc_small", "dev-other", 0.06987516683677475, True),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.03185864272671941, False),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.03201080340839927, True),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05708499703876354, False),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05737156830903846, True),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.027811477519208854, False),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.0280136759677953, True),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.0547224621182382, False),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.05462432283897307, True),
            ("stt_en_conformer_ctc_large", "test-clean", 0.027255782105903834, False),
            ("stt_en_conformer_ctc_large", "test-clean", 0.027369902617163724, True),
            ("stt_en_conformer_ctc_large", "test-other", 0.04529736545478861, False),
            ("stt_en_conformer_ctc_large", "test-other", 0.04531647020614027, True),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.02499908091614279, False),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.02485202749898901, True),
            ("stt_en_conformer_ctc_large", "dev-other", 0.0434364450027479, False),
            ("stt_en_conformer_ctc_large", "dev-other", 0.0431420271649525, True),
        ],
    )
    def test_vanilla_ctc_topo_wer_mbr(self, nemo_model_name, dataset, expected_wer, half_precision):
        work_dir = os.path.join(self.temp_dir, f"ctc_{nemo_model_name}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc", work_dir, nemo_model_name)

        acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale, nbest = (
            1.0 / 0.55,
            1024,
            0.0,
            -0.5,
            0.8,
            1,
        )
        my_wer, _, _ = self.run_decoder(
            asr_model,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7"),
            self.dataset_map[dataset],
            half_precision,
            acoustic_scale,
            blank_penalty,
            blank_ilabel,
            length_penalty,
            lm_scale,
            nbest,
            DecodeType.MBR,
        )
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(f"GALVEZ:model={nemo_model_name} dataset={dataset} vanilla half_precision={half_precision} wer={my_wer}")
        # Accept a very tiny margin of error
        assert my_wer <= expected_wer + ERROR_MARGIN


    def test_batch_size_1(self):
        """
        Integration test for https://github.com/wjakob/nanobind/pull/162
        """
        work_dir = os.path.join(self.temp_dir, "ctc")
        nemo_model_name = "stt_en_conformer_ctc_small"

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_name, map_location=torch.device("cuda")
        )

        self.create_TLG("ctc", work_dir, nemo_model_name)

        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        decoder_config = self.create_decoder_config()
        decoder = BatchedMappedDecoderCuda(
            decoder_config,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            num_tokens_including_blank,
        )

        logits = torch.ones((1,100, num_tokens_including_blank), dtype=torch.float32).cuda()
        lengths = torch.tensor([100], dtype=torch.int64).cpu()
        decoder.decode_mbr(logits.detach(), lengths.detach())

    # Note that nbest decoding tends to produce a worse WER than mbr
    # decoding. This is expected.
    @pytest.mark.parametrize(
        "nemo_model_name, dataset, expected_wer, half_precision",
        [
            ("stt_en_conformer_ctc_small", "test-clean", 0.03581482045039562, False),
            ("stt_en_conformer_ctc_small", "test-clean", 0.03560559951308582, True),
            ("stt_en_conformer_ctc_small", "test-other", 0.07074489425520127, False),
            ("stt_en_conformer_ctc_small", "test-other", 0.07168102707143266, True),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.034392117936840556, False),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.03468622477114812, True),
            ("stt_en_conformer_ctc_small", "dev-other", 0.07054251393577765, False),
            ("stt_en_conformer_ctc_small", "dev-other", 0.07050325822407161, True),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.03185864272671941, False),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.03201080340839927, True),
            ("stt_en_conformer_ctc_medium", "test-other", 0.0573524635576868, False),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05763903482796171, True),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.0280136759677953, False),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.0280136759677953, True),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.05509539137944571, False),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.055370181361388084, True),
            ("stt_en_conformer_ctc_large", "test-clean", 0.027255782105903834, False),
            ("stt_en_conformer_ctc_large", "test-clean", 0.027369902617163724, True),
            ("stt_en_conformer_ctc_large", "test-other", 0.04529736545478861, False),
            ("stt_en_conformer_ctc_large", "test-other", 0.04531647020614027, True),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.02499908091614279, False),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.02485202749898901, True),
            ("stt_en_conformer_ctc_large", "dev-other", 0.0437308628405433, False),
            ("stt_en_conformer_ctc_large", "dev-other", 0.04367197927298422, True),
        ],
    )
    def test_vanilla_ctc_topo_wer_nbest(self, nemo_model_name, dataset, expected_wer, half_precision):
        work_dir = os.path.join(self.temp_dir, f"ctc_{nemo_model_name}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc", work_dir, nemo_model_name)

        acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale, nbest = (
            1.0 / 0.55,
            1024,
            0.0,
            -0.5,
            0.8,
            1,
        )
        my_wer, _, _ = self.run_decoder(
            asr_model,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7"),
            self.dataset_map[dataset],
            half_precision,
            acoustic_scale,
            blank_penalty,
            blank_ilabel,
            length_penalty,
            lm_scale,
            nbest,
            DecodeType.NBEST,
        )
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(f"GALVEZ:model={nemo_model_name} dataset={dataset} vanilla half_precision={half_precision} wer={my_wer}")
        # Accept a very tiny margin of error
        assert my_wer <= expected_wer + ERROR_MARGIN


    @pytest.mark.ci
    def test_vanilla_ctc_ci(self):
        self.test_vanilla_ctc_topo_wer_nbest(
            "stt_en_conformer_ctc_small", "test-clean",
            0.03560559951308582, False)
        self.test_vanilla_ctc_topo_wer_nbest(
            "stt_en_conformer_ctc_small", "test-clean",
            0.03581482045039562, True)
        self.test_vanilla_ctc_topo_wer_mbr(
            "stt_en_conformer_ctc_small", "test-clean",
            0.03581482045039562, False)
        self.test_vanilla_ctc_topo_wer_mbr(
            "stt_en_conformer_ctc_small", "test-clean",
            0.03560559951308582, True)

    @pytest.mark.ci
    def test_compact_ctc_ci(self):
        self.test_compact_ctc_topo_wer(
            "stt_en_conformer_ctc_small", "test-clean",
            0.03164942178940962, False)
        self.test_compact_ctc_topo_wer(
            "stt_en_conformer_ctc_small", "test-clean",
            0.03174452221545952, True)

    @pytest.mark.parametrize(
        "nemo_model_name, dataset, half_precision",
        [
            ("stt_en_conformer_ctc_small", "test-clean", False),
            ("stt_en_conformer_ctc_small", "test-clean", True),
            ("stt_en_conformer_ctc_small", "test-other", False),
            ("stt_en_conformer_ctc_small", "test-other", True),
            ("stt_en_conformer_ctc_small", "dev-clean", False),
            ("stt_en_conformer_ctc_small", "dev-clean", True),
            ("stt_en_conformer_ctc_small", "dev-other", False),
            ("stt_en_conformer_ctc_small", "dev-other", True),
            ("stt_en_conformer_ctc_medium", "test-clean", False),
            ("stt_en_conformer_ctc_medium", "test-clean", True),
            ("stt_en_conformer_ctc_medium", "test-other", False),
            ("stt_en_conformer_ctc_medium", "test-other", True),
            ("stt_en_conformer_ctc_medium", "dev-clean", False),
            ("stt_en_conformer_ctc_medium", "dev-clean", True),
            ("stt_en_conformer_ctc_medium", "dev-other", False),
            ("stt_en_conformer_ctc_medium", "dev-other", True),
            ("stt_en_conformer_ctc_large", "test-clean", False),
            ("stt_en_conformer_ctc_large", "test-clean", True),
            ("stt_en_conformer_ctc_large", "test-other", False),
            ("stt_en_conformer_ctc_large", "test-other", True),
            ("stt_en_conformer_ctc_large", "dev-clean", False),
            ("stt_en_conformer_ctc_large", "dev-clean", True),
            ("stt_en_conformer_ctc_large", "dev-other", False),
            ("stt_en_conformer_ctc_large", "dev-other", True),
        ],
    )
    def test_vanilla_ctc_topo_throughput(self, nemo_model_name, dataset, half_precision):
        work_dir = os.path.join(self.temp_dir, f"ctc_{nemo_model_name}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc", work_dir, nemo_model_name)

        acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale, nbest = (
            1.0 / 0.55,
            1024,
            0.0,
            -0.5,
            0.8,
            1,
        )
        warmup_iters = 2
        benchmark_iters = 2
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(f"GALVEZ:model={nemo_model_name} dataset={dataset} vanilla half_precision={half_precision}")
        self.run_decoder_throughput(
            asr_model,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7"),
            self.dataset_map[dataset],
            half_precision,
            acoustic_scale,
            blank_penalty,
            blank_ilabel,
            length_penalty,
            lm_scale,
            nbest,
            warmup_iters,
            benchmark_iters,
            DecodeType.MBR,
        )

    @pytest.mark.parametrize(
        "nemo_model_name, dataset, expected_wer, half_precision",
        [
            ("stt_en_conformer_ctc_small", "test-clean", 0.03164942178940962, False),
            ("stt_en_conformer_ctc_small", "test-clean", 0.03174452221545952, True),
            ("stt_en_conformer_ctc_small", "test-other", 0.06761171503352884, False),
            ("stt_en_conformer_ctc_small", "test-other", 0.0683185908335403, True),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.029631263556486893, False),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.02968640858791956, True),
            ("stt_en_conformer_ctc_small", "dev-other", 0.06661694276517233, False),
            ("stt_en_conformer_ctc_small", "dev-other", 0.0668132213237026, True),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.026361838101034693, False),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.02655203895313451, True),
            ("stt_en_conformer_ctc_medium", "test-other", 0.052996580249508055, False),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05324494201707965, True),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.02249917282452851, False),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.022682989595970735, True),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.05034545026301327, False),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.05032582240716024, True),
            ("stt_en_conformer_ctc_large", "test-clean", 0.022120359099208765, False),
            ("stt_en_conformer_ctc_large", "test-clean", 0.022120359099208765, True),
            ("stt_en_conformer_ctc_large", "test-other", 0.040998796400664846, False),
            ("stt_en_conformer_ctc_large", "test-other", 0.041094320157423155, True),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.01874931068710709, False),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.01865740230138598, True),
            ("stt_en_conformer_ctc_large", "dev-other", 0.038980921724110856, False),
            ("stt_en_conformer_ctc_large", "dev-other", 0.038980921724110856, True),
        ],
    )
    def test_compact_ctc_topo_wer(self, nemo_model_name, dataset, expected_wer, half_precision):
        work_dir = os.path.join(self.temp_dir, f"ctc_compact_{nemo_model_name}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc_compact", work_dir, nemo_model_name)

        acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale, nbest = (
            1.0 / 0.55,
            1024,
            0.0,
            -0.5,
            0.8,
            1,
        )
        my_wer, _, _ = self.run_decoder(
            asr_model,
            os.path.join(work_dir, "graph/graph_ctc_compact_3-gram.pruned.3e-7"),
            self.dataset_map[dataset],
            half_precision,
            acoustic_scale,
            blank_penalty,
            blank_ilabel,
            length_penalty,
            lm_scale,
            nbest,
            DecodeType.MBR,
        )
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(f"GALVEZ:model={nemo_model_name} dataset={dataset} compact half_precision={half_precision} wer={my_wer}")
        # Accept a very tiny margin of error
        assert my_wer <= expected_wer + ERROR_MARGIN

    @pytest.mark.parametrize(
        "nemo_model_name, dataset, half_precision",
        [
            ("stt_en_conformer_ctc_small", "test-clean", False),
            ("stt_en_conformer_ctc_small", "test-clean", True),
            ("stt_en_conformer_ctc_small", "test-other", False),
            ("stt_en_conformer_ctc_small", "test-other", True),
            ("stt_en_conformer_ctc_small", "dev-clean", False),
            ("stt_en_conformer_ctc_small", "dev-clean", True),
            ("stt_en_conformer_ctc_small", "dev-other", False),
            ("stt_en_conformer_ctc_small", "dev-other", True),
            ("stt_en_conformer_ctc_medium", "test-clean", False),
            ("stt_en_conformer_ctc_medium", "test-clean", True),
            ("stt_en_conformer_ctc_medium", "test-other", False),
            ("stt_en_conformer_ctc_medium", "test-other", True),
            ("stt_en_conformer_ctc_medium", "dev-clean", False),
            ("stt_en_conformer_ctc_medium", "dev-clean", True),
            ("stt_en_conformer_ctc_medium", "dev-other", False),
            ("stt_en_conformer_ctc_medium", "dev-other", True),
            ("stt_en_conformer_ctc_large", "test-clean", False),
            ("stt_en_conformer_ctc_large", "test-clean", True),
            ("stt_en_conformer_ctc_large", "test-other", False),
            ("stt_en_conformer_ctc_large", "test-other", True),
            ("stt_en_conformer_ctc_large", "dev-clean", False),
            ("stt_en_conformer_ctc_large", "dev-clean", True),
            ("stt_en_conformer_ctc_large", "dev-other", False),
            ("stt_en_conformer_ctc_large", "dev-other", True),
        ],
    )
    def test_compact_ctc_topo_throughput(self, nemo_model_name, dataset, half_precision):
        work_dir = os.path.join(self.temp_dir, f"ctc_compact_{nemo_model_name}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc_compact", work_dir, nemo_model_name)

        acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale, nbest = (
            1.0 / 0.55,
            1024,
            0.0,
            -0.5,
            0.8,
            1,
        )
        warmup_iters = 2
        benchmark_iters = 1
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(f"GALVEZ:model={nemo_model_name} dataset={dataset} compact half_precision={half_precision}")
        self.run_decoder_throughput(
            asr_model,
            os.path.join(work_dir, "graph/graph_ctc_compact_3-gram.pruned.3e-7"),
            self.dataset_map[dataset],
            half_precision,
            acoustic_scale,
            blank_penalty,
            blank_ilabel,
            length_penalty,
            lm_scale,
            nbest,
            warmup_iters,
            benchmark_iters,
            DecodeType.MBR,
        )

    @pytest.mark.parametrize(
        "nemo_model_name, dataset, expected_wer, half_precision",
        [
            ("stt_en_conformer_ctc_small", "test-clean", 0.03305690809494827, False),
            ("stt_en_conformer_ctc_small", "test-clean", 0.03315200852099817, True),
            ("stt_en_conformer_ctc_small", "test-other", 0.06854784784976023, False),
            ("stt_en_conformer_ctc_small", "test-other", 0.06936935215788166, True),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.03064225579941914, False),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.03086283592514981, True),
            ("stt_en_conformer_ctc_small", "dev-other", 0.06746094056685248, False),
            ("stt_en_conformer_ctc_small", "dev-other", 0.06785349768391301, True),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.028663268411442483, False),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.02851110772976263, True),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05446764610358596, False),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05452496035764094, True),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.02391456196463365, False),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.024484393956104553, True),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.051640888749313024, False),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.051346470911517623, True),
            ("stt_en_conformer_ctc_large", "test-clean", 0.023527845404747415, False),
            ("stt_en_conformer_ctc_large", "test-clean", 0.023508825319537432, True),
            ("stt_en_conformer_ctc_large", "test-other", 0.04191582446554458, False),
            ("stt_en_conformer_ctc_large", "test-other", 0.04214508148176452, True),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.020109554795779565, False),
            ("stt_en_conformer_ctc_large", "dev-clean", 0.020017646410058453, True),
            ("stt_en_conformer_ctc_large", "dev-other", 0.04057077804820602, False),
            ("stt_en_conformer_ctc_large", "dev-other", 0.04029598806626364, True),
        ],
    )
    def test_identity_ctc_topo(self, nemo_model_name, dataset, expected_wer, half_precision):
        work_dir = os.path.join(self.temp_dir, f"ctc_identity_{nemo_model_name}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("identity", work_dir, nemo_model_name)

        acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale, nbest = (
            1.0 / 0.55,
            1024,
            0.0,
            -0.5,
            0.8,
            1,
        )
        my_wer, _, _ = self.run_decoder(
            asr_model,
            os.path.join(work_dir, "graph/graph_identity_3-gram.pruned.3e-7"),
            self.dataset_map[dataset],
            half_precision,
            acoustic_scale,
            blank_penalty,
            blank_ilabel,
            length_penalty,
            lm_scale,
            nbest,
            DecodeType.MBR,
        )
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(
            f"GALVEZ:model={nemo_model_name} dataset={dataset} identity half_precision={half_precision} wer={my_wer}"
        )
        # Accept a very tiny margin of error
        assert my_wer <= expected_wer + ERROR_MARGIN

    # Skip flashlight tests. They use hard-coded paths for flashlight
    # because the code to create a kenlm language model is not
    # directly part of the NeMo library.
    # @pytest.mark.skip
    @pytest.mark.parametrize("nemo_model_name, dataset_name",
                             [
                                 ("stt_en_conformer_ctc_small","test-clean"),
                                 ("stt_en_conformer_ctc_small","test-other"),
                                 ("stt_en_conformer_ctc_small","dev-clean"),
                                 ("stt_en_conformer_ctc_small","dev-other"),
                                 ("stt_en_conformer_ctc_medium","test-clean"),
                                 ("stt_en_conformer_ctc_medium","test-other"),
                                 ("stt_en_conformer_ctc_medium","dev-clean"),
                                 ("stt_en_conformer_ctc_medium","dev-other"),
                                 ("stt_en_conformer_ctc_large","test-clean"),
                                 ("stt_en_conformer_ctc_large","test-other"),
                                 ("stt_en_conformer_ctc_large","dev-clean"),
                                 ("stt_en_conformer_ctc_large","dev-other"),
                             ]
    )
    def test_flashlight_alone(self, nemo_model_name, dataset_name):
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_name, map_location=torch.device("cuda")
        )

        _ = self.run_decoder_flashlight2(asr_model,
                                         CacheDataset(torchaudio.datasets.LIBRISPEECH(self.temp_dir, dataset_name, download=True)),
                                         True,
                                         1024,
                                         nemo_model_name,
                                         dataset_name,
        )

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "dataset_name, nbest",
        [
            ("test-clean", 10),
            ("test-clean", 1),
            ("test-other", 10),
            ("test-other", 1),
        ],
    )
    def test_flashlight_vs_wfst(self, dataset_name, nbest):
        work_dir = os.path.join(self.temp_dir, "ctc")
        flashlight_predictions, references, durations, file_paths = self.run_decoder_flashlight(dataset_name)
        # No caching or sorting by sequence length allowed here
        dataset = CacheDataset(torchaudio.datasets.LIBRISPEECH(self.temp_dir, dataset_name, download=True))
        _, wfst_predictions, wfst_references = self.run_decoder(
            asr_model,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7"),
            dataset,
            False,
            1.0 / 0.55,
            0.0,
            1024,
            -0.5,
            0.8,
            nbest,
            DecodeType.MBR,
        )

        with open(f"wfst_{dataset_name}_nbest_{nbest}.txt", "w") as fh:
            for utt_id, wfst_prediction in wfst_predictions:
                fh.write(f"{utt_id} {wfst_prediction}\n")

        with open(f"ref_{dataset_name}.txt", "w") as fh:
            for utt_id, reference in wfst_references:
                fh.write(f"{utt_id} {reference}\n")

        flashlight_predictions, references, durations, file_paths = self.run_decoder_flashlight(dataset_name)

    def test_delete_decoder(self):
        """Ensure that allocating a decoder, decoding with it, deleting it,
        and then reallocating a new one, and deocding with that one,
        does not crash.
        """
        work_dir = os.path.join(self.temp_dir, "ctc")
        nemo_model_name = "stt_en_conformer_ctc_small"

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_name, map_location=torch.device("cuda")
        )

        self.create_TLG("ctc", work_dir, nemo_model_name)

        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        torch.cuda.cudart().cudaProfilerStart()

        decoder1_config = self.create_decoder_config()
        decoder2_config = self.create_decoder_config()

        data_loader = torch.utils.data.DataLoader(
            self.dataset_map["test-clean"],
            batch_size=decoder1_config.online_opts.max_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            for batch in data_loader:
                input_signal, input_signal_length, target, utterance_ids = batch
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                log_probs, lengths, _ = asr_model.forward(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )
                cpu_lengths = lengths.to(torch.int64).to('cpu')

                decoder1 = BatchedMappedDecoderCuda(
                    decoder1_config,
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
                    num_tokens_including_blank,
                )

                decoder1.decode_mbr(log_probs.to(torch.float32), cpu_lengths)

                del decoder1

                decoder2 = BatchedMappedDecoderCuda(
                    decoder2_config,
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
                    num_tokens_including_blank,
                )
                decoder2.decode_mbr(log_probs.to(torch.float32), cpu_lengths)
                break

        torch.cuda.cudart().cudaProfilerStop()

    def create_TLG(self, topo, work_dir, nemo_model_name):
        # If TLG was already created, skip this process.
        if len(glob.glob(os.path.join(work_dir, "graph/*/TLG.fst"))) != 0:
            return
        os.makedirs(work_dir, exist_ok=True)
        asr_model_config = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, return_config=True)
        units_txt = os.path.join(work_dir, "units.txt")
        with open(os.path.join(work_dir, "units.txt"), "w") as fh:
            for unit in asr_model_config["decoder"]["vocabulary"]:
                fh.write(f"{unit}\n")

        (path,) = riva.asrlib.decoder.__path__
        prep_subw_lexicon = os.path.join(path, "scripts/prepare_TLG_fst/prep_subw_lexicon.sh")
        lexicon_path = os.path.join(work_dir, "lexicon.txt")
        subprocess.check_call(
            [
                prep_subw_lexicon,
                "--target",
                "words",
                "--model_path",
                nemo_model_name,
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
                units_txt,
                "--topo",
                topo,
                lexicon_path,
                self.gzipped_lm_path,
                dest_dir,
            ]
        )

    @staticmethod
    def create_decoder_config():
        config = BatchedMappedDecoderCudaConfig()
        config.n_input_per_chunk = 50
        config.online_opts.decoder_opts.default_beam = 17.0
        config.online_opts.decoder_opts.lattice_beam = 8.0
        config.online_opts.decoder_opts.max_active = 10_000
        config.online_opts.decoder_opts.ntokens_pre_allocated = 10_000_000
        config.online_opts.determinize_lattice = True
        config.online_opts.max_batch_size = 200
        config.online_opts.num_channels = config.online_opts.max_batch_size * 2
        config.online_opts.frame_shift_seconds = 0.04
        config.online_opts.lattice_postprocessor_opts.acoustic_scale = 1.0
        config.online_opts.lattice_postprocessor_opts.lm_scale = 1.0
        config.online_opts.lattice_postprocessor_opts.word_ins_penalty = 0.0
        config.online_opts.lattice_postprocessor_opts.nbest = 1
        config.online_opts.num_decoder_copy_threads = 2
        config.online_opts.num_post_processing_worker_threads = (
            multiprocessing.cpu_count() - config.online_opts.num_decoder_copy_threads
        )

        return config

    def run_decoder(
        self,
        asr_model,
        graph_path: str,
        dataset: torch.utils.data.IterableDataset,
        half_precision: bool,
        acoustic_scale,
        blank_penalty,
        blank_ilabel,
        length_penalty,
        lm_scale,
        nbest,
        decode_type: DecodeType,
    ):
        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        config = self.create_decoder_config()
        config.online_opts.decoder_opts.blank_penalty = blank_penalty
        config.online_opts.decoder_opts.blank_ilabel = blank_ilabel
        config.online_opts.decoder_opts.length_penalty = length_penalty
        config.online_opts.lattice_postprocessor_opts.lm_scale = lm_scale
        config.online_opts.lattice_postprocessor_opts.nbest = nbest

        word_id_to_word_str = load_word_symbols(os.path.join(graph_path, "words.txt"))

        decoder = BatchedMappedDecoderCuda(
            config,
            os.path.join(graph_path, "TLG.fst"),
            os.path.join(graph_path, "words.txt"),
            num_tokens_including_blank,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.online_opts.num_channels, num_workers=0, collate_fn=collate_fn, pin_memory=True
        )

        references = []
        results = []
        all_utterance_ids = []
        all_greedy_predictions = []

        total_audio_length_samples = 0

        # wer = torchmetrics.WordErrorRate()
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if half_precision:
                stack.enter_context(torch.autocast("cuda"))
            start_time = time.time_ns()
            for batch in data_loader:
                torch.cuda.nvtx.range_push("batch")
                input_signal, input_signal_length, target, utterance_ids = batch
                all_utterance_ids.extend(utterance_ids)
                total_audio_length_samples += torch.sum(input_signal_length)
                references.extend(target)
                torch.cuda.nvtx.range_push("H2D")
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("forward")
                log_probs, lengths, _ = asr_model.forward(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )
                log_probs *= acoustic_scale
                torch.cuda.nvtx.range_pop()
                # from IPython import embed; embed()
                # greedy_predictions, _ = asr_model.decoding.ctc_decoder_predictions_tensor(log_probs, decoder_lengths=lengths, return_hypotheses=False)
                # all_greedy_predictions.extend(greedy_predictions)

                torch.cuda.nvtx.range_push("D2H lengths")
                cpu_lengths = lengths.to(torch.int64).to('cpu')

                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("beam search decoder")
                if decode_type == DecodeType.MBR:
                    results.extend(decoder.decode_mbr(log_probs.to(torch.float32), cpu_lengths))
                elif decode_type == DecodeType.NBEST:
                    results.extend(decoder.decode_nbest(log_probs.to(torch.float32), cpu_lengths))
                else:
                    assert False, decode_type
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
            end_time = time.time_ns()
        run_time_seconds = (end_time - start_time) / 1_000_000_000
        input_time_seconds = total_audio_length_samples / 16_000
        print("non-warmed-up RTFx:", input_time_seconds / run_time_seconds)
        # print("run time:", run_time_seconds)
        predictions = []
        if decode_type == DecodeType.MBR:
            for result in results:
                predictions.append(" ".join(piece[0] for piece in result))
        elif decode_type == DecodeType.NBEST:
            for result in results:
                # Just get first best result for WER comparison
                result = result[0]
                predictions.append(" ".join(word_id_to_word_str[word] for word in result.words))
        references = [s.lower() for s in references]
        # Might want to try a different WER implementation, for sanity.
        my_wer = wer(references, predictions)
        other_wer = word_error_rate(references, predictions)
        print("beam search WER:", my_wer)
        print("other beam search WER:", other_wer)
        # print("greedy WER:", wer(references, all_greedy_predictions))
        return my_wer[0], predictions, references

    def run_decoder_flashlight2(self,
                                asr_model,
                                dataset: torch.utils.data.IterableDataset,
                                half_precision: bool,
                                # acoustic_scale,
                                # blank_penalty,
                                blank_ilabel,
                                # length_penalty,
                                # lm_scale,
                                nemo_model_name,
                                dataset_name,
    ):
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        ctc_infer = BeamCTCInfer(blank_id=blank_ilabel,
                                 beam_size=16,
                                 kenlm_path=os.path.join(self.temp_dir, 'lm.bin'),
                                 flashlight_cfg=FlashlightConfig(
                                     lexicon_path=os.path.join(
                                         self.temp_dir, 'flashlight_lexicon.txt/3-gram.pruned.3e-7.lexicon'
                                     ),
                                     beam_size_token=16,
                                     beam_threshold=20.0,
                                 ))
        ctc_infer.set_vocabulary(asr_model.tokenizer.tokenizer.get_vocab())
        ctc_infer.set_decoding_type("subword")
        ctc_infer.set_tokenizer(asr_model.tokenizer)


        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=400, num_workers=0, collate_fn=collate_fn, pin_memory=True
        )

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if half_precision:
                stack.enter_context(torch.autocast("cuda"))

            warmup_iters = 1
            benchmark_iters = 1

            total_audio_length_samples = 0

            references = []
            hypotheses = []

            for i in range(warmup_iters + benchmark_iters):
                if i == warmup_iters:
                    start_time = time.time_ns()
                    torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("iteration")
                for batch in data_loader:
                    torch.cuda.nvtx.range_push("single batch")
                    input_signal, input_signal_length, target, utterance_ids = batch
                    references.extend(target)
                    if i == 0:
                        total_audio_length_samples += torch.sum(input_signal_length) * benchmark_iters
                    input_signal = input_signal.cuda()
                    input_signal_length = input_signal_length.cuda()
                    torch.cuda.nvtx.range_push("ASR model")
                    log_probs, lengths, _ = asr_model.forward(
                        input_signal=input_signal, input_signal_length=input_signal_length
                    )
                    torch.cuda.nvtx.range_pop()
                    cpu_lengths = lengths.to(torch.int64).to('cpu')
                    torch.cuda.nvtx.range_push("decoder")
                    _hypotheses = ctc_infer.flashlight_beam_search(log_probs.to(torch.float32), cpu_lengths)

                    for nbest_hypothesis in _hypotheses:
                        text = asr_model.tokenizer.ids_to_text(nbest_hypothesis.n_best_hypotheses[0].y_sequence)
                        hypotheses.append(text)
                        # print("GALVEZ:", text)

                    # for hyp in _hypotheses:
                    #     print("GALVEZ:", asr_model.tokenizer.ids_to_text(hyp.n_best_hypotheses[0].y_sequence))

                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()  # iteration
        end_time = time.time_ns()
        run_time_seconds = (end_time - start_time) / 1_000_000_000
        input_time_seconds = total_audio_length_samples / 16_000
        print("RTFx:", input_time_seconds / run_time_seconds, nemo_model_name, dataset_name)

        references = [s.lower() for s in references]
        my_wer = wer(references, hypotheses)
        print("WER:", my_wer)

        torch.cuda.cudart().cudaProfilerStop()


    # 336.2256739139557 seconds
    def run_decoder_flashlight(self, dataset_string):
        asr_model = nemo_asr.models.ASRModel.restore_from(self.nemo_model_path, map_location=torch.device("cuda"))

        # subprocess.check_call(shlex.split(f"python /home/dgalvez/scratch/code/asr/riva-asrlib-decoder/NeMo/scripts/asr_language_modeling/ngram_lm/create_lexicon_from_arpa.py --arpa {self.lm_path} --model {self.nemo_model_path} --lower --dst {os.path.join(self.temp_dir, 'flashlight_lexicon.txt')}"))
        # lowercase_lm_path = os.path.join(self.temp_dir, "lowercase.arpa")
        # subprocess.check_call(f"tr '[:upper:]' '[:lower:]' < {self.lm_path} > {lowercase_lm_path}", shell=True)

        # kenlm_args = [
        #     "/home/dgalvez/scratch/code/asr/riva-asrlib-decoder/NeMo/scripts/asr_language_modeling/ngram_lm/decoders/kenlm/build/bin/build_binary",
        #     "trie",
        #     lowercase_lm_path,
        #     os.path.join(self.temp_dir, 'lm.bin')
        # ]
        # subprocess.check_call(kenlm_args)

        decoding_cfg = CTCDecodingConfig()

        decoding_cfg.strategy = "flashlight"
        decoding_cfg.beam.search_type = "flashlight"
        decoding_cfg.beam.kenlm_path = os.path.join(self.temp_dir, 'lm.bin')
        decoding_cfg.beam.flashlight_cfg.lexicon_path = os.path.join(
            self.temp_dir, 'flashlight_lexicon.txt/3-gram.pruned.3e-7.lexicon'
        )
        decoding_cfg.beam.beam_size = 32
        # Why set both of these small? It's a bit strange, isn't it?
        decoding_cfg.beam.beam_alpha = 0.2
        decoding_cfg.beam.beam_beta = 0.2
        decoding_cfg.beam.flashlight_cfg.beam_size_token = 32
        decoding_cfg.beam.flashlight_cfg.beam_threshold = 25.0

        asr_model.change_decoding_strategy(decoding_cfg)

        file_paths = []
        durations = []
        references = []
        for i in range(len(self.original_dataset_map[dataset_string])):
            file_path, _, transcript, *_ = self.original_dataset_map[dataset_string].get_metadata(i)
            waveform, sample_rate, *_ = self.original_dataset_map[dataset_string][i]
            file_paths.append(os.path.join(self.temp_dir, "LibriSpeech", file_path))
            durations.append(waveform.size(1) / sample_rate)
            references.append(transcript.lower())

        # Need to permute these arrays based on length as well...

        # files = glob.glob(os.path.join(self.temp_dir, "LibriSpeech/test-clean/*/*/*.flac"))
        start_time = time.time()
        predictions = asr_model.transcribe(
            paths2audio_files=file_paths, batch_size=160, return_hypotheses=False, logprobs=False
        )
        end_time = time.time()
        print("GALVEZ:difference=", end_time - start_time)
        # print(predictions[:2])
        # print(references[:2])
        my_wer = wer(references, predictions)
        print(f"GALVEZ: {dataset_string} flashlight wer=", my_wer)
        return predictions, references, durations, file_paths

    def run_decoder_throughput(
        self,
        asr_model,
        graph_path: str,
        dataset: torch.utils.data.IterableDataset,
        half_precision: bool,
        acoustic_scale,
        blank_penalty,
        blank_ilabel,
        length_penalty,
        lm_scale,
        nbest,
        warmup_iters: int,
        benchmark_iters: int,
        decode_type: DecodeType,
    ):
        assert warmup_iters > 0
        assert benchmark_iters > 0

        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        config = self.create_decoder_config()
        config.online_opts.decoder_opts.blank_penalty = blank_penalty
        config.online_opts.decoder_opts.blank_ilabel = blank_ilabel
        config.online_opts.decoder_opts.length_penalty = length_penalty
        config.online_opts.lattice_postprocessor_opts.lm_scale = lm_scale
        config.online_opts.lattice_postprocessor_opts.nbest = nbest
        decoder = BatchedMappedDecoderCuda(
            config,
            os.path.join(graph_path, "TLG.fst"),
            os.path.join(graph_path, "words.txt"),
            num_tokens_including_blank,
        )

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.online_opts.num_channels, num_workers=0, collate_fn=collate_fn, pin_memory=True
        )

        total_audio_length_samples = 0

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if half_precision:
                stack.enter_context(torch.autocast("cuda"))

            for i in range(warmup_iters + benchmark_iters):
                if i == warmup_iters:
                    start_time = time.time_ns()
                    torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("iteration")
                for batch in data_loader:
                    torch.cuda.nvtx.range_push("single batch")
                    input_signal, input_signal_length, target, utterance_ids = batch
                    if i == 0:
                        total_audio_length_samples += torch.sum(input_signal_length) * benchmark_iters
                    input_signal = input_signal.cuda()
                    input_signal_length = input_signal_length.cuda()
                    torch.cuda.nvtx.range_push("ASR model")
                    log_probs, lengths, _ = asr_model.forward(
                        input_signal=input_signal, input_signal_length=input_signal_length
                    )
                    log_probs *= acoustic_scale
                    torch.cuda.nvtx.range_pop()
                    cpu_lengths = lengths.to(torch.int64).to('cpu')
                    torch.cuda.nvtx.range_push("decoder")
                    if decode_type == DecodeType.MBR:
                        decoder.decode_mbr(log_probs.to(torch.float32), cpu_lengths)
                    elif decode_type == DecodeType.NBEST:
                        decoder.decode_nbest(log_probs.to(torch.float32), cpu_lengths)
                    else:
                        assert False, decode_type
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()  # iteration
        end_time = time.time_ns()
        run_time_seconds = (end_time - start_time) / 1_000_000_000
        input_time_seconds = total_audio_length_samples / 16_000
        print("RTFx:", input_time_seconds / run_time_seconds)
        torch.cuda.cudart().cudaProfilerStop()


    def test_pad_vs_tensor_list(self):
        work_dir = os.path.join(self.temp_dir, "ctc")
        nemo_model_name = "stt_en_conformer_ctc_small"

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_name, map_location=torch.device("cuda")
        )

        self.create_TLG("ctc", work_dir, nemo_model_name)

        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        torch.cuda.cudart().cudaProfilerStart()

        decoder_config = self.create_decoder_config()

        data_loader = torch.utils.data.DataLoader(
            self.dataset_map["test-clean"],
            batch_size=decoder_config.online_opts.max_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            for batch in data_loader:
                input_signal, input_signal_length, target, utterance_ids = batch
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                log_probs, lengths, _ = asr_model.forward(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )
                cpu_lengths = lengths.to(torch.int64).to('cpu')

                decoder = BatchedMappedDecoderCuda(
                    decoder_config,
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
                    num_tokens_including_blank,
                )

                log_probs = log_probs.to(torch.float32)
                pad_results = decoder.decode_mbr(log_probs, cpu_lengths)

                log_probs_list = []
                for i, length in enumerate(cpu_lengths):
                    log_probs_list.append(log_probs[i, :length, :])
                list_results = decoder.decode_mbr(log_probs_list, cpu_lengths)
                assert pad_results == list_results


                pad_results_nbest = decoder.decode_nbest(log_probs, cpu_lengths)
                list_results_nbest = decoder.decode_nbest(log_probs_list, cpu_lengths)
                print("GALVEZ:", pad_results_nbest)
                print("GALVEZ:", list_results_nbest)
                assert pad_results_nbest == list_results_nbest
                # break

        torch.cuda.cudart().cudaProfilerStop()
        # data_loader = torch.utils.data.DataLoader(
        #     self.dataset_map["test-clean"],
        #     batch_size=decoder1_config.online_opts.max_batch_size,
        #     collate_fn=collate_fn,
        #     pin_memory=True,
        # )


    @pytest.mark.xfail
    def test_online_start_stop_decoder(self):
        nemo_model_name = "stt_en_conformer_ctc_small"

        work_dir = os.path.join(self.temp_dir, "ctc")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc", work_dir, nemo_model_name)
        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        offline_config = self.create_decoder_config()
        offline_config.online_opts.max_batch_size = 16
        config = offline_config.online_opts


        decoder = BatchedMappedOnlineDecoderCuda(
            config,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            num_tokens_including_blank,
        )

        data_loader = torch.utils.data.DataLoader(
            self.dataset_map["test-clean"],
            batch_size=config.max_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_name, map_location=torch.device("cuda")
        )

        half_precision = True
        acoustic_scale = 2.2

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        word_id_to_word_str = load_word_symbols(os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"))

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if half_precision:
                stack.enter_context(torch.autocast("cuda"))
            for batch in data_loader:
                input_signal, input_signal_length, target, utterance_ids = batch
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                log_probs, lengths, _ = asr_model.forward(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )
                log_probs *= acoustic_scale

                cpu_lengths = lengths.to(torch.int64).to('cpu').tolist()

                batch_size = log_probs.shape[0]
                corr_ids = list(range(batch_size))
                for corr_id in corr_ids:
                    success = decoder.try_init_corr_id(corr_id)
                    # Do SetLatticeCallback() here if you want
                    # Is there some way that I can get the lattice other than callbacks?
                    assert success
                log_probs_list = [0] * batch_size
                is_first_chunk = [0] * batch_size
                is_last_chunk = [0] * batch_size
                for i in range(batch_size):
                    log_probs_list[i] = log_probs[i, :cpu_lengths[i], :]
                    is_first_chunk[i] = True
                    is_last_chunk[i]  = True
                channels, full_partial_hypotheses = \
                    decoder.decode_batch(corr_ids, log_probs_list,
                                         is_first_chunk, is_last_chunk)

                
                for corr_id in corr_ids:
                    success = decoder.try_init_corr_id(corr_id)
                    # Do SetLatticeCallback() here if you want
                    # Is there some way that I can get the lattice other than callbacks?
                    assert success
                log_probs_list = [0] * batch_size
                is_first_chunk = [0] * batch_size
                is_last_chunk = [0] * batch_size
                for i in range(batch_size):
                    log_probs_list[i] = log_probs[i, :cpu_lengths[i] // 2, :]
                    is_first_chunk[i] = True
                    is_last_chunk[i]  = False
                channels, chunked_partial_hypotheses1 = \
                    decoder.decode_batch(corr_ids, log_probs_list,
                                         is_first_chunk, is_last_chunk)
                for i in range(batch_size):
                    log_probs_list[i] = log_probs[i, cpu_lengths[i] // 2:, :]
                    is_first_chunk[i] = False
                    is_last_chunk[i]  = True
                channels, chunked_partial_hypotheses2 = \
                    decoder.decode_batch(corr_ids, log_probs_list,
                                         is_first_chunk, is_last_chunk)

                for ph, ph2 in zip(full_partial_hypotheses, chunked_partial_hypotheses2):
                    print("full:", ph.score, " chunked:", ph2.score)
                    print("Full:", " ".join(word_id_to_word_str[word] for word in ph.words))
                    print("Chunked:", " ".join(word_id_to_word_str[word] for word in ph2.words))
                    if abs(ph.score - ph2.score) > 0.1:
                        print("Different scores")
                    # this assertion fails sometimes
                    assert abs(ph.score - ph2.score) <= 0.1
                    assert ph.words == ph2.words

    @pytest.mark.ci
    def test_online_decoder(self):
        nemo_model_name = "stt_en_conformer_ctc_small"

        work_dir = os.path.join(self.temp_dir, "ctc")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc", work_dir, nemo_model_name)
        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        offline_config = self.create_decoder_config()
        offline_config.online_opts.max_batch_size = 32
        # Make sure that that the offline decoder does not use final
        # probabilities when computing a best path.
        offline_config.online_opts.use_final_probs = False
        config = offline_config.online_opts


        decoder = BatchedMappedOnlineDecoderCuda(
            config,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            num_tokens_including_blank,
        )

        offline_decoder = BatchedMappedDecoderCuda(
            offline_config,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            num_tokens_including_blank,
        )

        # TODO: Need to test case where we try to input a larger batch
        # size than config.max_batch_size
        data_loader = torch.utils.data.DataLoader(
            self.dataset_map["test-clean"],
            batch_size=config.max_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_name, map_location=torch.device("cuda")
        )

        half_precision = True
        acoustic_scale = 2.2

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        word_id_to_word_str = load_word_symbols(os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"))

        problem_transcripts = []

        start_time = time.time()
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if half_precision:
                stack.enter_context(torch.autocast("cuda"))
            for j, batch in enumerate(data_loader):
                input_signal, input_signal_length, target, utterance_ids = batch
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                log_probs, lengths, _ = asr_model.forward(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )
                log_probs *= acoustic_scale

                cpu_lengths = lengths.to(torch.int64).to('cpu').tolist()

                batch_size = log_probs.shape[0]
                corr_ids = list(range(batch_size))
                for corr_id in corr_ids:
                    success = decoder.try_init_corr_id(corr_id)
                    # Do SetLatticeCallback() here if you want
                    # Is there some way that I can get the lattice other than callbacks?
                    assert success
                # Are my results going to be contiguous in general?
                # log_probs_ptrs = [0] * batch_size
                log_probs_list = [0] * batch_size
                is_first_chunk = [0] * batch_size
                is_last_chunk = [0] * batch_size
                # print("GALVEZ:")
                # torch.cuda.synchronize()
                for i in range(batch_size):
                    # Do I need itemsize() here???
                    # log_probs_ptrs[i] = log_probs.data_ptr() + log_probs.stride(0) * log_probs.element_size() * i
                    log_probs_list[i] = log_probs[i, :cpu_lengths[i], :]
                    is_first_chunk[i] = True
                    is_last_chunk[i]  = True
                # print(corr_ids)
                # print(log_probs_list)
                # print(is_first_chunk)
                # print(is_last_chunk)
                channels, partial_hypotheses = \
                    decoder.decode_batch(corr_ids, log_probs_list,
                                         is_first_chunk, is_last_chunk)
                nbest_results = offline_decoder.decode_nbest(log_probs_list, lengths.to(torch.int64).to('cpu')[:batch_size])
                for nbest_result, ph in zip(nbest_results, partial_hypotheses):
                    print("offline:", nbest_result[0].score, " online:", ph.score)
                    print("Offline:", " ".join(word_id_to_word_str[word] for word in nbest_result[0].words))
                    print("Online:", " ".join(word_id_to_word_str[word] for word in ph.words))
                    print("Start times")
                    print("Offline:", [round(start_time / 0.04) for start_time in nbest_result[0].word_start_times_seconds])
                    print("Online:", [start_time for start_time in ph.word_start_times_frames])
                    print("End times")
                    print("Offline:", [round((start_time + duration) / 0.04) for start_time, duration in
                          zip(nbest_result[0].word_start_times_seconds,
                              nbest_result[0].word_durations_seconds)])
                    print("Online:", [end_time for end_time in ph.word_end_times_frames])
                    # Not a reliable test
                    if nbest_result[0].words != ph.words:
                        problem_transcripts.append((nbest_result[0].words, ph.words))
                    # assert [w for w in nbest_result[0].words if w != 0] == ph.words
                # for ph in partial_hypotheses:
                #     print("Online:", " ".join(word_id_to_word_str[word] for word in ph.words))
                # break
                # print(partial_hypotheses)
                # if j == 10:
                #     break
                # print("GALVEZ:iter=",j)
        end_time = time.time()
        print("Total time:", end_time - start_time)

        if len(problem_transcripts) > 0:
            print("There are some problematic transcripts that differ between offline and online mode.")
            for t1, t2 in problem_transcripts:
                print(t1)
                print("vs")
                print(t2)
        assert len(problem_transcripts) < 5, "Too many transcripts differ"

    def test_online_one_step(self):
        nemo_model_name = "stt_en_conformer_ctc_small"

        work_dir = os.path.join(self.temp_dir, "ctc")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, map_location=torch.device("cuda"))
        self.create_TLG("ctc", work_dir, nemo_model_name)
        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        offline_config = self.create_decoder_config()
        offline_config.online_opts.max_batch_size = 1
        config = offline_config.online_opts

        decoder = BatchedMappedOnlineDecoderCuda(
            config,
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            num_tokens_including_blank,
        )

        data_loader = torch.utils.data.DataLoader(
            self.original_dataset_map["test-clean"],
            batch_size=config.max_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            nemo_model_name, map_location=torch.device("cuda")
        )

        half_precision = True
        acoustic_scale = 2.2

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        word_id_to_word_str = load_word_symbols(os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"))

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if half_precision:
                stack.enter_context(torch.autocast("cuda"))
            for batch in data_loader:
                input_signal, input_signal_length, target, utterance_ids = batch
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                log_probs, lengths, _ = asr_model.forward(
                    input_signal=input_signal, input_signal_length=input_signal_length
                )
                log_probs *= acoustic_scale

                batch_size = config.max_batch_size  # log_probs.shape[0]
                cpu_lengths = [1] * batch_size
                corr_ids = list(range(batch_size))
                for corr_id in corr_ids:
                    success = decoder.try_init_corr_id(corr_id)
                    # Do SetLatticeCallback() here if you want
                    # Is there some way that I can get the lattice other than callbacks?
                    assert success
                log_probs_list = [0] * batch_size
                is_first_chunk = [0] * batch_size
                is_last_chunk = [0] * batch_size
                for i in range(batch_size):
                    log_probs_list[i] = log_probs[i, :cpu_lengths[i], :]
                    is_first_chunk[i] = True
                    # Not really true...
                    is_last_chunk[i]  = True
                channels, full_partial_hypotheses = \
                    decoder.decode_batch(corr_ids, log_probs_list,
                                         is_first_chunk, is_last_chunk)

                for ph in full_partial_hypotheses:
                    print("Score:", ph.score)
                    print("ilabels:", ph.ilabels)
                    print("Words:", " ".join(word_id_to_word_str[word] for word in ph.words))
                break

def write_ctm_output(key, result):
    for word, start, end in result:
        print(f"{key} 1 {start:.2f} {end - start:.2f} {word} 1.0")


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors = []
    targets = []
    lengths = []
    utterance_ids = []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, speaker_id, chapter_id, utterance_id in batch:
        waveform = waveform.squeeze()
        tensors += [waveform]
        # targets += [torch.zeros(1)]
        targets.append(label)
        lengths.append(waveform.size(0))
        utterance_ids.append(f"{speaker_id}-{chapter_id}-{utterance_id}")
        # targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)
    # targets = torch.stack(targets)

    return tensors, lengths, targets, utterance_ids


def trace_back_stats(r, h, d):
    i = len(r)
    j = len(h)
    insertions = 0
    substitutions = 0
    deletions = 0
    while True:
        if i == 0 and j == 0:
            break
        elif i >= 1 and j >= 1 and d[i][j] == d[i - 1][j - 1] and r[i - 1] == h[j - 1]:
            i = i - 1
            j = j - 1
        elif j >= 1 and d[i][j] == d[i][j - 1] + 1:
            insertions += 1
            i = i
            j = j - 1
        elif i >= 1 and j >= 1 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i = i - 1
            j = j - 1
        else:
            deletions += 1
            i = i - 1
            j = j
    return insertions, substitutions, deletions


def wer(references, hypotheses):
    total_wer = 0
    total_insertions = 0
    total_substitutions = 0
    total_deletions = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        transcript_wer, i, s, d = wer_single(ref.split(), hyp.split())
        total_words += len(ref.split())
        total_wer += transcript_wer
        total_insertions += i
        total_substitutions += s
        total_deletions += d
    if total_words == 0:
        wer_ratio = 0
    else:
        wer_ratio = total_wer / total_words

    return wer_ratio, total_insertions, total_substitutions, total_deletions


def wer_single(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 2**16 - 1 elements (uint16).
    O(nm) time and space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    """
    # initialisation

    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.uint16)
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1].lower() == h[j - 1].lower():
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    insertions, substitions, deletions = trace_back_stats(r, h, d)

    assert insertions + substitions + deletions == d[len(r)][len(h)]

    return d[len(r)][len(h)], insertions, substitions, deletions
