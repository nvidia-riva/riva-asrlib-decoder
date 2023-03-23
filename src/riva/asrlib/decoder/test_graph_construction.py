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
import zipfile
from contextlib import ExitStack

import more_itertools
import nemo.collections.asr as nemo_asr
import numpy as np
import pytest
import torch
import torchaudio
import torchmetrics
from nemo.collections.asr.metrics.wer import CTCDecodingConfig
from nemo.collections.asr.models import ASRModel
from ruamel.yaml import YAML
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

import riva.asrlib.decoder
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedDecoderCudaConfig

# os.environ["TORCH_CUDNN_V8_API_ENABLED"]="1"


# importing this causes pytest never to finish "collecting". No idea why.
# import pywrapfst


logging.getLogger("nemo").setLevel(logging.INFO)


# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


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
        self.create_TLG("ctc_eesen", os.path.join(self.temp_dir, "ctc_eesen"))

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
            # Something is going wrong for the large models. Graph construction probably went wrong.
            # ("stt_en_conformer_ctc_large", "test-clean", 0.025943396226415096, False),
            # ("stt_en_conformer_ctc_large", "test-clean", 0.025924376141205113, True),
            # ("stt_en_conformer_ctc_large", "test-other", 0.04447586114666718, False),
            # ("stt_en_conformer_ctc_large", "test-other", 0.04460959440612881, True),
            # ("stt_en_conformer_ctc_large", "dev-clean", 1.025943396226415096, False),
            # ("stt_en_conformer_ctc_large", "dev-clean", 1.025924376141205113, True),
            # ("stt_en_conformer_ctc_large", "dev-other", 1.04447586114666718, False),
            # ("stt_en_conformer_ctc_large", "dev-other", 1.04460959440612881, True),
        ],
    )
    def test_vanilla_ctc_topo_wer(self, nemo_model_name, dataset, expected_wer, half_precision):
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
        )
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(f"GALVEZ:model={nemo_model_name} dataset={dataset} vanilla half_precision={half_precision} wer={my_wer}")
        # Accept a very tiny margin of error
        assert my_wer <= expected_wer + 0.0001

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
            # Something is going wrong for the large models. Graph construction probably went wrong.
            # ("stt_en_conformer_ctc_large", "test-clean", False),
            # ("stt_en_conformer_ctc_large", "test-clean", True),
            # ("stt_en_conformer_ctc_large", "test-other", False),
            # ("stt_en_conformer_ctc_large", "test-other", True),
            # ("stt_en_conformer_ctc_large", "dev-clean", False),
            # ("stt_en_conformer_ctc_large", "dev-clean", True),
            # ("stt_en_conformer_ctc_large", "dev-other", False),
            # ("stt_en_conformer_ctc_large", "dev-other", True),
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
            # Something is going wrong for the large models. Graph construction probably went wrong.
            # ("stt_en_conformer_ctc_large", "test-clean", 0.025943396226415096, False),
            # ("stt_en_conformer_ctc_large", "test-clean", 0.025924376141205113, True),
            # ("stt_en_conformer_ctc_large", "test-other", 0.04447586114666718, False),
            # ("stt_en_conformer_ctc_large", "test-other", 0.04460959440612881, True),
            # ("stt_en_conformer_ctc_large", "dev-clean", 1.025943396226415096, False),
            # ("stt_en_conformer_ctc_large", "dev-clean", 1.025924376141205113, True),
            # ("stt_en_conformer_ctc_large", "dev-other", 1.04447586114666718, False),
            # ("stt_en_conformer_ctc_large", "dev-other", 1.04460959440612881, True),
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
        )
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(f"GALVEZ:model={nemo_model_name} dataset={dataset} compact half_precision={half_precision} wer={my_wer}")
        # Accept a very tiny margin of error
        assert my_wer <= expected_wer + 0.0001

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
            # Something is going wrong for the large models. Graph construction probably went wrong.
            # ("stt_en_conformer_ctc_large", "test-clean", False),
            # ("stt_en_conformer_ctc_large", "test-clean", True),
            # ("stt_en_conformer_ctc_large", "test-other", False),
            # ("stt_en_conformer_ctc_large", "test-other", True),
            # ("stt_en_conformer_ctc_large", "dev-clean", False),
            # ("stt_en_conformer_ctc_large", "dev-clean", True),
            # ("stt_en_conformer_ctc_large", "dev-other", False),
            # ("stt_en_conformer_ctc_large", "dev-other", True),
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
        benchmark_iters = 2
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
        )

    @pytest.mark.parametrize(
        "nemo_model_name, dataset, expected_wer, half_precision",
        [
            ("stt_en_conformer_ctc_small", "test-clean", 0.03305690809494827, False),
            ("stt_en_conformer_ctc_small", "test-clean", 0.03294278758368838, True),
            ("stt_en_conformer_ctc_small", "test-other", 0.06854784784976023, False),
            ("stt_en_conformer_ctc_small", "test-other", 0.06936935215788166, True),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.03064225579941914, False),
            ("stt_en_conformer_ctc_small", "dev-clean", 0.03058711076798647, True),
            ("stt_en_conformer_ctc_small", "dev-other", 0.06746094056685248, False),
            ("stt_en_conformer_ctc_small", "dev-other", 0.0674805684227055, True),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.02697048082775411, False),
            ("stt_en_conformer_ctc_medium", "test-clean", 0.02697048082775411, True),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05446764610358596, False),
            ("stt_en_conformer_ctc_medium", "test-other", 0.05452496035764094, True),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.023565310098893424, False),
            ("stt_en_conformer_ctc_medium", "dev-clean", 0.024061615381787433, True),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.051640888749313024, False),
            ("stt_en_conformer_ctc_medium", "dev-other", 0.051346470911517623, True),
            # Something is going wrong for the large models. Graph construction probably went wrong.
            # ("stt_en_conformer_ctc_large", "test-clean", 0.025943396226415096, False),
            # ("stt_en_conformer_ctc_large", "test-clean", 0.025924376141205113, True),
            # ("stt_en_conformer_ctc_large", "test-other", 0.04447586114666718, False),
            # ("stt_en_conformer_ctc_large", "test-other", 0.04460959440612881, True),
            # ("stt_en_conformer_ctc_large", "dev-clean", 1.025943396226415096, False),
            # ("stt_en_conformer_ctc_large", "dev-clean", 1.025924376141205113, True),
            # ("stt_en_conformer_ctc_large", "dev-other", 1.04447586114666718, False),
            # ("stt_en_conformer_ctc_large", "dev-other", 1.04460959440612881, True),
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
        )
        print("GALVEZ:", acoustic_scale, blank_ilabel, blank_penalty, length_penalty, lm_scale)
        print(
            f"GALVEZ:model={nemo_model_name} dataset={dataset} identity half_precision={half_precision} wer={my_wer}"
        )
        # Accept a very tiny margin of error
        assert my_wer <= expected_wer + 0.0001

    # Skip flashlight tests. They use hard-coded paths for flashlight
    # because the code to create a kenlm language model is not
    # directly part of the NeMo library.
    @pytest.mark.skip
    @pytest.mark.parametrize("dataset_name,", ["test-clean", "test-other"])
    def test_flashlight_alone(self, dataset_name):
        _ = self.run_decoder_flashlight(dataset_name)

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
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            "stt_en_conformer_ctc_small", map_location=torch.device("cuda")
        )

        num_tokens_including_blank = len(asr_model.to_config_dict()["decoder"]["vocabulary"]) + 1

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        torch.cuda.cudart().cudaProfilerStart()

        work_dir = os.path.join(self.temp_dir, "ctc")

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
    ):
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
                # batch_results = decoder.decode_map(log_probs.to(torch.float32), cpu_lengths)
                # Would ideally like to pass it just as DLTensor instead of
                # results.extend(decoder.decode_map(log_probs.to(torch.float32), cpu_lengths))
                results.extend(decoder.decode_mbr(log_probs.to(torch.float32), cpu_lengths))
                # print("GALVEZ:", results[-1])
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
            end_time = time.time_ns()
        run_time_seconds = (end_time - start_time) / 1_000_000_000
        input_time_seconds = total_audio_length_samples / 16_000
        print("RTFx:", input_time_seconds / run_time_seconds)
        # print("run time:", run_time_seconds)
        predictions = []
        for result in results:
            predictions.append(" ".join(piece[0] for piece in result))
        references = [s.lower() for s in references]
        my_wer = wer(references, predictions)
        # assert len(all_utterance_ids) == len(results)
        # for utt_id, result in zip(all_utterance_ids, results):
        #     # Have nbest=1
        #     # Ugh, need to be able to output a dictionary mapping
        #     for nth_result in result:
        #         score = nth_result[0]
        #         words_and_times = nth_result[1]
        #         predictions.append((utt_id, " ".join(piece[0] for piece in words_and_times if piece[0] != "<eps>")))
        # references = [(utt_id, s.lower()) for utt_id, s in zip(all_utterance_ids, references)]
        my_wer = wer(references, predictions)
        print("beam search WER:", my_wer)
        # print("greedy WER:", wer(references, all_greedy_predictions))
        return my_wer[0], predictions, references

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
                    decoder.decode_mbr(log_probs.to(torch.float32), cpu_lengths)
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()  # iteration
        end_time = time.time_ns()
        run_time_seconds = (end_time - start_time) / 1_000_000_000
        input_time_seconds = total_audio_length_samples / 16_000
        print("RTFx:", input_time_seconds / run_time_seconds)
        print("run time:", run_time_seconds)
        torch.cuda.cudart().cudaProfilerStop()


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
