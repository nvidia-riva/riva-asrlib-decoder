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

import os
# os.environ["TORCH_CUDNN_V8_API_ENABLED"]="1"

import gzip
from contextlib import ExitStack
import json
import multiprocessing
import os
import pathlib
import pytest
import shutil
import subprocess
import tarfile
import tempfile
import time
import unittest
import zipfile

import more_itertools
import nemo.collections.asr as nemo_asr
import numpy as np
from ruamel.yaml import YAML
import torch
import torchaudio
import torchmetrics
from tqdm import tqdm

import riva.asrlib.decoder
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedDecoderCudaConfig

from torch.utils.data import Dataset, Subset

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
    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_dir = os.path.abspath("tmp_graph_construction")
        os.makedirs(self.temp_dir, exist_ok=True)
        # self.wfst_dir = os.path.join(self.temp_dir, "wfst")
        # os.makedirs(self.wfst_dir, exist_ok=True)
        # self.am_dir = os.path.join(self.temp_dir, "am")
        # os.makedirs(self.am_dir, exist_ok=True)

        lm_zip_file = os.path.join(self.temp_dir, "speechtotext_english_lm_deployable_v1.0.zip")
        if not os.path.exists(lm_zip_file):
            subprocess.check_call(
                f"wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tao/speechtotext_english_lm/versions/deployable_v1.0/zip -O {lm_zip_file}",
                shell=True,
            )
            with zipfile.ZipFile(lm_zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)

        am_zip_file = os.path.join(self.temp_dir, "stt_en_conformer_ctc_small_1.6.0.zip")
        if not os.path.exists(am_zip_file):
            subprocess.check_call(
                f"wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.6.0/zip -O {am_zip_file}",
                shell=True,
            )
            with zipfile.ZipFile(am_zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)

        # Work around: At the time of writing this test, the words.txt
        # file downloaded from NGC is simply a git lfs stub file, not
        # the actual file itself, so overwrite self.words_path by
        # exracting the symbol table from the arpa file
        lm_path = os.path.join(self.temp_dir, "3-gram.pruned.3e-7.arpa")
        self.words_path = os.path.join(self.temp_dir, "words.mixed_lm.3-gram.pruned.3e-7.txt")
        temp_words_path = os.path.join(self.temp_dir, "words_with_ids.txt")
        if not os.path.exists(temp_words_path):
            subprocess.check_call(
                [
                    os.path.join(riva.asrlib.decoder.__path__[0], "scripts/prepare_TLG_fst/bin/arpa2fst"),
                    f"--write-symbol-table={temp_words_path}",
                    lm_path,
                    "/dev/null",
                ]
            )
        self.gzipped_lm_path = lm_path + ".gz"
        with open(lm_path, 'rb') as f_in, gzip.open(self.gzipped_lm_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        with open(temp_words_path, "r") as words_with_ids_fh, open(self.words_path, "w") as words_fh:
            for word_with_id in words_with_ids_fh:
                word = word_with_id.split()[0].lower()
                if word in {"<eps>", "<s>", "</s>", "<unk>"}:
                    continue
                words_fh.write(word)
                words_fh.write("\n")

        self.nemo_model_path = os.path.join(self.temp_dir, "stt_en_conformer_ctc_small.nemo")
        config_yaml = os.path.join(self.temp_dir, "model_config.yaml")

        yaml = YAML(typ='safe')
        with tarfile.open(self.nemo_model_path, "r:gz") as tar_fh:
            with tar_fh.extractfile("./model_config.yaml") as fh:
                data = yaml.load(fh)
        self.units_txt = os.path.join(self.temp_dir, "units.txt")
        with open(self.units_txt, "w") as fh:
            for unit in data["decoder"]["vocabulary"]:
                fh.write(f"{unit}\n")

        self.num_tokens_including_blank = len(data["decoder"]["vocabulary"]) + 1
        assert self.num_tokens_including_blank == 1025

        librispeech_test_clean = CacheDataset(torchaudio.datasets.LIBRISPEECH(self.temp_dir, "test-clean", download=True))
        librispeech_test_other = CacheDataset(torchaudio.datasets.LIBRISPEECH(self.temp_dir, "test-other", download=True))

        self.dataset_map = {}

        for key, dataset in (("test_clean", librispeech_test_clean),
                             ("test_other", librispeech_test_other)):
            lengths = []
            for i in range(len(dataset)):
                waveform, *_ = dataset[i]
                lengths.append(waveform.size(1))
            sorted_indices = list(np.argsort(lengths))
            self.dataset_map[key] = Subset(dataset, sorted_indices)
        # self.dataset_map = {
        #     "test_clean": librispeech_test_clean,
        #     "test_other": librispeech_test_other,
        # }


    def test_eesen_ctc_topo(self):
        self.create_TLG("ctc_eesen", os.path.join(self.temp_dir, "ctc_eesen"))

    # TODO: Debug why these WERs are a bit higher than the ones reported here for "N-gram LM"
    # https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_small
    # The reason is probably that the language model is not the same. The NeMo LM is trained
    # on the Librispeech training text as well.
    @pytest.mark.parametrize("dataset, expected_wer, half_precision",
                             [("test_clean", 0.03509205721241631, False),
                              ("test_clean", 0.035263237979306146, True),
                              ("test_other", 0.07034369447681639, False),
                              ("test_other", 0.0706302657470913, True)
                             ])
    def test_vanilla_ctc_topo(self, dataset, expected_wer, half_precision):
        work_dir = os.path.join(self.temp_dir, "ctc")
        # self.create_TLG("ctc", work_dir)
        wer = self.run_decoder(work_dir, self.dataset_map[dataset], half_precision)
        assert wer <= expected_wer

        self.run_decoder_throughput(work_dir, self.dataset_map[dataset], half_precision, 2, 10)
        print("GALVEZ:wer=", wer)

    def test_delete_decoder(self):
        """Ensure that allocating a decoder, decoding with it, deleting it,
        and then reallocating a new one, and deocding with that one,
        does not crash.
        """
        asr_model = nemo_asr.models.ASRModel.restore_from(self.nemo_model_path, map_location=torch.device("cuda"))

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
            self.dataset_map["test_clean"],
            batch_size=decoder1_config.online_opts.max_batch_size,
            collate_fn=collate_fn,
            pin_memory=True)

        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            for batch in data_loader:
                input_signal, input_signal_length, target = batch
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                log_probs, lengths, _ = asr_model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
                cpu_lengths = lengths.to(torch.int64).to('cpu')

                decoder1 = BatchedMappedDecoderCuda(
                    decoder1_config,
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
                    self.num_tokens_including_blank
                )

                decoder1.decode_mbr(log_probs.to(torch.float32), cpu_lengths)

                del decoder1

                decoder2 = BatchedMappedDecoderCuda(
                    decoder2_config,
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
                    os.path.join(work_dir, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
                    self.num_tokens_including_blank
                )
                decoder2.decode_mbr(log_probs.to(torch.float32), cpu_lengths)
                break

        torch.cuda.cudart().cudaProfilerStop()
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

    @staticmethod
    def create_decoder_config():
        config = BatchedMappedDecoderCudaConfig()
        config.n_input_per_chunk = 50
        config.online_opts.decoder_opts.default_beam = 17.0
        config.online_opts.decoder_opts.lattice_beam = 8.0
        config.online_opts.decoder_opts.max_active = 10_000
        # From when I was testing some penalties discussed in wenet. These can be added back later.
        # config.online_opts.decoder_opts.blank_penalty = 0.01
        # config.online_opts.decoder_opts.blank_ilabel = 1024
        # config.online_opts.decoder_opts.length_penalty = -4.5
        config.online_opts.determinize_lattice = True
        config.online_opts.max_batch_size = 160
        config.online_opts.num_channels = config.online_opts.max_batch_size * 2
        config.online_opts.frame_shift_seconds = 0.04
        config.online_opts.lattice_postprocessor_opts.acoustic_scale = 10.0
        config.online_opts.lattice_postprocessor_opts.lm_scale = 6.0
        config.online_opts.lattice_postprocessor_opts.word_ins_penalty = 0.0
        config.online_opts.num_decoder_copy_threads = 2
        config.online_opts.num_post_processing_worker_threads = multiprocessing.cpu_count() - config.online_opts.num_decoder_copy_threads

        return config

    def run_decoder(self, graph_path: str, dataset: torch.utils.data.IterableDataset, half_precision: bool):
        asr_model = nemo_asr.models.ASRModel.restore_from(self.nemo_model_path, map_location=torch.device("cuda"))

        config = self.create_decoder_config()
        decoder = BatchedMappedDecoderCuda(
            config,
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            self.num_tokens_including_blank
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.online_opts.num_channels,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True)

        references = []
        results = []
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
                input_signal, input_signal_length, target = batch
                total_audio_length_samples += torch.sum(input_signal_length)
                references.extend(target)
                torch.cuda.nvtx.range_push("H2D")
                input_signal = input_signal.cuda()
                input_signal_length = input_signal_length.cuda()
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("forward")
                log_probs, lengths, _ = asr_model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
                torch.cuda.nvtx.range_pop()
                # from IPython import embed; embed()
                # greedy_predictions, _ = asr_model.decoding.ctc_decoder_predictions_tensor(log_probs, decoder_lengths=lengths, return_hypotheses=False)
                # all_greedy_predictions.extend(greedy_predictions)

                torch.cuda.nvtx.range_push("D2H lengths")
                cpu_lengths = lengths.to(torch.int64).to('cpu')
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push("beam search decoder")
                results.extend(decoder.decode_mbr(log_probs.to(torch.float32), cpu_lengths))
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
            end_time = time.time_ns()
        run_time_seconds = (end_time - start_time) / 1_000_000_000
        input_time_seconds = total_audio_length_samples / 16_000
        # print("RTFx:", input_time_seconds / run_time_seconds)
        # print("run time:", run_time_seconds)
        predictions = []
        for result in results:
            predictions.append(" ".join(piece[0] for piece in result))
        references = [s.lower() for s in references]
        my_wer = wer(references, predictions)
        print("beam search WER:", my_wer)
        # print("greedy WER:", wer(references, all_greedy_predictions))
        return my_wer[0]

    def run_decoder_throughput(self, graph_path: str, dataset: torch.utils.data.IterableDataset,
                               half_precision: bool, warmup_iters: int, benchmark_iters: int):
        assert warmup_iters > 0
        assert benchmark_iters > 0
        asr_model = nemo_asr.models.ASRModel.restore_from(self.nemo_model_path, map_location=torch.device("cuda"))

        config = self.create_decoder_config()
        decoder = BatchedMappedDecoderCuda(
            config,
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            self.num_tokens_including_blank
        )

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.online_opts.num_channels,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True)

        total_audio_length_samples = 0

        # wer = torchmetrics.WordErrorRate()
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        # from torch._jit_internal import FunctionModifiers
        # asr_model.preprocessor.forward._torchscript_modifier = FunctionModifiers.IGNORE #  torch.jit.ignore(asr_model.preprocessor.forward)
        # asr_model.encoder = torch.jit.script(asr_model.encoder)
        # asr_model.decoder = torch.jit.script(asr_model.decoder)


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
                    input_signal, input_signal_length, target = batch
                    if i == 0:
                        total_audio_length_samples += torch.sum(input_signal_length) * benchmark_iters
                    input_signal = input_signal.cuda()
                    input_signal_length = input_signal_length.cuda()
                    torch.cuda.nvtx.range_push("ASR model")
                    log_probs, lengths, _ = asr_model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
                    torch.cuda.nvtx.range_pop()
                    cpu_lengths = lengths.to(torch.int64).to('cpu')
                    torch.cuda.nvtx.range_push("decoder")
                    decoder.decode_mbr(log_probs.to(torch.float32), cpu_lengths)
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                end_time = time.time_ns()
                torch.cuda.nvtx.range_pop() # iteration
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

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        waveform = waveform.squeeze()
        tensors += [waveform]
        # targets += [torch.zeros(1)]
        targets.append(label)
        lengths.append(waveform.size(0))
        # targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)
    # targets = torch.stack(targets)

    return tensors, lengths, targets

def trace_back_stats(r, h, d):
    i = len(r)
    j = len(h)
    insertions = 0
    substitutions = 0
    deletions = 0
    while True:
        if i == 0 and j == 0:
            break
        elif i >= 1 and j >= 1 and d[i][j] == d[i-1][j-1] and r[i-1] == h[j-1]:
            i = i - 1
            j = j - 1
        elif j >= 1 and d[i][j] == d[i][j-1]+1:
            insertions += 1
            i = i
            j = j - 1
        elif i >= 1 and j >= 1 and d[i][j] == d[i-1][j-1]+1:
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
    return total_wer / total_words, total_insertions, total_substitutions, total_deletions

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
