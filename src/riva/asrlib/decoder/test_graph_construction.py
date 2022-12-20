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
import multiprocessing
import os
import pathlib
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
        if not os.path.exists(temp_words_path):
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

    # @pytest.mark.parametrize("")
    def test_vanilla_ctc_topo(self):
        # self.create_TLG("ctc", os.path.join(self.temp_dir, "ctc"))
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
        config.online_opts.decoder_opts.max_active = 10_000
        # From when I was testing some penalties discussed in wenet. These can be added back later.
        # config.online_opts.decoder_opts.blank_penalty = 0.01
        # config.online_opts.decoder_opts.blank_ilabel = 1024
        # config.online_opts.decoder_opts.length_penalty = -4.5
        config.online_opts.determinize_lattice = True
        config.online_opts.max_batch_size = 200
        config.online_opts.num_channels = config.online_opts.max_batch_size * 2
        config.online_opts.frame_shift_seconds = 0.04
        config.online_opts.lattice_postprocessor_opts.acoustic_scale = 10.0
        config.online_opts.lattice_postprocessor_opts.lm_scale = 6.0
        config.online_opts.lattice_postprocessor_opts.word_ins_penalty = 0.0
        config.online_opts.num_post_processing_worker_threads = multiprocessing.cpu_count()
        config.online_opts.num_decoder_copy_threads = 4
        decoder = BatchedMappedDecoderCuda(
            config,
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/TLG.fst"),
            os.path.join(graph_path, "graph/graph_ctc_3-gram.pruned.3e-7/words.txt"),
            self.num_tokens_including_blank
        )

        librispeech = torchaudio.datasets.LIBRISPEECH(self.temp_dir, "test-clean", download=True)
        data_loader = torch.utils.data.DataLoader(
            librispeech,
            batch_size=config.online_opts.num_channels,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True)
        # batch_size=config.online_opts.max_batch_size,
        # pin_memory=True,
        # shuffle=False,
        # num_workers=4)

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
        torch.cuda.cudart().cudaProfilerStart()
        with torch.inference_mode():
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
                results.extend(decoder.decode(log_probs.to(torch.float32), cpu_lengths))
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
            end_time = time.time_ns()
        torch.cuda.cudart().cudaProfilerStop()
        run_time_seconds = (end_time - start_time) / 1_000_000_000
        input_time_seconds = total_audio_length_samples / 16_000
        print("RTFx:", input_time_seconds / run_time_seconds)
        print("run time:", run_time_seconds)
        predictions = []
        for result in results:
            predictions.append(" ".join(piece[0] for piece in result))
        references = [s.lower() for s in references]
        print("beam search WER:", wer(references, predictions))
        # print("greedy WER:", wer(references, all_greedy_predictions))

def write_ctm_output(key, result):
    for word, start, end in result:
        print(f"{key} 1 {start:.2f} {end - start:.2f} {word} 1.0")

        # manifest = "/mnt/disks/sda_hdd/librispeech/dev_clean.json"
        # paths = []
        # with open(manifest) as fh:
        #     for line in fh:
        #         entry = json.loads(line)
        #         paths.append(entry["audio_filepath"])

        # for path in paths:
        #     logprobs = asr_model.transcribe([path], batch_size=1, logprobs=True)
        #     sequences = [torch.from_numpy(logprobs[0]).cuda()]
        #     sequence_lengths = [logprobs[0].shape[0]]
        #     padded_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        #     sequence_lengths_tensor = torch.tensor(sequence_lengths, dtype=torch.long)
        #     for result in decoder.decode(padded_sequence,
        #                                  sequence_lengths_tensor):
        #         print(result)

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
