#!/usr/bin/env python3

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

import argparse
import os

from joblib import Parallel, delayed
from nemo.collections.asr.models import ASRModel
from nemo.collections.common import tokenizers
from nemo.utils import logging, model_utils
from tqdm.auto import tqdm


def write_dataset(chunks, path):
    basedir = os.path.dirname(path)

    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

    with open(path, 'a+', encoding='utf-8') as f:
        for chunk_idx in tqdm(range(len(chunks)), desc='Chunk ', total=len(chunks), unit=' chunks'):
            for text in chunks[chunk_idx]:
                line = ' '.join(text)
                f.write(f"{line}\n")


def read_train_file(path, lowercase: bool = False):
    lines_read = 0
    text_dataset = []

    with open(path, 'r') as f:
        reader = tqdm(iter(lambda: f.readline(), ''), desc="Read 0 lines", unit=' lines')
        for i, line in enumerate(reader):
            if path.endswith('.json'):
                line = json.loads(line)['text']

            line = line.replace("\n", "").strip()
            if lowercase:
                line = line.lower()

            if line:
                text_dataset.append(line)

                lines_read += 1
                if lines_read % 100000 == 0:
                    reader.set_description(f"Read {lines_read} lines")

    return text_dataset


def tokenize_str(texts, tokenizer):
    # we need to manually set tokens by their ids to unify the token style between wpe and spe
    return [[tokenizer.inv_vocabulary[i] for i in tokenizer.text_to_ids(text)] for text in texts]


def tokenize_text(data, tokenizer, path, chunk_size=8192, buffer_size=32):
    dataset_len = len(data)
    logging.info(
        f"Chunking {dataset_len} rows into {dataset_len / float(chunk_size):0.4f} tasks (each chunk contains {chunk_size} elements)"
    )

    current_step = 0
    with Parallel(n_jobs=-2, verbose=10) as parallel:
        while True:
            start = current_step * chunk_size
            end = min((current_step + buffer_size) * chunk_size, dataset_len)

            tokenized_data = parallel(
                delayed(tokenize_str)(data[start : start + chunk_size], tokenizer)
                for start in range(start, end, chunk_size)
            )

            # Write dataset
            write_dataset(tokenized_data, path)
            current_step += len(tokenized_data)
            logging.info(
                f"Finished writing {len(tokenized_data)} chunks to {path}. Current chunk index = {current_step}"
            )
            del tokenized_data
            if end >= dataset_len:
                break


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""Tokenize text into the corresponding subwords.""",
    )
    parser.add_argument(
        "--text_file",
        required=True,
        type=str,
        help="Path to the text file to tokenize, it can be a plain text file or JSON manifest",
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        help="The directory path to the tokenizer vocabulary + additional metadata",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["bpe", "wpe"],
        help="The type of the tokenizer. Currently supports `bpe` and `wpe`",
    )
    parser.add_argument(
        "--nemo_model",
        type=str,
        help="The .nemo model path to extract the tokenizer",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="The path to store the tokenized subwords",
    )
    parser.add_argument(
        "--do_lowercase", action='store_true', help="Whether to apply lower case conversion on the text"
    )
    args = parser.parse_args()

    """ TOKENIZER SETUP """
    if not hasattr(args, "nemo_model") and not (hasattr(args, "tokenizer_dir") or hasattr(args, "tokenizer_type")):
        raise AttributeError("Either --nemo_model or --tokenizer_dir and --tokenizer_type must be set.")

    if not hasattr(args, "nemo_model"):
        logging.info(f"Loading {args.tokenizer_type} tokenizer from '{args.tokenizer_dir}' ...")
        if args.tokenizer_type == 'bpe':
            # This is a BPE Tokenizer
            model_path = os.path.join(args.tokenizer_dir, 'tokenizer.model')

            # Update special tokens
            tokenizer = tokenizers.SentencePieceTokenizer(model_path=model_path)
        else:
            # This is a WPE Tokenizer
            vocab_path = os.path.join(args.tokenizer_dir, 'vocab.txt')
            tokenizer = tokenizers.AutoTokenizer(pretrained_model_name='bert-base-cased', vocab_file=vocab_path)
    else:
        if args.nemo_model.endswith(".nemo"):
            logging.info(f"Loading tokenizer from .nemo model '{args.nemo_model}' ...")
            model_cfg = ASRModel.restore_from(restore_path=args.nemo_model, return_config=True)
            classpath = model_cfg.target  # original class path
            imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
            asr_model = imported_class.restore_from(restore_path=args.nemo_model, map_location="cpu")  # type: ASRModel
        else:
            model_cfg = ASRModel.from_pretrained(args.nemo_model, return_config=True)
            classpath = model_cfg.target  # original class path
            imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
            asr_model = imported_class.from_pretrained(args.nemo_model, map_location="cpu")  # type: ASRModel
        tokenizer = asr_model.tokenizer

    inv_vocabulary = ['' for i in range(tokenizer.vocab_size)]
    for key, value in tokenizer.tokenizer.get_vocab().items():
        inv_vocabulary[value - 1] = '[UNK]' if key == '<unk>' else key
    setattr(tokenizer, "inv_vocabulary", inv_vocabulary)

    logging.info(f"Tokenizer {tokenizer.__class__.__name__} loaded with {tokenizer.vocab_size} tokens")

    """ DATA PROCESSING """
    logging.info(f"Encoding the text file '{args.text_file}' ...")
    dataset = read_train_file(args.text_file, lowercase=args.do_lowercase)

    if os.path.exists(args.output_file):
        logging.info(f"Deleting previous file : {args.output_file}")
        os.remove(args.output_file)

    tokenize_text(dataset, tokenizer, path=args.output_file)
    logging.info(f"The text file '{args.text_file}' encoded and saved to {args.output_file}")


if __name__ == '__main__':
    main()
