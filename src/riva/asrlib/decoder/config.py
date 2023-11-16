import glob
import gzip
import multiprocessing
import os
import shutil
import subprocess


import nemo.collections.asr as nemo_asr

import riva.asrlib.decoder
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCudaConfig


def create_default_offline_decoder_config(*, batch_size, frame_shift_seconds):
    """Creates a configuration with reasonable
    defaults. frame_shift_seconds can be set to 0.01 if you're not
    sure what to pick. It won't affect accuracy, just timestamps. This
    corresponds to a 10ms frame shift.

    In the offline case, we can assume that max_batch_size and
    num_channels is the same.
    """
    config = BatchedMappedDecoderCudaConfig()
    config.n_input_per_chunk = 50
    config.online_opts.decoder_opts.default_beam = 17.0
    config.online_opts.decoder_opts.lattice_beam = 8.0
    config.online_opts.decoder_opts.max_active = 10_000
    config.online_opts.decoder_opts.ntokens_pre_allocated = 10_000_000
    config.online_opts.determinize_lattice = True
    config.online_opts.max_batch_size = batch_size
    config.online_opts.num_channels = batch_size
    config.online_opts.frame_shift_seconds = frame_shift_seconds
    config.online_opts.lattice_postprocessor_opts.acoustic_scale = 1.0
    config.online_opts.lattice_postprocessor_opts.lm_scale = 1.0
    config.online_opts.lattice_postprocessor_opts.word_ins_penalty = 0.0
    config.online_opts.lattice_postprocessor_opts.nbest = 1
    config.online_opts.num_decoder_copy_threads = 2
    config.online_opts.num_post_processing_worker_threads = (
        multiprocessing.cpu_count() - config.online_opts.num_decoder_copy_threads
    )
    return config

def create_TLG_from_nemo(topo, work_dir, nemo_model_name, gzipped_arpa_lm_path):
    arpa_base_name = os.path.basename(gzipped_arpa_lm_path).split(".arpa.gz")[0]
    graph_dir = os.path.join(work_dir, "graph", f"graph_ctc_{topo}_{arpa_base_name}")

    # If TLG was already created, skip this process.
    if len(glob.glob(os.path.join(graph_dir, "TLG.fst"))) != 0:
        return

    os.makedirs(work_dir, exist_ok=True)
    asr_model_config = nemo_asr.models.ASRModel.from_pretrained(nemo_model_name, return_config=True)
    units_txt = os.path.join(work_dir, "units.txt")
    with open(os.path.join(work_dir, "units.txt"), "w") as fh:
        for unit in asr_model_config["decoder"]["vocabulary"]:
            fh.write(f"{unit}\n")

    arpa_lm_name = os.path.splitext(os.path.basename(gzipped_arpa_lm_path))[0]

    with gzip.open(gzipped_arpa_lm_path, 'rb') as fh_in, \
         open(arpa_lm_name, 'wb') as fh_out:
        shutil.copyfileobj(fh_in, fh_out)

    (path,) = riva.asrlib.decoder.__path__

    words_path = os.path.join(work_dir, "words.txt")
    temp_words_path = os.path.join(work_dir, "words_with_ids.txt")
    subprocess.check_call(
        [
            os.path.join(path, "scripts/prepare_TLG_fst/bin/arpa2fst"),
            f"--write-symbol-table={temp_words_path}",
            gzipped_arpa_lm_path,
            "/dev/null",
        ]
    )

    with open(temp_words_path, "r") as words_with_ids_fh, \
         open(words_path, "w") as words_fh:
        for word_with_id in words_with_ids_fh:
            word = word_with_id.split()[0].lower()
            if word in {"<eps>", "<s>", "</s>", "<unk>"}:
                continue
            words_fh.write(word)
            words_fh.write("\n")

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
            words_path,
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
            gzipped_arpa_lm_path,
            dest_dir,
        ]
    )

    return os.path.join(graph_dir, "TLG.fst"), os.path.join(graph_dir, "words.txt")
