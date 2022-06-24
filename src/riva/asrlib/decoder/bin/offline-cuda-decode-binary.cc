/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "riva/asrlib/decoder/batched-mapped-decoder-cuda.h"
#include "riva/asrlib/decoder/ctc_transition_information.h"

namespace riva::asrlib {
void WriteCTMOutput(std::ostream &ostream, const std::string &key,
                    const kaldi::cuda_decoder::CTMResult &ctm,
                    const fst::SymbolTable &word_syms) {
  ostream << std::fixed;
  ostream.precision(2);
  for (size_t iword = 0; iword < ctm.times_seconds.size(); ++iword) {
    ostream << key << " 1 " << ctm.times_seconds[iword].first << ' '
            << (ctm.times_seconds[iword].second -
                ctm.times_seconds[iword].first)
            << ' ';
    int32_t word_id = ctm.words[iword];
    ostream << word_syms.Find(word_id);
    ostream << ' ' << ctm.conf[iword] << '\n';
  }
}
} // namespace riva::asrlib

int main(int argc, char **argv) {
  using namespace kaldi;
  using namespace riva::asrlib;

  const char *usage =
      "Reads in wav file(s) and decodes them with "
      "neural nets\n"
      "(nnet3 setup).  Note: some configuration values "
      "and inputs "
      "are\n"
      "set via config files whose filenames are passed as "
      "options\n"
      "Output can either be a lattice wspecifier or a ctm filename"
      "\n"
      "Usage: batched-wav-nnet3-cuda2 [options] <trans-in> "
      "<fst-in> "
      "<wav-rspecifier> lattice-wspecifier ctm-wxdirname\n";

  std::string word_syms_rxfilename;

  std::string lattice_postprocessor_config_rxfilename;
  kaldi::ParseOptions po(usage);
  po.Register("word-symbol-table", &word_syms_rxfilename,
              "Symbol table for words [for debug output and ctm output]");
  // po.Register("lattice-postprocessor-rxfilename",
  //             &lattice_postprocessor_config_rxfilename,
  //             "(optional) Config file for lattice postprocessor");

  // Multi-threaded CPU and batched GPU decoder
  BatchedMappedDecoderCudaConfig batched_decoder_config;
  batched_decoder_config.Register(&po);

  po.Read(argc, argv);

  if (po.NumArgs() != 5) {
    po.PrintUsage();
    return 1;
  }

  // initialize cuda
  CU_SAFE_CALL(cudaFree(0));
  std::string trans_info_rxfilename = po.GetArg(1),
              fst_rxfilename = po.GetArg(2), logits_rspecifier = po.GetArg(3),
              output_wspecifier = po.GetArg(4), ctm_output_path = po.GetArg(5);
  fst::StdFst *fst_ptr = fst::ReadFstKaldiGeneric(trans_info_rxfilename);
  std::unique_ptr<kaldi::TransitionInformation> trans_info =
      std::make_unique<CTCTransitionInformation>(*fst_ptr);
  delete fst_ptr;

  fst::Fst<fst::StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_rxfilename);

  std::unique_ptr<fst::SymbolTable> word_syms = nullptr;
  if (word_syms_rxfilename != "") {
    if (!(word_syms = std::unique_ptr<fst::SymbolTable>(
              fst::SymbolTable::ReadText(word_syms_rxfilename))))
      assert(0 && "Could not read symbol table from file ");
  }

  BatchedMappedDecoderCuda cuda_pipeline(batched_decoder_config, *decode_fst,
                                         std::move(trans_info), *word_syms);

  // TODO: Is this required?
  delete decode_fst;

  kaldi::CompactLatticeWriter clat_writer(output_wspecifier);
  std::unique_ptr<Output> ctm_writer;

  std::size_t total_frames = 0;

  std::mutex output_writer_m;

  SequentialBaseFloatMatrixReader loglikes_reader(logits_rspecifier);
  Timer timer;
  for (; !loglikes_reader.Done(); loglikes_reader.Next()) {
    std::string utt = loglikes_reader.Key();
    std::string key = utt;
    const kaldi::Matrix<float> &loglikes = loglikes_reader.Value();
    // must free this
    float *d_loglikes;
    std::size_t pitch;
    CU_SAFE_CALL(cudaMallocPitch(&d_loglikes, &pitch,
                                 loglikes.NumCols() * sizeof(float),
                                 loglikes.NumRows()));
    // no need for asynchronous copy, as far as I know...
    CU_SAFE_CALL(cudaMemcpy2D(d_loglikes, pitch, loglikes.Data(),
                              loglikes.Stride() * sizeof(float),
                              loglikes.NumCols() * sizeof(float),
                              loglikes.NumRows(), cudaMemcpyHostToDevice));

    kaldi::Matrix<float> loglikes_copy(loglikes);
    loglikes_copy.SetZero();
    CU_SAFE_CALL(cudaMemcpy2D(
        loglikes_copy.Data(), loglikes_copy.Stride() * sizeof(float),
        d_loglikes, pitch, loglikes_copy.NumCols() * sizeof(float),
        loglikes_copy.NumRows(), cudaMemcpyDeviceToHost));

    assert(loglikes_copy.Equal(loglikes));

    // calculating number of utterances per
    // iteration calculating total audio
    // time per iteration
    total_frames += loglikes.NumRows();

    // Callback used when results are ready
    //
    // If lattice output, write all lattices to clat_writer
    // If segmentation is true, then the keys are:
    // [utt_key]-[segment_offset]
    //
    // If CTM output, merging segment results together
    // and writing this single output to ctm_writer

    // must copy the captured variables. Otherwise, they will be lost.
    auto callback =
        [key, d_loglikes, &output_writer_m, &clat_writer, &ctm_output_path,
         &word_syms = cuda_pipeline.GetSymbolTable()](
            std::tuple<std::optional<kaldi::CompactLattice>,
                       std::optional<kaldi::cuda_decoder::CTMResult>> &result) {
          CU_SAFE_CALL(cudaFree(d_loglikes));

          if (std::get<0>(result).has_value()) {
            std::lock_guard<std::mutex> lk(output_writer_m);
            clat_writer.Write(key, std::get<0>(result).value());
          }

          if (std::get<1>(result).has_value()) {
            Output ctm_output(ctm_output_path + "/" + key + ".ctm",
                              /*binary=*/false);
            WriteCTMOutput(ctm_output.Stream(), key,
                           std::get<1>(result).value(), word_syms);
          }
        };

    assert(pitch % sizeof(float) == 0);
    cuda_pipeline.DecodeWithCallback(d_loglikes, pitch / sizeof(float),
                                     loglikes.NumRows(), callback);
  } // end utterance loop

  cuda_pipeline.WaitForAllTasks();
  double total_compute_time = timer.Elapsed();

  double total_input_time = total_frames * (80.0 / 1000.0);

  KALDI_LOG << "input_time: " << total_input_time
            << ", compute_time: " << total_compute_time
            << ", RTFx:" << total_input_time / total_compute_time;
}
