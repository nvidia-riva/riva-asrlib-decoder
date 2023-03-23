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
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "riva/asrlib/decoder/batched-mapped-decoder-cuda.h"
#include "riva/asrlib/decoder/ctc_transition_information.h"

namespace nb = nanobind;

using namespace nb::literals;

namespace {

void
PybindOnlineEndpointRule(nb::module_& m)
{
  using PyClass = kaldi::OnlineEndpointRule;
  nb::class_<PyClass> pyclass(m, "OnlineEndpointRule");
  pyclass.def(
      nb::init<bool, float, float, float>(), "must_contain_nonsilence"_a = true,
      "min_trailing_silence"_a = 1.0,
      "max_relative_cost"_a = std::numeric_limits<float>::infinity(),
      "min_utterance_length"_a = 0.0);
  pyclass.def_rw("must_contain_nonsilence", &PyClass::must_contain_nonsilence);
  pyclass.def_rw("min_trailing_silence", &PyClass::min_trailing_silence);
  pyclass.def_rw("max_relative_cost", &PyClass::max_relative_cost);
  pyclass.def_rw("min_utterance_length", &PyClass::min_utterance_length);
}

void
PybindOnlineEndpointConfig(nb::module_& m)
{
  using PyClass = kaldi::OnlineEndpointConfig;
  nb::class_<PyClass> pyclass(m, "OnlineEndpointConfig");
  pyclass.def(nb::init<>());
  pyclass.def_rw("silence_phones", &PyClass::silence_phones);
  pyclass.def_rw("rule1", &PyClass::rule1);
  pyclass.def_rw("rule2", &PyClass::rule2);
  pyclass.def_rw("rule3", &PyClass::rule3);
  pyclass.def_rw("rule4", &PyClass::rule4);
  pyclass.def_rw("rule5", &PyClass::rule5);
}

void
PybindCudaDecoderConfig(nb::module_& m)
{
  using PyClass = kaldi::cuda_decoder::CudaDecoderConfig;
  nb::class_<PyClass> pyclass(m, "CudaDecoderConfig");
  pyclass.def(nb::init<>());
  pyclass.def_rw("default_beam", &PyClass::default_beam);
  pyclass.def_rw("lattice_beam", &PyClass::lattice_beam);
  pyclass.def_rw("ntokens_pre_allocated", &PyClass::ntokens_pre_allocated);
  pyclass.def_rw("main_q_capacity", &PyClass::main_q_capacity);
  pyclass.def_rw("aux_q_capacity", &PyClass::aux_q_capacity);
  pyclass.def_rw("max_active", &PyClass::max_active);
  pyclass.def_rw("endpointing_config", &PyClass::endpointing_config);
  pyclass.def_rw("blank_penalty", &PyClass::blank_penalty);
  pyclass.def_rw("blank_ilabel", &PyClass::blank_ilabel);
  pyclass.def_rw("length_penalty", &PyClass::length_penalty);
}

void
PybindDeterminizeLatticePhonePrunedOptions(nb::module_& m)
{
  using PyClass = fst::DeterminizeLatticePhonePrunedOptions;
  nb::class_<PyClass> pyclass(m, "DeterminizeLatticePhonePrunedOptions");
  pyclass.def(nb::init<>());
  pyclass.def_rw("delta", &PyClass::delta);
  pyclass.def_rw("max_mem", &PyClass::max_mem);
  pyclass.def_rw("phone_determinize", &PyClass::phone_determinize);
  pyclass.def_rw("word_determinize", &PyClass::word_determinize);
  pyclass.def_rw("minimize", &PyClass::minimize);
}

void
PybindMinimumBayesRiskOptions(nb::module_& m)
{
  using PyClass = kaldi::MinimumBayesRiskOptions;
  nb::class_<PyClass> pyclass(m, "MinimumBayesRiskOptions");
  pyclass.def(nb::init<>());
  pyclass.def_rw("decode_mbr", &PyClass::decode_mbr);
  pyclass.def_rw("print_silence", &PyClass::print_silence);
}

void
PybindWordBoundaryInfoNewOpts(nb::module_& m)
{
  using PyClass = kaldi::WordBoundaryInfoNewOpts;
  nb::class_<PyClass> pyclass(m, "WordBoundaryInfoNewOpts");
  pyclass.def(nb::init<>());
  pyclass.def_rw("silence_label", &PyClass::silence_label);
  pyclass.def_rw("partial_word_label", &PyClass::partial_word_label);
  pyclass.def_rw("reorder", &PyClass::reorder);
}

void
PybindLatticePostprocessorConfig(nb::module_& m)
{
  using PyClass = kaldi::cuda_decoder::LatticePostprocessorConfig;
  nb::class_<PyClass> pyclass(m, "LatticePostprocessorConfig");
  pyclass.def(nb::init<>());
  pyclass.def_rw("word_boundary_rxfilename", &PyClass::word_boundary_rxfilename);
  pyclass.def_rw("mbr_opts", &PyClass::mbr_opts);
  pyclass.def_rw("wip_opts", &PyClass::wip_opts);
  pyclass.def_rw("max_expand", &PyClass::max_expand);
  pyclass.def_rw("acoustic_scale", &PyClass::acoustic_scale);
  pyclass.def_rw("lm_scale", &PyClass::lm_scale);
  pyclass.def_rw("acoustic2lm_scale", &PyClass::acoustic2lm_scale);
  pyclass.def_rw("lm2acoustic_scale", &PyClass::lm2acoustic_scale);
  pyclass.def_rw("word_ins_penalty", &PyClass::word_ins_penalty);
  pyclass.def_rw("nbest", &PyClass::nbest);
}

void
PybindBatchedMappedOnlineDecoderCudaConfig(nb::module_& m)
{
  using PyClass = riva::asrlib::BatchedMappedOnlineDecoderCudaConfig;
  nb::class_<PyClass> pyclass(m, "BatchedMappedOnlineDecoderCudaConfig");
  pyclass.def(nb::init<>());
  pyclass.def_rw("max_batch_size", &PyClass::max_batch_size);
  pyclass.def_rw("num_channels", &PyClass::num_channels);
  pyclass.def_rw(
      "num_post_processing_worker_threads", &PyClass::num_post_processing_worker_threads);
  pyclass.def_rw("determinize_lattice", &PyClass::determinize_lattice);
  pyclass.def_rw("num_decoder_copy_threads", &PyClass::num_decoder_copy_threads);
  pyclass.def_rw("frame_shift_seconds", &PyClass::frame_shift_seconds);
  pyclass.def_rw("decoder_opts", &PyClass::decoder_opts);
  pyclass.def_rw("det_opts", &PyClass::det_opts);
  pyclass.def_rw("lattice_postprocessor_opts", &PyClass::lattice_postprocessor_opts);
  pyclass.def_rw("use_lattice_postprocessor", &PyClass::use_lattice_postprocessor);
}

void
PybindBatchedMappedDecoderCudaConfig(nb::module_& m)
{
  using PyClass = riva::asrlib::BatchedMappedDecoderCudaConfig;
  nb::class_<PyClass> pyclass(m, "BatchedMappedDecoderCudaConfig");
  pyclass.def(nb::init<>());
  pyclass.def_rw("online_opts", &PyClass::online_opts);
  pyclass.def_rw("n_input_per_chunk", &PyClass::n_input_per_chunk);
}

void
PybindBatchedMappedDecoderCuda(nb::module_& m)
{
  using PyClass = riva::asrlib::BatchedMappedDecoderCuda;
  nb::class_<PyClass> pyclass(m, "BatchedMappedDecoderCuda");
  // Need to wrap fsts somehow, or make the user provide paths to them on disk.
  // Paths on disk might be a better start.
  // ot sure how pybind11 interacts with cython or whatever openfst uses...
  // pywrapfst
  // pyclass.def(nb::init<const BatchedMappedDecoderCudaConfig&,
  //                      const fst::Fst<fst::StdArc>&,
  //                      std::unique_ptr<kaldi::TransitionInformation> &&>());
  pyclass.def("__init__", [](PyClass* decoder,
                          const riva::asrlib::BatchedMappedDecoderCudaConfig& config,
                          const std::string& wfst_path_on_disk,
                          const std::string& symbol_table_path_on_disk,
                          int num_tokens_including_blank) {
    std::unique_ptr<kaldi::TransitionInformation> trans_info =
        std::make_unique<riva::asrlib::CTCTransitionInformation>(num_tokens_including_blank);
    std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst =
        std::unique_ptr<fst::Fst<fst::StdArc>>(fst::ReadFstKaldiGeneric(wfst_path_on_disk));

    auto word_syms =
        std::unique_ptr<fst::SymbolTable>(fst::SymbolTable::ReadText(symbol_table_path_on_disk));

    new (decoder) PyClass(config, *decode_fst, std::move(trans_info), *word_syms);
                          });

  using LogitsArray = nb::ndarray<float, nb::shape<nb::any, nb::any, nb::any>, nb::c_contig, nb::device::cuda>;
  using LogitsLengthsArray = nb::ndarray<long, nb::shape<nb::any>, nb::c_contig, nb::device::cpu>;

  pyclass.def(
              "decode_write_lattice",
              [](PyClass& cuda_pipeline, LogitsArray& logits,
                 LogitsLengthsArray& logits_lengths,
                 const std::vector<std::string>& keys,
                 const std::string& output_wspecifier) {
        int64_t batch_size = logits_lengths.shape(0);

        kaldi::CompactLatticeWriter clat_writer(output_wspecifier);
        for (int64_t i = 0; i < batch_size; ++i) {
          int64_t valid_time_steps = logits_lengths(i);

          const float* single_sample_logits_start = &logits(i, 0, 0);
          // number of rows is number of frames
          // number of cols is number of logits
          // stride of each row is stride. Always greater than number of cols
          auto write_results =
              [i, &clat_writer, &keys](
                  riva::asrlib::BatchedMappedOnlineDecoderCuda::ReturnType& asr_results) {
              const kaldi::CompactLattice& lattice = std::get<0>(asr_results).value();
              clat_writer.Write(keys[i], lattice);
              };
          cuda_pipeline.DecodeWithCallback(
              single_sample_logits_start,
              logits.stride(1),
              valid_time_steps,
              write_results);
        }
        cuda_pipeline.WaitForAllTasks();
      });


  pyclass.def("decode_mbr",
      [](PyClass& cuda_pipeline, LogitsArray& logits,
         LogitsLengthsArray& logits_lengths)
          -> std::vector<std::vector<std::tuple<std::string, float, float, float>>> {
        int64_t batch_size = logits_lengths.shape(0);
        std::vector<std::vector<std::tuple<std::string, float, float, float>>> results(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
          int64_t valid_time_steps = logits_lengths(i);
          const float* single_sample_logits_start = &logits(i, 0, 0);
          auto place_results =
              [i, &results, &word_syms = cuda_pipeline.GetSymbolTable()](
                  riva::asrlib::BatchedMappedOnlineDecoderCuda::ReturnType& asr_results) {
                const kaldi::cuda_decoder::CTMResult& ctm_result = std::get<1>(asr_results).value();
                for (size_t iword = 0; iword < ctm_result.times_seconds.size(); ++iword) {
                  results[i].emplace_back(
                      word_syms.Find(ctm_result.words[iword]),
                      ctm_result.times_seconds[iword].first,
                      ctm_result.times_seconds[iword].second,
                      ctm_result.conf[iword]);
                }
              };
          cuda_pipeline.DecodeWithCallback(
              single_sample_logits_start,
              logits.stride(1),
              valid_time_steps,
              place_results);
        }
        cuda_pipeline.WaitForAllTasks();
        return results;
      });

  pyclass.def("decode_map",
              [](PyClass& cuda_pipeline, LogitsArray& logits,
                 LogitsLengthsArray& logits_lengths)
              -> std::vector<std::vector<std::tuple<float, std::vector<std::tuple<std::string, float, float>>>>>
              {
        int64_t batch_size = logits_lengths.shape(0);
        // batch, nbest result, words with times
        std::vector<std::vector<std::tuple<float, std::vector<std::tuple<std::string, float, float>>>>> results(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
          int64_t valid_time_steps = logits_lengths(i);

          // this may not be right... Yes, it seems quite wrong...
          const float* single_sample_logits_start = &logits(i, 0, 0);
          // number of rows is number of frames
          // number of cols is number of logits
          // stride of each row is stride. Always greater than number of cols
          auto place_results =
              [i, &results, &word_syms = cuda_pipeline.GetSymbolTable()](
                  riva::asrlib::BatchedMappedOnlineDecoderCuda::ReturnType& asr_results) {
                const std::vector<kaldi::cuda_decoder::NBestResult>& nbest_results = std::get<2>(asr_results).value();
                // this type doesn't match results above
                std::vector<
                std::tuple<float, // score
                std::vector<std::tuple<std::string, float, float>>
                >> result_this_utt;
                for (const kaldi::cuda_decoder::NBestResult& nbest_result: nbest_results) {
                  std::vector<std::tuple<std::string, float, float>> words; words.reserve(nbest_result.words.size());
                  std::size_t i = 0;
                  for (auto&& word_id: nbest_result.words) {
                    words.emplace_back(word_syms.Find(word_id),
                                       nbest_result.times_seconds[i].first,
                                       nbest_result.times_seconds[i].second);
                    ++i;
                  }
                  result_this_utt.emplace_back(nbest_result.score, words);
                }

                results[i] = std::move(result_this_utt);
          };
          cuda_pipeline.DecodeWithCallback(
              single_sample_logits_start,
              logits.stride(1),
              valid_time_steps,
              place_results);
        }
        cuda_pipeline.WaitForAllTasks();
        return results;
              });
}
} // anonymous namespace

NB_MODULE(python_decoder, m)
{
  m.doc() = "pybind11 bindings for the CUDA WFST decoder";

  PybindOnlineEndpointRule(m);
  PybindOnlineEndpointConfig(m);
  PybindCudaDecoderConfig(m);
  PybindDeterminizeLatticePhonePrunedOptions(m);
  PybindMinimumBayesRiskOptions(m);
  PybindWordBoundaryInfoNewOpts(m);
  PybindLatticePostprocessorConfig(m);
  PybindBatchedMappedOnlineDecoderCudaConfig(m);
  PybindBatchedMappedDecoderCudaConfig(m);
  PybindBatchedMappedDecoderCuda(m);
}
