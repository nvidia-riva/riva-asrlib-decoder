/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <torch/extension.h>

#include "riva/asrlib/decoder/batched-mapped-decoder-cuda.h"
#include "riva/asrlib/decoder/ctc_transition_information.h"

namespace py = pybind11;

using namespace pybind11::literals;
using namespace torch::indexing;

namespace {

void PybindOnlineEndpointRule(py::module &m) {
  using PyClass = kaldi::OnlineEndpointRule;
  py::class_<PyClass> pyclass(m, "OnlineEndpointRule");
  pyclass.def(py::init<bool, float, float, float>(),
              "must_contain_nonsilence"_a = true,
              "min_trailing_silence"_a = 1.0,
              "max_relative_cost"_a = std::numeric_limits<float>::infinity(),
              "min_utterance_length"_a = 0.0);
  pyclass.def_readwrite("must_contain_nonsilence",
                        &PyClass::must_contain_nonsilence);
  pyclass.def_readwrite("min_trailing_silence", &PyClass::min_trailing_silence);
  pyclass.def_readwrite("max_relative_cost", &PyClass::max_relative_cost);
  pyclass.def_readwrite("min_utterance_length", &PyClass::min_utterance_length);
}

void PybindOnlineEndpointConfig(py::module &m) {
  using PyClass = kaldi::OnlineEndpointConfig;
  py::class_<PyClass> pyclass(m, "OnlineEndpointConfig");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("silence_phones", &PyClass::silence_phones);
  pyclass.def_readwrite("rule1", &PyClass::rule1);
  pyclass.def_readwrite("rule2", &PyClass::rule2);
  pyclass.def_readwrite("rule3", &PyClass::rule3);
  pyclass.def_readwrite("rule4", &PyClass::rule4);
  pyclass.def_readwrite("rule5", &PyClass::rule5);
}

void PybindCudaDecoderConfig(py::module &m) {
  using PyClass = kaldi::cuda_decoder::CudaDecoderConfig;
  py::class_<PyClass> pyclass(m, "CudaDecoderConfig");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("default_beam", &PyClass::default_beam);
  pyclass.def_readwrite("lattice_beam", &PyClass::lattice_beam);
  pyclass.def_readwrite("ntokens_pre_allocated",
                        &PyClass::ntokens_pre_allocated);
  pyclass.def_readwrite("main_q_capacity", &PyClass::main_q_capacity);
  pyclass.def_readwrite("aux_q_capacity", &PyClass::aux_q_capacity);
  pyclass.def_readwrite("max_active", &PyClass::max_active);
  pyclass.def_readwrite("endpointing_config", &PyClass::endpointing_config);
}

void PybindDeterminizeLatticePhonePrunedOptions(py::module &m) {
  using PyClass = fst::DeterminizeLatticePhonePrunedOptions;
  py::class_<PyClass> pyclass(m, "DeterminizeLatticePhonePrunedOptions");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("delta", &PyClass::delta);
  pyclass.def_readwrite("max_mem", &PyClass::max_mem);
  pyclass.def_readwrite("phone_determinize", &PyClass::phone_determinize);
  pyclass.def_readwrite("word_determinize", &PyClass::word_determinize);
  pyclass.def_readwrite("minimize", &PyClass::minimize);
}

void PybindMinimumBayesRiskOptions(py::module &m) {
  using PyClass = kaldi::MinimumBayesRiskOptions;
  py::class_<PyClass> pyclass(m, "MinimumBayesRiskOptions");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("decode_mbr", &PyClass::decode_mbr);
  pyclass.def_readwrite("print_silence", &PyClass::print_silence);
}

void PybindWordBoundaryInfoNewOpts(py::module &m) {
  using PyClass = kaldi::WordBoundaryInfoNewOpts;
  py::class_<PyClass> pyclass(m, "WordBoundaryInfoNewOpts");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("silence_label", &PyClass::silence_label);
  pyclass.def_readwrite("partial_word_label", &PyClass::partial_word_label);
  pyclass.def_readwrite("reorder", &PyClass::reorder);
}

void PybindLatticePostprocessorConfig(py::module &m) {
  using PyClass = kaldi::cuda_decoder::LatticePostprocessorConfig;
  py::class_<PyClass> pyclass(m, "LatticePostprocessorConfig");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("word_boundary_rxfilename",
                        &PyClass::word_boundary_rxfilename);
  pyclass.def_readwrite("mbr_opts", &PyClass::mbr_opts);
  pyclass.def_readwrite("wip_opts", &PyClass::wip_opts);
  pyclass.def_readwrite("max_expand", &PyClass::max_expand);
  pyclass.def_readwrite("acoustic_scale", &PyClass::acoustic_scale);
  pyclass.def_readwrite("lm_scale", &PyClass::lm_scale);
  pyclass.def_readwrite("acoustic2lm_scale", &PyClass::acoustic2lm_scale);
  pyclass.def_readwrite("lm2acoustic_scale", &PyClass::lm2acoustic_scale);
  pyclass.def_readwrite("word_ins_penalty", &PyClass::word_ins_penalty);
}

void PybindBatchedMappedOnlineDecoderCudaConfig(py::module &m) {
  using PyClass = riva::asrlib::BatchedMappedOnlineDecoderCudaConfig;
  py::class_<PyClass> pyclass(m, "BatchedMappedOnlineDecoderCudaConfig");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("max_batch_size", &PyClass::max_batch_size);
  pyclass.def_readwrite("num_channels", &PyClass::num_channels);
  pyclass.def_readwrite("num_post_processing_worker_threads",
                        &PyClass::num_post_processing_worker_threads);
  pyclass.def_readwrite("determinize_lattice", &PyClass::determinize_lattice);
  pyclass.def_readwrite("num_decoder_copy_threads",
                        &PyClass::num_decoder_copy_threads);
  pyclass.def_readwrite("frame_shift_seconds", &PyClass::frame_shift_seconds);
  pyclass.def_readwrite("decoder_opts", &PyClass::decoder_opts);
  pyclass.def_readwrite("det_opts", &PyClass::det_opts);
  pyclass.def_readwrite("lattice_postprocessor_opts",
                        &PyClass::lattice_postprocessor_opts);
}

void PybindBatchedMappedDecoderCudaConfig(py::module &m) {
  using PyClass = riva::asrlib::BatchedMappedDecoderCudaConfig;
  py::class_<PyClass> pyclass(m, "BatchedMappedDecoderCudaConfig");
  pyclass.def(py::init<>());
  pyclass.def_readwrite("online_opts", &PyClass::online_opts);
  pyclass.def_readwrite("n_input_per_chunk", &PyClass::n_input_per_chunk);
}

void PybindBatchedMappedDecoderCuda(py::module &m) {
  using PyClass = riva::asrlib::BatchedMappedDecoderCuda;
  py::class_<PyClass> pyclass(m, "BatchedMappedDecoderCuda");
  // Need to wrap fsts somehow, or make the user provide paths to them on disk.
  // Paths on disk might be a better start.
  // ot sure how pybind11 interacts with cython or whatever openfst uses...
  // pywrapfst
  // pyclass.def(py::init<const BatchedMappedDecoderCudaConfig&,
  //                      const fst::Fst<fst::StdArc>&,
  //                      std::unique_ptr<kaldi::TransitionInformation> &&>());
  pyclass.def(
      py::init([](const riva::asrlib::BatchedMappedDecoderCudaConfig &config,
                  const std::string &wfst_path_on_disk,
                  const std::string &symbol_table_path_on_disk,
                  int num_tokens_including_blank) {
        std::unique_ptr<kaldi::TransitionInformation> trans_info =
            std::make_unique<riva::asrlib::CTCTransitionInformation>(
                num_tokens_including_blank);
        std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst =
            std::unique_ptr<fst::Fst<fst::StdArc>>(
                fst::ReadFstKaldiGeneric(wfst_path_on_disk));

        auto word_syms = std::unique_ptr<fst::SymbolTable>(
            fst::SymbolTable::ReadText(symbol_table_path_on_disk));

        auto decoder =
            new PyClass(config, *decode_fst, std::move(trans_info), *word_syms);
        return decoder;
      }));
  pyclass.def(
      "decode",
      [](PyClass &cuda_pipeline, const torch::Tensor &logits,
         const torch::Tensor &logits_lengths)
          -> std::vector<std::vector<std::tuple<std::string, float, float>>> {
        // contiguousness might not mean what I think it means. It may just mean
        // stride has no padding.
        if (!logits.is_contiguous() ||
            logits.scalar_type() != torch::ScalarType::Float ||
            logits_lengths.scalar_type() != torch::ScalarType::Long) {
          throw std::invalid_argument("Invalid input tensors");
        }
        // logits should be batch x time x logits
        std::size_t batch_size = logits_lengths.size(0);
        std::vector<std::vector<std::tuple<std::string, float, float>>> results(
            batch_size);
        for (int64_t i = 0; i < logits_lengths.size(0); ++i) {
          // TODO: Check that the logits_lengths tensor actually contains long
          // values
          std::size_t valid_time_steps =
              logits_lengths.index({TensorIndex(i)}).item<long>();
          torch::Tensor single_sample_logits =
              logits.index({i, Slice(None, valid_time_steps), "..."});
          // number of rows is number of frames
          // number of cols is number of logits
          // stride of each row is stride. Always greater than number of cols
          auto place_results =
              [i, &results, &word_syms = cuda_pipeline.GetSymbolTable()](
                  std::tuple<std::optional<kaldi::CompactLattice>,
                             std::optional<kaldi::cuda_decoder::CTMResult>>
                      &asr_results) {
                const kaldi::cuda_decoder::CTMResult &ctm_result =
                    std::get<1>(asr_results).value();
                for (size_t iword = 0; iword < ctm_result.times_seconds.size();
                     ++iword) {
                  results[i].emplace_back(
                      word_syms.Find(ctm_result.words[iword]),
                      ctm_result.times_seconds[iword].first,
                      ctm_result.times_seconds[iword].second);
                }
              };
          cuda_pipeline.DecodeWithCallback(
              single_sample_logits.data_ptr<float>(),
              single_sample_logits.stride(1), single_sample_logits.size(0),
              place_results);
        }
        cuda_pipeline.WaitForAllTasks();
        return results;
      });


  // TODO: Overload decode to accept a DLPack Tensor.
  // pyclass.def(
  //     "decode",
  //     [](PyClass &cuda_pipeline, const DLManagedTensor &logits,
  //        const torch::Tensor &logits_lengths)
  //         -> std::vector<std::vector<std::tuple<std::string, float, float>>> {
  //       // contiguousness might not mean what I think it means. It may just mean
  //       // stride has no padding.
  //         assert(logits.dl_tensor.ndim == 3);
  //         assert(logits.dl_tensor.dtype == kDLFloat);
  //       if (!logits.is_contiguous() ||
  //           logits.scalar_type() != torch::ScalarType::Float ||
  //           logits_lengths.scalar_type() != torch::ScalarType::Long) {
  //         throw std::invalid_argument("Invalid input tensors");
  //       }
  //       // logits should be batch x time x logits
  //       std::size_t batch_size = logits_lengths.size(0);
  //       std::vector<std::vector<std::tuple<std::string, float, float>>> results(
  //           batch_size);
  //       for (int64_t i = 0; i < logits_lengths.size(0); ++i) {
  //         // TODO: Check that the logits_lengths tensor actually contains long
  //         // values
  //         std::size_t valid_time_steps =
  //             logits_lengths.index({TensorIndex(i)}).item<long>();
  //         torch::Tensor single_sample_logits =
  //             logits.index({i, Slice(None, valid_time_steps), "..."});
  //         // number of rows is number of frames
  //         // number of cols is number of logits
  //         // stride of each row is stride. Always greater than number of cols
  //         auto place_results =
  //             [i, &results, &word_syms = cuda_pipeline.GetSymbolTable()](
  //                 std::tuple<std::optional<kaldi::CompactLattice>,
  //                            std::optional<kaldi::cuda_decoder::CTMResult>>
  //                     &asr_results) {
  //               const kaldi::cuda_decoder::CTMResult &ctm_result =
  //                   std::get<1>(asr_results).value();
  //               for (size_t iword = 0; iword < ctm_result.times_seconds.size();
  //                    ++iword) {
  //                 results[i].emplace_back(
  //                     word_syms.Find(ctm_result.words[iword]),
  //                     ctm_result.times_seconds[iword].first,
  //                     ctm_result.times_seconds[iword].second);
  //               }
  //             };
  //         cuda_pipeline.DecodeWithCallback(
  //             single_sample_logits.data_ptr<float>(),
  //             single_sample_logits.stride(1), single_sample_logits.size(0),
  //             place_results);
  //       }
  //       cuda_pipeline.WaitForAllTasks();
  //       return results;
  //     });

}
} // anonymous namespace

PYBIND11_MODULE(python_decoder, m) {
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
