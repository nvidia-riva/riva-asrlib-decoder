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
#include "riva/asrlib/decoder/batched-mapped-online-decoder-cuda.h"
// kaldi
#include <itf/options-itf.h>

namespace riva::asrlib {

struct BatchedMappedDecoderCudaConfig {
  BatchedMappedDecoderCudaConfig() {}
  BatchedMappedOnlineDecoderCudaConfig online_opts;
  std::int32_t n_input_per_chunk = 50;

  void Register(kaldi::OptionsItf *po) {
    po->Register("n-input-per-chunk", &n_input_per_chunk,
                 "Maximum number of log likelihood frames to process in a single chunk.");
    online_opts.Register(po);
  }
};

class BatchedMappedDecoderCuda {
 public:
  using CorrelationID = BatchedMappedOnlineDecoderCuda::CorrelationID;
  BatchedMappedDecoderCuda(
      const BatchedMappedDecoderCudaConfig &config,
      const fst::Fst<fst::StdArc> &decode_fst,
      std::unique_ptr<kaldi::TransitionInformation> &&trans_information,
      fst::SymbolTable word_syms);
  ~BatchedMappedDecoderCuda();
  // std::future/promise
  void DecodeWithCallback(
      const float *d_logits,
      std::size_t logits_frame_stride,
      std::size_t logits_n_input_frames_valid,
      // this should not be copied by reference. It should be passed as an
      // lvalue or an rvalue because it is always copied.
      const std::function<void(std::tuple<std::optional<kaldi::CompactLattice>,
                               std::optional<kaldi::cuda_decoder::CTMResult>> &)> &callback);

  void ComputeTasks();
  void AcquireTasks();
  void BuildBatchFromCurrentTasks();
  void WaitForAllTasks();

  const fst::SymbolTable& GetSymbolTable() const;

 private:
  BatchedMappedOnlineDecoderCuda cuda_online_pipeline_;
  std::atomic<bool> threads_running_;
  std::atomic<int> n_tasks_not_done_;
  std::thread online_pipeline_control_thread_;
  struct UtteranceTask; // Forward declaration
  std::atomic<CorrelationID> corr_id_cnt_;
  std::mutex outstanding_utt_m_;
  std::queue<UtteranceTask> outstanding_utt_q_;

  std::mutex current_tasks_m_;
  std::vector<UtteranceTask> current_tasks_;
  std::vector<UtteranceTask> tasks_last_chunk_;
  std::int32_t max_batch_size_;
  std::int32_t n_input_per_chunk_;

  // built by BuildBatchFromCurrentTasks()
  // consumed by cuda_online_pipeline_.DecodeBatch
  std::vector<CorrelationID> batch_corr_ids_;
  std::vector<const float *> batch_d_logits_;
  std::vector<std::size_t> batch_logits_frame_stride_;
  std::vector<std::size_t> batch_n_input_frames_valid_;
  std::vector<bool> batch_is_last_chunk_;
  std::vector<bool> batch_is_first_chunk_;

  struct UtteranceTask {
    UtteranceTask &operator=(const UtteranceTask &) = delete;
    UtteranceTask(const UtteranceTask &) = delete;
    UtteranceTask(UtteranceTask &&) = default;
    UtteranceTask &operator=(UtteranceTask &&) = default;
    UtteranceTask() = default;
    // this can't really own just a chunk of an allocated matrix,
    // hmmm....
    const float *d_logits;
    std::size_t logits_frame_stride;
    std::size_t logits_n_input_frames_valid;
    CorrelationID corr_id;
    std::function<void(std::tuple<std::optional<kaldi::CompactLattice>,
                       std::optional<kaldi::cuda_decoder::CTMResult>> &)> callback;

    std::size_t loglikes_time_offset = std::size_t(0);
  };
    
};

}  // namespace riva::asrlib
