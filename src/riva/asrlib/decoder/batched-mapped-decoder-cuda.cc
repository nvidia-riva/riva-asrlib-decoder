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
#include "riva/asrlib/decoder/batched-mapped-decoder-cuda.h"

namespace riva::asrlib {

const float kSleepForTaskComplete = 10e-3;
const float kSleepForNewTask = 100e-6;

BatchedMappedDecoderCuda::BatchedMappedDecoderCuda(
    const BatchedMappedDecoderCudaConfig &config,
    const fst::Fst<fst::StdArc> &decode_fst,
    std::unique_ptr<kaldi::TransitionInformation> &&trans_information,
    fst::SymbolTable word_syms)
    : cuda_online_pipeline_(config.online_opts, decode_fst,
                            std::move(trans_information)),
      threads_running_(true), n_tasks_not_done_(0), corr_id_cnt_(0),
      max_batch_size_(config.online_opts.max_batch_size),
      n_input_per_chunk_(config.n_input_per_chunk) {
  assert(n_input_per_chunk_ > 0 && "n_input_per_chunk must be greater than 0");
  online_pipeline_control_thread_ =
      std::thread(&BatchedMappedDecoderCuda::ComputeTasks, this);
  cuda_online_pipeline_.SetSymbolTable(std::move(word_syms));
}

BatchedMappedDecoderCuda::~BatchedMappedDecoderCuda() {
  threads_running_.store(false);
  online_pipeline_control_thread_.join();
}

void BatchedMappedDecoderCuda::DecodeWithCallback(
    const float *d_logits, std::size_t logits_frame_stride,
    std::size_t logits_n_input_frames_valid,
    const std::function<
        void(std::tuple<std::optional<kaldi::CompactLattice>,
                        std::optional<kaldi::cuda_decoder::CTMResult>> &)>
        &callback) {
  UtteranceTask task;
  // at 5000 files/s, expected to overflow in ~116 million years
  task.corr_id = corr_id_cnt_.fetch_add(1);
  task.callback = callback;
  task.d_logits = d_logits;
  task.logits_frame_stride = logits_frame_stride;
  task.logits_n_input_frames_valid = logits_n_input_frames_valid;
  n_tasks_not_done_.fetch_add(1);
  // TODO
  {
    std::lock_guard<std::mutex> lk(outstanding_utt_m_);
    outstanding_utt_q_.push(std::move(task));
  }
}

void BatchedMappedDecoderCuda::AcquireTasks() {
  std::lock_guard<std::mutex> lk(outstanding_utt_m_);
  while (current_tasks_.size() < max_batch_size_) {
    if (outstanding_utt_q_.empty()) {
      // KALDI_LOG << "outstanding_utt_q_.empty()";
      break;
    }

    UtteranceTask &task = outstanding_utt_q_.front();

    bool was_created = cuda_online_pipeline_.TryInitCorrID(task.corr_id);
    if (!was_created)
      break;

    auto &callback = task.callback;

    cuda_online_pipeline_.SetLatticeCallback(
        task.corr_id,
        [this, callback](
            std::tuple<std::optional<kaldi::CompactLattice>,
                       std::optional<kaldi::cuda_decoder::CTMResult>> &result) {
          if (callback)
            callback(result);
          n_tasks_not_done_.fetch_sub(1, std::memory_order_release);
        });
    current_tasks_.push_back(std::move(task));
    outstanding_utt_q_.pop();
  }
}

void BatchedMappedDecoderCuda::ComputeTasks() {
  // try using a condition variable?
  while (threads_running_.load()) {
    // Is this optimized out?
    if (current_tasks_.size() < max_batch_size_)
      AcquireTasks();
    if (current_tasks_.empty()) {
      // lk.unlock();
      kaldi::Sleep(kSleepForNewTask);
      continue;
    } else {
      BuildBatchFromCurrentTasks();

      cuda_online_pipeline_.DecodeBatch(
          batch_corr_ids_, batch_d_logits_, batch_logits_frame_stride_,
          batch_n_input_frames_valid_, batch_is_first_chunk_,
          batch_is_last_chunk_, nullptr);
      // call destructors of completed tasks
      tasks_last_chunk_.clear();
    }
  }
}

void BatchedMappedDecoderCuda::BuildBatchFromCurrentTasks() {
  batch_corr_ids_.clear();
  batch_d_logits_.clear();
  batch_logits_frame_stride_.clear();
  batch_n_input_frames_valid_.clear();
  batch_is_last_chunk_.clear();
  batch_is_first_chunk_.clear();

  for (size_t task_id = 0; task_id < current_tasks_.size();) {
    // what happens if someone pushes to the queue while I pop from it?
    // Wait, there are the current tasks, and then there is the queue.
    // Confusing...
    UtteranceTask &task = current_tasks_[task_id];
    std::int32_t total_n_input = task.logits_n_input_frames_valid;

    std::int32_t loglikes_offset = task.loglikes_time_offset;
    std::int32_t loglikes_remaining = total_n_input - loglikes_offset;
    std::int32_t num_loglikes =
        std::min(n_input_per_chunk_, loglikes_remaining);
    assert(num_loglikes > 0);
    bool is_last_chunk = (loglikes_remaining == num_loglikes);
    bool is_first_chunk = (loglikes_offset == 0);
    CorrelationID corr_id = task.corr_id;

    batch_corr_ids_.push_back(task.corr_id);
    batch_d_logits_.push_back(task.d_logits + task.loglikes_time_offset *
                                                  task.logits_frame_stride);
    batch_logits_frame_stride_.push_back(task.logits_frame_stride);
    batch_n_input_frames_valid_.push_back(num_loglikes);
    batch_is_last_chunk_.push_back(is_last_chunk);
    batch_is_first_chunk_.push_back(is_first_chunk);

    task.loglikes_time_offset += num_loglikes;

    if (is_last_chunk) {
      // is this messing something up?
      // is std::move messing something up?
      // is something else corrupting tasks_last_chunk_?
      tasks_last_chunk_.push_back(std::move(task));
      std::size_t last_task_id = current_tasks_.size() - 1;
      // if last_task_id == task_id, what is the outcome of this std::move?
      current_tasks_[task_id] = std::move(current_tasks_[last_task_id]);
      current_tasks_.pop_back();
    } else {
      // If it was the last chunk, we replaced the current
      // task with another one we must process that task_id
      // again (because it is now another task) If it was not
      // the last chunk, then we must take care of the next
      // task_id
      ++task_id;
    }
  }
}

void BatchedMappedDecoderCuda::WaitForAllTasks() {
  // I feel like we should have a condition variable of some sort here...
  while (n_tasks_not_done_.load() != 0) {
    // why not just call join on the one thread?
    kaldi::Sleep(kSleepForTaskComplete);
  }
}

const fst::SymbolTable &BatchedMappedDecoderCuda::GetSymbolTable() const {
  return cuda_online_pipeline_.GetSymbolTable();
}

} // end namespace riva::asrlib
