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
#pragma once

namespace riva::asrlib {

class AcousticModel {
 public:
  virtual void RunBatch(
      const std::vector<int>& channels, const std::vector<float*>& d_features,
      const std::vector<int>& features_strides,
      // const std::vector<float *> &d_ivectors,
      const std::vector<int>& n_input_frames_valid, const std::vector<bool>& is_first_chunk,
      const std::vector<bool>& is_last_chunk, const float* d_all_log_posteriors,
      std::vector<std::vector<std::pair<int, const float*>>>* all_frames_log_posteriors) = 0;

  void FormatOutputPtrs(
      const std::vector<int>& channels, const float* d_all_log_posteriors,
      std::vector<std::vector<std::pair<int, const float*>>>* all_frames_log_posteriors_ptrs,
      const std::vector<int>& n_output_frames_valid,
      const std::vector<int>* n_output_frames_valid_offset = NULL);

  virtual int GetNOutputFramesPerChunk() = 0;
  virtual int GetTotalNnet3RightContext() = 0;
  virtual std::size_t SubSamplingFactor() const = 0;
};

}  // namespace riva::asrlib
