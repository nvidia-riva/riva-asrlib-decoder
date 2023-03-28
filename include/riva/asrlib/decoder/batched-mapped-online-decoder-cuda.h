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
#include <fst/fstlib.h>

// kaldi includes
#include <cudadecoder/cuda-decoder-common.h>
#include <cudadecoder/cuda-decoder.h>
#include <cudadecoder/lattice-postprocessor.h>
#include <fstext/lattice-utils.h>
#include <itf/options-itf.h>
#include <itf/transition-information.h>
#include <lat/determinize-lattice-pruned.h>
#include <lat/lattice-functions.h>

#include <optional>

#define KALDI_CUDA_DECODER_MIN_NCHANNELS_FACTOR 2

namespace riva::asrlib {

const double kSleepForChannelAvailable = 1e-3;
const double kSleepForCallBack = 10e-3;

// BatchedMappedOnlineDecoderCuda
struct BatchedMappedOnlineDecoderCudaConfig {
  BatchedMappedOnlineDecoderCudaConfig()
      : max_batch_size(400), num_channels(800), num_post_processing_worker_threads(-1),
        determinize_lattice(true), num_decoder_copy_threads(2),
        frame_shift_seconds(std::numeric_limits<float>::max()), use_lattice_postprocessor(true)
  {
  }
  void Register(kaldi::OptionsItf* po)
  {
    // equal to num-lanes?
    po->Register(
        "max-batch-size", &max_batch_size,
        "The maximum execution batch size."
        " Larger = better throughput, but higher latency.");
    po->Register(
        "num-channels", &num_channels,
        "The number of parallel audio channels. This is the maximum"
        " number of parallel audio channels supported by the pipeline."
        " This should be larger than max_batch_size.");
    po->Register(
        "cpu-post-processing-threads", &num_post_processing_worker_threads,
        "The total number of CPU threads launched to process CPU"
        " tasks. -1 = use std::hardware_concurrency().");
    po->Register(
        "determinize-lattice", &determinize_lattice, "Determinize the lattice before output.");
    // why do host to host copies occur again?
    po->Register(
        "cuda-decoder-copy-threads", &num_decoder_copy_threads,
        "Advanced - Number of worker threads used in the"
        " decoder for the host to host copies.");
    po->Register(
        "cuda-decoder-frame-shift-seconds", &frame_shift_seconds,
        "The sampling period of log-likelihood vectors output by the "
        "acoustic model.");
    po->Register("use-lattice-postprocessor", &use_lattice_postprocessor, "");

    decoder_opts.Register(po);
    det_opts.Register(po);
    lattice_postprocessor_opts.Register(po);
  }
  // how does max_batch_size relate to lanes?
  int max_batch_size;
  int num_channels;
  int num_post_processing_worker_threads;
  bool determinize_lattice;
  int num_decoder_copy_threads;
  float frame_shift_seconds{0.0f};
  bool use_lattice_postprocessor;

  kaldi::cuda_decoder::CudaDecoderConfig decoder_opts;
  // can't necessarily determinize in this way... Actually, yes, we
  // can, but we may not be using phones.
  fst::DeterminizeLatticePhonePrunedOptions det_opts;
  kaldi::cuda_decoder::LatticePostprocessorConfig lattice_postprocessor_opts;

  void CheckAndFixConfigs()
  {
    KALDI_ASSERT(max_batch_size > 0);
    // Lower bound on nchannels.
    // Using strictly more than max_batch_size because channels are still used
    // when the lattice postprocessing is running. We still want to run full
    // max_batch_size batches in the meantime
    // Need to do work while latency postprocessing is occurring in order to
    // maximize performance.

    // computing 400 * 2 = 800 here...
    int min_nchannels = max_batch_size * KALDI_CUDA_DECODER_MIN_NCHANNELS_FACTOR;
    num_channels = std::max(num_channels, min_nchannels);

    // If not set use number of physical threads
    num_post_processing_worker_threads = (num_post_processing_worker_threads > 0)
                                             ? num_post_processing_worker_threads
                                             : std::thread::hardware_concurrency();
    KALDI_ASSERT(frame_shift_seconds != 0.0f);
  }
};

class BatchedMappedOnlineDecoderCuda {
 public:
  using CorrelationID = uint64_t;
  using ReturnType = std::tuple<
      std::optional<kaldi::CompactLattice>,
      std::optional<kaldi::cuda_decoder::CTMResult>,
      std::optional<std::vector<kaldi::cuda_decoder::NBestResult>>>;
  using LatticeCallback = std::function<void(ReturnType&)>;
  BatchedMappedOnlineDecoderCuda(
      const BatchedMappedOnlineDecoderCudaConfig& config, const fst::Fst<fst::StdArc>& decode_fst,
      std::unique_ptr<kaldi::TransitionInformation>&& trans_information)
      : config_(config), trans_information_(std::move(trans_information)),
        partial_hypotheses_(nullptr), end_points_(nullptr),
        cuda_fst_(
            std::make_unique<kaldi::cuda_decoder::CudaFst>(decode_fst, trans_information_.get())),
        // cuda_decoder_(std::make_unique<kaldi::cuda_decoder::CudaDecoder>(
        //     *cuda_fst_, config_.decoder_opts,
        //     // here's the bug!
        //     /*nlanes=*/config_.max_batch_size, config_.num_channels)),
        word_syms_()
  {
    config_.CheckAndFixConfigs();

    cuda_decoder_ = std::make_unique<kaldi::cuda_decoder::CudaDecoder>(
        *cuda_fst_, config_.decoder_opts,
        /*nlanes=*/config_.max_batch_size, config_.num_channels);

    if (config_.use_lattice_postprocessor) {
      lattice_postprocessor_ = std::make_unique<kaldi::cuda_decoder::LatticePostprocessor>(
          config_.lattice_postprocessor_opts);
      lattice_postprocessor_->SetTransitionInformation(trans_information_.get());
      lattice_postprocessor_->SetDecoderFrameShift(config_.frame_shift_seconds);
    }

    cuda_decoder_->SetOutputFrameShiftInSeconds(config_.frame_shift_seconds);

    available_channels_.resize(config_.num_channels);
    std::iota(available_channels_.begin(), available_channels_.end(),
              0);  // 0,1,2,3..
    lattice_callbacks_.reserve(config_.num_channels);
    corr_id2channel_.reserve(config_.num_channels);
    n_input_frames_valid_.resize(config_.max_batch_size);
    n_lattice_callbacks_not_done_.store(0);

    int num_worker_threads = config_.num_post_processing_worker_threads;
    if (num_worker_threads > 0) {
      thread_pool_ = std::make_unique<kaldi::futures_thread_pool>(num_worker_threads);
    }
    if (config_.num_decoder_copy_threads > 0) {
      cuda_decoder_->SetThreadPoolAndStartCPUWorkers(
          thread_pool_.get(), config_.num_decoder_copy_threads);
    }
  }

  ~BatchedMappedOnlineDecoderCuda()
  {
    // The destructor races with callback completion. Even if all callbacks have
    // finished, the counter may (non-deterministically) lag behind by a few ms.
    // Deleting the object when all callbacks had been called is UB: the
    // variable n_lattice_callbacks_not_done_ is accessed after a callback has
    // returned.
    WaitForLatticeCallbacks();
    assert(n_lattice_callbacks_not_done_ == 0);
    assert(available_channels_.empty() || available_channels_.size() == config_.num_channels);
  }

  void WaitForLatticeCallbacks() noexcept
  {
    while (n_lattice_callbacks_not_done_.load() != 0) kaldi::Sleep(kSleepForCallBack);
  }
  void DecodeBatch(
      const std::vector<CorrelationID>& corr_ids, const std::vector<const float*>& d_logits,
      const std::vector<std::size_t>& logits_frame_stride,
      const std::vector<std::size_t>& n_logit_frames_valid, const std::vector<bool>& is_first_chunk,
      // doesn't is_last_chunk depend upon the minimum
      // value in n_logit_frames_valid?
      const std::vector<bool>& is_last_chunk, std::vector<int>* channels)
  {
    nvtxRangePushA("DecodeBatch");
    if (channels != nullptr) {
      channels = &channels_;
    }
    ListIChannelsInBatch(corr_ids, &channels_);

    std::vector<int> list_channels_first_chunk;
    for (std::size_t i = 0; i < is_first_chunk.size(); ++i) {
      if (is_first_chunk[i]) {
        list_channels_first_chunk.push_back(channels_[i]);
      }
    }
    if (!list_channels_first_chunk.empty()) {
      // KALDI_LOG << "GALVEZ:InitDecoding()";
      cuda_decoder_->InitDecoding(list_channels_first_chunk);
    }

    std::vector<std::pair<kaldi::cuda_decoder::ChannelId, const float*>> lanes_assignments;
    lanes_assignments.reserve(channels_.size());
    std::size_t frames_to_decode =
        *std::max_element(n_logit_frames_valid.begin(), n_logit_frames_valid.end());
    for (std::size_t f = 0; f < frames_to_decode; ++f) {
      lanes_assignments.clear();
      for (int32_t ilane = 0; ilane < channels_.size(); ++ilane) {
        const kaldi::cuda_decoder::ChannelId ichannel = channels_[ilane];
        if (f < n_logit_frames_valid[ilane]) {
          lanes_assignments.emplace_back(
              ichannel, d_logits[ilane] + f * logits_frame_stride[ilane]);
        }
      }
      cuda_decoder_->AdvanceDecoding(lanes_assignments);
    }

    RunCallbacksAndFinalize(corr_ids, channels_, is_last_chunk);
    nvtxRangePop();
  }

  // Set the callback function to call with the final lattice for a given
  // corr_id
  void SetLatticeCallback(CorrelationID corr_id, const LatticeCallback& callback)
  {
    std::lock_guard<std::mutex> lk(map_callbacks_m_);
    lattice_callbacks_.insert({corr_id, callback});
  }

  void RunCallbacksAndFinalize(
      const std::vector<CorrelationID>& corr_ids, const std::vector<int>& channels,
      const std::vector<bool>& is_last_chunk)
  {
    // KALDI_LOG << "RunCallbacksAndFinalize" << is_last_chunk.size();
    std::vector<kaldi::cuda_decoder::ChannelId> list_channels_last_chunk;
    std::vector<LatticeCallback*> list_lattice_callbacks_last_chunk;
    {
      std::lock_guard<std::mutex> lk_callbacks(map_callbacks_m_);
      std::lock_guard<std::mutex> lk_channels(available_channels_m_);
      for (std::size_t i = 0; i < is_last_chunk.size(); ++i) {
        if (is_last_chunk[i]) {
          kaldi::cuda_decoder::ChannelId ichannel = channels[i];
          CorrelationID corr_id = corr_ids[i];

          bool has_lattice_callback = false;
          decltype(lattice_callbacks_.end()) it_lattice_callback;
          if (!lattice_callbacks_.empty()) {
            it_lattice_callback = lattice_callbacks_.find(corr_id);
            has_lattice_callback = (it_lattice_callback != lattice_callbacks_.end());
          }
          if (has_lattice_callback) {
            // const LatticeCallback cannot be meaningfully moved...
            LatticeCallback* lattice_callback =
                new LatticeCallback(std::move(it_lattice_callback->second));
            lattice_callbacks_.erase(it_lattice_callback);
            list_channels_last_chunk.push_back(ichannel);
            list_lattice_callbacks_last_chunk.push_back(lattice_callback);
          } else {
            available_channels_.push_back(ichannel);
            int32 ndeleted = corr_id2channel_.erase(corr_id);
            KALDI_ASSERT(ndeleted == 1);
          }
        }
      }
    }

    // KALDI_LOG << "GALVEZ:RunCallbacksAndFinalize=" << list_channels_last_chunk.size();

    if (list_channels_last_chunk.empty()) {
      return;
    }

    // this must finish before ConcurrentGetRawLatticeSingleChannel is called
    // but it clearly does?
    cuda_decoder_->PrepareForGetRawLattice(list_channels_last_chunk, true);
    n_lattice_callbacks_not_done_.fetch_add(
        list_channels_last_chunk.size(), std::memory_order_acquire);

    for (std::size_t i = 0; i < list_channels_last_chunk.size(); ++i) {
      uint64_t ichannel = list_channels_last_chunk[i];
      LatticeCallback* lattice_callback = list_lattice_callbacks_last_chunk[i];
      thread_pool_->submit(std::bind(
          &BatchedMappedOnlineDecoderCuda::FinalizeDecoding, this, ichannel, lattice_callback));
    }
  }

  bool TryInitCorrID(CorrelationID corr_id, std::int32_t wait_for_us = 0)
  {
    // KALDI_LOG << "GALVEZ:TryInitCorrID";
    bool inserted;
    decltype(corr_id2channel_.end()) it;
    std::tie(it, inserted) = corr_id2channel_.insert({corr_id, -1});
    int32 ichannel;
    if (inserted) {
      // The corr_id was not in use
      std::unique_lock<std::mutex> lk(available_channels_m_);
      bool channel_available = (available_channels_.size() > 0);
      if (!channel_available) {
        // We cannot use that corr_id
        int waited_for = 0;
        while (waited_for < wait_for_us) {
          lk.unlock();
          kaldi::Sleep(kSleepForChannelAvailable);
          waited_for += int32(kSleepForChannelAvailable * 1e6);
          lk.lock();
          channel_available = (available_channels_.size() > 0);
          if (channel_available)
            break;
        }

        // If still not available return
        if (!channel_available) {
          corr_id2channel_.erase(it);
          return false;
        }
      }

      ichannel = available_channels_.back();
      available_channels_.pop_back();
      it->second = ichannel;
    } else {
      // This corr id was already in use but not closed
      // It can happen if for instance a channel lost connection and
      // did not send its last chunk Cleaning up
      KALDI_WARN << "This corr_id was already in use";
      ichannel = it->second;
    }

    return true;
  }

  void SetSymbolTable(fst::SymbolTable word_syms)
  {
    word_syms_ = std::move(word_syms);
    cuda_decoder_->SetSymbolTable(word_syms_);
  }

  const fst::SymbolTable& GetSymbolTable() const { return word_syms_; }


 private:
  // could use this as an opportunity to clear saved variables for the channel
  void FinalizeDecoding(int32 ichannel, const LatticeCallback* callback)
  {
    kaldi::Lattice lat;
    cuda_decoder_->ConcurrentGetRawLatticeSingleChannel(ichannel, &lat);

    // Getting the channel callback now, we're going to free that channel
    // Done with this channel. Making it available again
    {
      std::lock_guard<std::mutex> lk(available_channels_m_);
      available_channels_.push_back(ichannel);
    }

    // If necessary, determinize the lattice
    kaldi::CompactLattice dlat;
    if (config_.determinize_lattice) {
      // may *not* determinize the lattice
      fst::DeterminizeLatticePhonePrunedWrapper(
          *trans_information_, &lat, config_.decoder_opts.lattice_beam, &dlat, config_.det_opts);
    } else {
      //
      fst::ConvertLattice(lat, &dlat);
    }

    // this is wasteful, since GetCTM calls GetPostprocessedLattice
    // but doesn't return the result.
    ReturnType result;
    if (config_.use_lattice_postprocessor) {
      kaldi::CompactLattice clat;
      lattice_postprocessor_->GetPostprocessedLattice(dlat, &clat);
      kaldi::cuda_decoder::CTMResult ctm;
      // is this okay? I am modifying the refernece twice...
      lattice_postprocessor_->GetCTM(dlat, &ctm);
      std::vector<kaldi::cuda_decoder::NBestResult> nbest =
          lattice_postprocessor_->GetNBestList(dlat);
      result = {std::make_optional(clat), std::make_optional(ctm), std::make_optional(nbest)};
    } else {
      result = {std::make_optional(dlat), std::nullopt, std::nullopt};
    }

    // if ptr set and if callback func callable
    if (callback && *callback) {
      (*callback)(result);
      delete callback;
    }

    n_lattice_callbacks_not_done_.fetch_sub(1, std::memory_order_release);
  }

  void ListIChannelsInBatch(const std::vector<CorrelationID>& corr_ids, std::vector<int>* channels)
  {
    channels->clear();
    for (int i = 0; i < corr_ids.size(); ++i) {
      int corr_id = corr_ids[i];
      int ichannel = corr_id2channel_.at(corr_id);
      channels->push_back(ichannel);
    }
  }

  BatchedMappedOnlineDecoderCudaConfig config_;
  // Models
  std::unique_ptr<kaldi::TransitionInformation> trans_information_;

  // Decoder channels currently available, w/ mutex
  std::vector<int32> available_channels_;
  std::mutex available_channels_m_;

  // corr_id -> decoder channel map
  std::unordered_map<CorrelationID, int32> corr_id2channel_;

  // Where to store partial_hypotheses_ and end_points_ if available
  std::vector<const std::string*>* partial_hypotheses_;
  std::vector<bool>* end_points_;

  // The callback is called once the final lattice is ready
  std::unordered_map<CorrelationID, const LatticeCallback> lattice_callbacks_;
  // Lock for callbacks
  std::mutex map_callbacks_m_;
  std::mutex stdout_m_;

  std::vector<int> n_input_frames_valid_;

  std::vector<std::vector<std::pair<int, float*>>> all_frames_log_posteriors_;

  // Parameters extracted from the models
  int input_frames_per_chunk_;
  int output_frames_per_chunk_;
  float seconds_per_chunk_;
  float samples_per_chunk_;
  float model_frequency_;
  int32 ivector_dim_, input_dim_;

  // Used with CPU features extraction. Contains the number of CPU FE tasks
  // still running
  std::atomic<int> n_compute_features_not_done_;
  // Number of CPU lattice postprocessing tasks still running
  std::atomic<int> n_lattice_callbacks_not_done_;

  // Current assignement buffers, when DecodeBatch is running
  std::vector<int> channels_;

  // Ordering of the cuda_fst_ w.r.t. thread_pool_ and the decoder is important:
  // order of destruction is bottom-up, opposite to the order of construction.
  // We want the FST object, which is entirely passive and only frees device
  // FST representation when destroyed, to survive both the thread pool and the
  // decoder, which both may perform pending work during destruction. Since no
  // new work may be fed into this object while it is being destroyed, the
  // relative order of the latter two is unimportant, but just in case, FST must
  // stay around until the other two are positively quiescent.

  // HCLG graph. CudaFst is a host object, but owns pointers to the data stored
  // in GPU memory.
  std::unique_ptr<kaldi::cuda_decoder::CudaFst> cuda_fst_;

  // The thread pool receives data from device and post-processes it. This class
  // destructor blocks until the thread pool is drained of work items.
  std::unique_ptr<kaldi::futures_thread_pool> thread_pool_;

  // The decoder owns thread(s) that reconstruct lattices transferred from the
  // device in a compacted form as arrays with offsets instead of pointers.
  std::unique_ptr<kaldi::cuda_decoder::CudaDecoder> cuda_decoder_;

  // Used for debugging
  fst::SymbolTable word_syms_;

  std::unique_ptr<kaldi::cuda_decoder::LatticePostprocessor> lattice_postprocessor_;
};

}  // namespace riva::asrlib
