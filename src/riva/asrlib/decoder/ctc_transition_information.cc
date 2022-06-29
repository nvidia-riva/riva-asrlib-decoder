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

#include "riva/asrlib/decoder/ctc_transition_information.h"

#include <type_traits>

namespace riva::asrlib {

using namespace kaldi;

static_assert(std::is_same<fst::StdFst::StateId, int32_t>::value, "");

// I need a precise defintion of a transition id before continuing
// transition id should be the "label" of each. They should be
// 1-based (i.e., the caller will add 1)
CTCTransitionInformation::CTCTransitionInformation(const fst::StdFst& ctc_topo)
{
  std::unordered_set<int32_t> unique_pdfs;
  for (fst::StateIterator<fst::StdFst> siter(ctc_topo); !siter.Done(); siter.Next()) {
    fst::StdFst::StateId state_id = siter.Value();
    assert(ctc_topo.Final(state_id) != fst::StdFst::Weight::Zero() && "all states should be final");
    for (fst::ArcIterator<fst::StdFst> aiter(ctc_topo, state_id); !aiter.Done(); aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      if (arc.ilabel == 0) {
        continue;
      }
      unique_pdfs.emplace(arc.ilabel);
    }
  }

  num_pdfs_ = unique_pdfs.size();

  trans_id_to_pdf_id_ = std::vector<int32_t>(num_pdfs_ + 1, -1);

  for (fst::StateIterator<fst::StdFst> siter(ctc_topo); !siter.Done(); siter.Next()) {
    fst::StdFst::StateId state_id = siter.Value();
    for (fst::ArcIterator<fst::StdFst> aiter(ctc_topo, state_id); !aiter.Done(); aiter.Next()) {
      const fst::StdArc& arc = aiter.Value();
      if (arc.ilabel == 0) {
        continue;
      }
      trans_id_to_pdf_id_[arc.ilabel] = arc.ilabel - 1;
    }
  }
  assert(trans_id_to_pdf_id_[0] == -1);
}

CTCTransitionInformation::CTCTransitionInformation(int32_t num_tokens_including_blank)
    : num_pdfs_(num_tokens_including_blank), trans_id_to_pdf_id_(num_pdfs_ + 1, -1)
{
  for (int32_t i = 1; i < int32_t(trans_id_to_pdf_id_.size()); ++i) {
    trans_id_to_pdf_id_[i] = i - 1;
  }

  assert(trans_id_to_pdf_id_.front() == -1);
  assert(trans_id_to_pdf_id_.back() != -1);
}

bool
CTCTransitionInformation::TransitionIdsEquivalent(int32_t trans_id1, int32_t trans_id2) const
{
  return trans_id1 == trans_id2;  // || trans_id2 == blank_trans_id_;
}

bool
CTCTransitionInformation::TransitionIdIsStartOfPhone(int32_t trans_id) const
{
  return true;
}

int32
CTCTransitionInformation::TransitionIdToPhone(int32 trans_id) const
{
  return trans_id;
}

bool
CTCTransitionInformation::IsFinal(int32 trans_id) const
{
  // in a CTC topology, every state should be a final state. We
  // can verify this in the constructor, if necessary.
  return true;
}

bool
CTCTransitionInformation::IsSelfLoop(int32 trans_id) const
{
  return false;
}

const std::vector<int32_t>&
CTCTransitionInformation::TransitionIdToPdfArray() const
{
  return trans_id_to_pdf_id_;
}

int32_t
CTCTransitionInformation::NumPdfs() const
{
  return num_pdfs_;
}

}  // end namespace riva::asrlib
