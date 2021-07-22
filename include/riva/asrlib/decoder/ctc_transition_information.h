// itf/transition-information.h

// Copyright 2021 NVIDIA

// TODO: Apache2 headers

#include "itf/transition-information.h"
#include "fst/fst.h"

#include <stdint.h>
#include <cstdint>
#include <unordered_map>

namespace riva::asrlib {

// I need a precise defintion of a transition id before continuing
// transition id should be the "label" of each. They should be
// 1-based (i.e., the caller will add 1)
class CTCTransitionInformation: public kaldi::TransitionInformation {
 public:
    CTCTransitionInformation(const fst::StdFst& ctc_topo);

    CTCTransitionInformation(int32_t num_tokens_including_blank);

    bool TransitionIdsEquivalent(int32_t trans_id1, int32_t trans_id2) const override;

    bool TransitionIdIsStartOfPhone(int32_t trans_id) const override;

    int32_t TransitionIdToPhone(int32_t trans_id) const override;

    bool IsFinal(int32_t trans_id) const override;

    bool IsSelfLoop(int32_t trans_id) const override;

    const std::vector<int32_t>& TransitionIdToPdfArray() const override;

    int32_t NumPdfs() const override;    

private:
    int32_t num_pdfs_;
    std::vector<int32_t> trans_id_to_pdf_id_;
};

} // end namespace riva::asrlib
