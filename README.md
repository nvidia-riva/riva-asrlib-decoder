# Riva ASR Library CUDA WFST Decoder

[Paper](https://arxiv.org/abs/2311.04996).

This code repository exposes the CUDA WFST decoder originally
described here https://arxiv.org/abs/1910.10032 as a C++ library and
as a python library (via wrapping the C++ library with pybind11). It
can be used to decode CTC models in particular. The new paper can be
found at https://arxiv.org/abs/2311.04996.

It is the same decoder used when you specify --decoder_type=kaldi when
running `riva-build` in NVIDIA Riva. However, this can be used
standalone and with whatever CTC model you want, in any framework of
your choosing.

## Pre-built wheels

Install from [PyPi](https://pypi.org/project/riva-asrlib-decoder/).

```
pip install riva-asrlib-decoder
```

These manylinux2014 wheels, with the exception that they use cuda. You
need cuda 11.2.1 or greater and python 3.6 or greater.

Cuda as old as 10.2 could also be supported, but there is an
[incompatibility](https://forums.developer.nvidia.com/t/nvc-20-9-fails-to-compile-code-instantiating-any-std-tuple-with-gcc-10-2-on-c-17/160987)
with gcc-10 (which manylinux2014 uses) not fixed until cuda 11.2.1.

## Usage

```
import os
import urllib.request

import nemo.collections.asr as nemo_asr

from riva.asrlib.decoder.config import create_default_offline_decoder_config, create_TLG_from_nemo
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda

nemo_model_name = "stt_en_conformer_ctc_large"

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    nemo_model_name,
    map_location=torch.device("cuda")
)

work_dir = "work_dir"

os.makedirs(work_dir, exist_ok=True)

arpa_lm_path = os.path.join(work_dir, "3-gram.pruned.3e-7.arpa.gz")

urllib.request.urlretrieve("https://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz", arpa_lm_path)

config = create_default_offline_decoder_config(batch_size=16, frame_shift_seconds=0.04)
wfst_file, words_file = create_TLG_from_nemo("compact", work_dir, nemo_model_name, arpa_lm_path)

decoder = BatchedMappedDecoderCuda(
    config,
    wfst_file,
    words_file,
    num_tokens_including_blank,
)

with torch.inference_mode(), torch.autocast("cuda"):
    log_probs, lengths, _ = asr_model.forward(
        input_signal=input_signal, input_signal_length=input_signal_length
    )
    cpu_lengths = lengths.to(torch.int64).to('cpu')
    transcripts = decoder.decode_mbr(log_probs.to(torch.float32), cpu_lengths)
```

## Build and Test

To build just the offline binary:

```
mkdir build
cd build
cmake -DRIVA_ASRLIB_BUILD_PYTHON_BINDINGS=NO ..
cmake --build --target all --parallel
# Run build/offline-cuda-decode-binary
```

To build the python bindings from source and optionally run the python
test cases:

```
# Optionally set up a conda environment first.
pip install -e .[testing]
python -m unittest src/riva/asrlib/decoder/test_decoder.py 2>&1 | tee test.log
```


## Bazel

Bazel build is not being used. Please don't expect it to work.