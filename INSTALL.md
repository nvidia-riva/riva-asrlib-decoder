#Installation instructions

Run following sequence of commands in your virtual environment

```shell
pip install -r egs/requirements_libs.txt
git submodule init
git submodule update --recursive
pip install  -e .[testing]

#Ensure all scripts and binaries are in PATH
SCRIPT_PATH=`pwd`/src/riva/asrlib/decoder/scripts/prepare_TLG_fst/
export PATH=$SCRIPT_PATH:$SCRIPT_PATH/utils:$SCRIPT_PATH/local:$SCRIPT_PATH/bin:$PATH
```

#Creating RMIRs

```shell
sudo docker run --rm --gpus all -v $MODEL_PATH:/servicemaker-dev -i -t nvcr.io/nvidia/riva/riva-speech:2.8.1-servicem
aker riva-build speech_recognition  --decoder_type=kaldi --decoding_language_model_fst=/servicemaker-dev/<graph_folder>/TLG.fst --kaldi_decoder.vocab_file=/servicemaker-dev/<graph_folder>/words.txt   --kaldi
_decoder.asr_model_delay=5 --kaldi_decoder.default_beam=17.0 --kaldi_decoder.max_active=7000   --kaldi_decoder.determinize_lattice=true --kaldi_decoder.max_batch_size=200 /servicemaker-dev/rmir/Conformer-CTC-XL.rmir /servicemaker-dev/<am_model_folder>>/Conformer-CTC-XL_spe-128_en-US_Riva-ASR-SET-4.0.riva
```

