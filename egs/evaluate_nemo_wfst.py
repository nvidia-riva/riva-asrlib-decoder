import soundfile as sf
import torch
import sys
import numpy

from pathlib import Path
from riva.asrlib.decoder.python_decoder import BatchedMappedDecoderCuda, BatchedMappedDecoderCudaConfig
import multiprocessing
import nemo.collections.asr as nemo_asr
import more_itertools

def nemo_logits(file, processor, model):
    samples, fs = sf.read(file)

    audio_input = processor(samples, sampling_rate=fs, return_tensors="pt")
    with torch.no_grad():
        return model(**audio_input).logits.cpu().numpy()[0]


def generate_output(asr_model_name, graph_path, input_folder, results_file):
    device = torch.device('cuda')

    print(asr_model_name)
    with open(results_file,'w') as rf:
        p = Path(input_folder)



        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            restore_path=asr_model_name
        ).to(torch.device(device))

        config = BatchedMappedDecoderCudaConfig()
        config.n_input_per_chunk = 50
        config.online_opts.decoder_opts.default_beam = 17.0
        config.online_opts.decoder_opts.lattice_beam = 8.0
        config.online_opts.decoder_opts.max_active = 7000
        config.online_opts.determinize_lattice = True
        config.online_opts.max_batch_size = 200
        config.online_opts.num_channels = config.online_opts.max_batch_size * 2
        config.online_opts.frame_shift_seconds = 0.03
        config.online_opts.num_post_processing_worker_threads = max(multiprocessing.cpu_count()-4,4)
        config.online_opts.num_decoder_copy_threads = 4
        num_tokens_including_blank=len(asr_model.tokenizer.vocab)+2 # add 2 tokens: <eps> and #<blk> to num_tokens

        decoder = BatchedMappedDecoderCuda(
            config, str(f"{graph_path}/TLG.fst"),
            str(f"{graph_path}/words.txt"), num_tokens_including_blank
        )
        rf.write(f"Name\tTranscript")

        for file in p.glob('*.wav'):
            logits = asr_model.transcribe([str(file)], logprobs=True)[0]

            name = ' '.join(file.name[:-4].split('-')).lower()
            boosted=""
            print(logits.shape)
            log_probs=torch.from_numpy(numpy.array([logits])).to(device)
            seq_lens=torch.IntTensor([logits.shape[0]]).to(torch.int64)
            result = decoder.decode_mbr(log_probs, seq_lens)[0]
            transcript = ' '.join([word for word,*_ in result])

            print(result)

            rf.write(f"{name}\t{transcript}\n")

if __name__=="__main__":

    if len(sys.argv)==5:
        model_path = sys.argv[1]
        graph_path = sys.argv[2]
        data_path = sys.argv[3]
        output_path = sys.argv[4]
    else:
        exit(1)
    print(f"model_path : {model_path}\ngraph_path : {graph_path}\ndata_path : {data_path}\noutput_path : {output_path}\n")
    generate_output(model_path, graph_path, data_path, output_path)
