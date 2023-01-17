import logging
import soundfile
import torch

from typing import Optional, Sequence, Tuple, Union, List
from omegaconf import OmegaConf
from typing import List
from multiprocessing import Process, Queue
from typeguard import check_argument_types

from espnet2.tasks.asr import ASRTask
from espnet.nets.scorer_interface import BatchScorerInterface

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet_model_zoo.downloader import ModelDownloader
from torch.cuda.amp import autocast
from arguments import get_parser
from typing import Optional, Sequence, Tuple, Union, List
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoConfig
from speech2text import Speech2Text

import subprocess
import whisper
import os

def whisper_inference(
    output_dir: str,
    fp16: bool,
    beam_size: int,
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    language: Optional[str],
    model_size: str,
) -> None:
    print('whisper inference')
    scp_path, _, _ = data_path_and_name_and_type[0]
    with open(scp_path, 'r+') as f:
        lines = f.readlines()

    options = whisper.DecodingOptions(
        language=language,
        beam_size=beam_size,
        fp16=fp16
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_size, device=device)
    
    output = []
    for line in lines:
        i, path = line.strip().split(' ')
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # decode the audio
        result = whisper.decode(model, mel, options)
        print(result.text)

        output.append(" ".join([i, result.text]) + '\n')
    
    # make directory
    os.makedirs(path:=os.path.join(output_dir, '1best_recog'), exist_ok=True)
    with open(os.path.join(path, 'text'), 'w+') as f:
        for s in output:
            f.write(s)
        f.close()
    
def espnet_inference(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: str,
    asr_model_file: str,
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build speech2text
    speech2text = Speech2Text(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        lm_train_config=lm_train_config,
        lm_file=lm_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio,
        dtype=dtype,
        beam_size=beam_size,
        ctc_weight=ctc_weight,
        lm_weight=lm_weight,
        penalty=penalty,
        nbest=nbest,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(speech2text.asr_train_args, False),
        collate_fn=ASRTask.build_collate_fn(speech2text.asr_train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 7 .Start for-loop
    with DatadirWriter(output_dir) as writer:
        try:
            for keys, batch in loader:
                assert isinstance(batch, dict), type(batch)
                assert all(isinstance(s, str) for s in keys), keys
                _bs = len(next(iter(batch.values())))
                assert len(keys) == _bs, f"{len(keys)} != {_bs}"
                batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}

                # N-best list of (text, token, token_int, hyp_object)
                results = speech2text(**batch)

                # Only supporting batch_size==1
                key = keys[0]
                for n, (text, token, token_int, hyp) in zip(range(1, nbest + 1), results):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}best_recog"]

                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["text"][key] = text
        except TypeError as typeerror:
                print("error occurred : ", typeerror)
        except BaseException as e:
                print("error occurred : ", e)
    return