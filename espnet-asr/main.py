#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import os
import time
import numpy as np

import warnings
import re


from collections import defaultdict
from pydub import AudioSegment, silence
from tqdm import tqdm
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
from hanspell import spell_checker


from speech2text import Speech2Text
from inference import inference
from make_dataset import inference_dataset, aihub_dataset
from utils import SplitWavAudioMubin, change_sampling_rate_file_path


IGNORE_FOLDERS = ['.DS_Store']
warnings.filterwarnings(action='ignore', category=UserWarning)
# warnings.filterwarnings(action='ignore', category=FutureWarning)


def main(filepath):
    start_time = time.time()
    cfg = OmegaConf.load('./decode_conf.yaml')

    filename = change_sampling_rate_file_path(filepath)
    folder = f'./download/{filename}'
    file = f'{filename}.wav'
    
    split_wav = SplitWavAudioMubin(folder, file)
    # split_wav.multiple_split(min_per_split=1)                                         # 1분 단위 split
    split_wav.single_silent_split(os.path.join(folder, file), min_silence_len=500)      # wav 파일 전체를 침묵 기준으로 분리
    # split_wav.multiple_silent_split(min_per_split=2)

    print(f'num process : {cfg.num_process}')
    
    # split_wav.make_scp_file()
    scps = split_wav.make_split_scp_file(split=cfg.num_process)
    
    parser = get_parser()
    args = parser.parse_args()
    
    d = ModelDownloader('.cache/espnet')
    o = d.download_and_unpack(cfg.mdl)
    args.config = cfg.config
    args.ngpu = cfg.ngpu
    args.batch_size = cfg.batch_size
    args.output_dir = f'./output/{filename}/{filename}'

    kwargs = vars(args)
    kwargs.update(o)
    
    del args.mdl_file
    del args.wav_scp
    kwargs.pop('config', None)

    output_dir = args.output_dir
    if len(scps) > 1:
        processes = []
        for i, scp in enumerate(scps):
            kwargs.update({
                'data_path_and_name_and_type': [(scp, 'speech', 'sound')],
                'output_dir': output_dir + f'{i}'
            })
            process = Process(
                target=inference,
                kwargs=kwargs
            )

            process.start()
            processes.append(process)
            time.sleep(0.1)

        for process in processes:
            process.join()
    else:
        kwargs.update({
                'data_path_and_name_and_type': [(scps[0], 'speech', 'sound')],
                'output_dir': args.output_dir
            })
        inference(**kwargs)

    print("-"*50)
    print(f"\n\nend(sec) : {time.time()-start_time:.2f}")
    print("-"*50)



def dev(stage='train'):
    start_time = time.time()
    cfg = OmegaConf.load('./decode_conf.yaml')

    label_path = os.path.join('dataset', stage, 'raw_data')
    split_wav = SplitWavAudioMubin(
        folder=None,
        filename=None
    )

    all_scps = []
    for idx in [t for t in os.listdir(label_path) if t not in IGNORE_FOLDERS]:
        for domain in [t for t in os.listdir(os.path.join(label_path, idx)) if t not in IGNORE_FOLDERS]:
            for subdomain in [t for t in os.listdir(os.path.join(label_path, idx, domain)) if t not in IGNORE_FOLDERS]:      
                for directory in tqdm(tl:=[t for t in os.listdir(os.path.join(label_path, idx, domain, subdomain)) if t not in IGNORE_FOLDERS], desc=subdomain, total=len(tl)):
                    folder_path = os.path.join(label_path, idx, domain, subdomain, directory)                            # folder
                    file_paths = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.split('.')[1] == 'wav'])     # files

                    # resampling wav files (? -> 16000)
                    # file_paths = change_sampling_rate_file_paths(file_paths)

                    split_wav.scp_texts.clear()
                    for file_path in file_paths:
                        name = file_path.split('/')[-1][:-4]
                        split_wav.scp_texts.append(" ".join([name, file_path]) + '\n')
                    
                    split_wav.folder = folder_path
                    all_scps.append(split_wav.make_split_scp_file(split=cfg.num_process))

    parser = get_parser()
    args = parser.parse_args()

    d = ModelDownloader('.cache/espnet')
    o = d.download_and_unpack(cfg.mdl)
    args.config = cfg.config
    args.ngpu = cfg.ngpu
    args.batch_size = cfg.batch_size
    args.beam_size = cfg.beam_size

    kwargs = vars(args)
    kwargs.update(o)

    del args.mdl_file
    del args.wav_scp    
    kwargs.pop('config', None)

    if cfg.num_process > 1:
        processes = []
        for scps in all_scps:
            for i, scp in enumerate(scps):
                scp_split = scp.split('/')
                output_dir = os.path.join(
                    'output', *scp_split[1:-1], scp_split[-2]
                )
                kwargs.update({
                    'data_path_and_name_and_type': [(scp, 'speech', 'sound')],
                    'output_dir': output_dir + f'_{i}'
                })
                process = Process(
                    target=inference,
                    kwargs=kwargs
                )

                process.start()
                processes.append(process)
                time.sleep(0.1)     # sleep
            
            for process in processes:
                process.join()

    print("-"*50)
    print(f"\n\nend(sec) : {time.time()-start_time:.2f}")
    print("-"*50)


if __name__ == '__main__':
    main(filepath="./김상욱 교수님의 '양자역학' 강의.wav")
    # dev(stage='validation')
    
    # aihub_dataset(stage='validation')
    inference_dataset("김상욱교수님의'양자역학'강의")