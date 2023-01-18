#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import os
import time
import numpy as np
import whisper
import torch
import torch.multiprocessing as mp

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
from inference import espnet_inference, whisper_inference
from make_dataset import inference_dataset, aihub_dataset
from utils import SplitWavAudioMubin, change_sampling_rate_file_path


ROOT_PATH = '/opt/ml'
IGNORE_FOLDERS = ['.DS_Store']
warnings.filterwarnings(action='ignore', category=UserWarning)


def main(
        filepath, 
        min_per_split=None, 
        min_silence_len=None,
        stt='espnet'
    ):
    cfg = OmegaConf.load('./conf.yaml')

    filename = change_sampling_rate_file_path(filepath, resampling_sr=cfg.default.resampling_sr)
    folder = f'./download/{filename}'
    file = f'{filename}.wav'
    
    start_time = time.time()
    split_wav = SplitWavAudioMubin(folder, file)

    # split step
    if min_per_split == None:
        # wav 파일 전체를 침묵 기준으로 분리
        split_wav.single_silent_split(
            filepath=os.path.join(folder, file), 
            min_silence_len=min_silence_len
        )      
    else:
        split_wav.multiple_split(min_per_split=min_per_split)   # 1분 단위로 split

    print(f'num process : {cfg.default.num_process}')
    scps = split_wav.make_split_scp_file(split=cfg.default.num_process)
    print('split complete')
    
    # parameter setting
    kwargs = dict(getattr(cfg, stt))
    
    assert stt in ['whisper', 'espnet'], "choose 'whisper', 'espnet'"
    if stt == 'whisper':
        inference = whisper_inference
    elif stt == 'espnet':
        d = ModelDownloader('.cache/espnet')
        o = d.download_and_unpack(kwargs['mdl'])

        # argument
        parser = get_parser()
        args = parser.parse_args()
        kwargs.update(vars(args))
        kwargs.update(o)
        kwargs.update({'ngpu': 1 if torch.cuda.device_count() > 0 else 0})
        
        # unused argument
        kwargs.pop('mdl_file')
        kwargs.pop('wav_scp')
        kwargs.pop('mdl')
        kwargs.pop('config')
        
        inference = espnet_inference

    output_dir = f'./output/{filename}/{filename}'
    if len(scps) > 1:
        processes = []
        for i, scp in enumerate(scps):
            kwargs.update({
                'data_path_and_name_and_type': [(scp, 'speech', 'sound')],
                'output_dir': output_dir + f'{i}',
            })

            process = Process(
                target=inference,
                kwargs=kwargs
            )

            process.start()
            print(f"process {i} start")
            processes.append(process)
            time.sleep(0.5)

        for process in processes:
            process.join()
    else:
        kwargs.update({
                'data_path_and_name_and_type': [(scps[0], 'speech', 'sound')],
                'output_dir': output_dir,
            })
        inference(**kwargs)

    print("-"*50)
    print(f"\n\nend(sec) : {time.time()-start_time:.2f}")
    print("-"*50)
    
    return filename

def dev(stage='train', stt='whisper'):
    start_time = time.time()
    cfg = OmegaConf.load('./conf.yaml')

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
                    all_scps.append(split_wav.make_split_scp_file(split=cfg.default.num_process))

    # parameter setting
    kwargs = dict(getattr(cfg, stt))
    
    assert stt in ['whisper', 'espnet'], "choose 'whisper', 'espnet'"
    if stt == 'whisper':
        model_size = cfg.whisper.model_size
        
        if not os.path.exists(f'{ROOT_PATH}/.cache/whisper/{model_size}.pt'): 
            whisper.load_model(model_size)
        inference = whisper_inference
    elif stt == 'espnet':
        d = ModelDownloader('.cache/espnet')
        o = d.download_and_unpack(kwargs['mdl'])

        # argument
        parser = get_parser()
        args = parser.parse_args()
        kwargs.update(vars(args))
        kwargs.update(o)
        kwargs.update({'ngpu': 1 if torch.cuda.device_count() > 0 else 0})
        
        # unused argument
        kwargs.pop('mdl_file')
        kwargs.pop('wav_scp')
        kwargs.pop('mdl')
        kwargs.pop('config')
        
        inference = espnet_inference

    processes = []
    for scps in all_scps:
        if len(scps) > 1:
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
        else:
            scp_split = scps[0].split('/')
            output_dir = os.path.join(
                'output', *scp_split[1:-1], scp_split[-2]
            )
            kwargs.update({
                'data_path_and_name_and_type': [(scps[0], 'speech', 'sound')],
                'output_dir': output_dir
            })
            inference(**kwargs)

    print("-"*50)
    print(f"\n\nend(sec) : {time.time()-start_time:.2f}")
    print("-"*50)


if __name__ == '__main__':
    # filename = main(
    #     filepath="./[#한국사능력검정] 설민석 – 10분 순삭! 한 번에 정리되는 일제강점기!.wav", 
    #     min_per_split=0.25, 
    #     min_silence_len=None,
    #     stt='whisper'
    # )
    # inference_dataset(filename)
    
    # dev(stage='sample')     # stage : 'train', 'validation', 'sample'
    aihub_dataset(stage='train')