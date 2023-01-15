#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import os
import time
import numpy as np
import shutil
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


IGNORE_FOLDERS = ['.DS_Store']
warnings.filterwarnings(action='ignore', category=UserWarning)
# warnings.filterwarnings(action='ignore', category=FutureWarning)

class SplitWavAudioMubin():
    def __init__(self, 
            folder: Optional[str], 
            filename:  Optional[str]
        ):
        self.folder = folder
        self.filename = filename

        self.split_dir = '/split'
        self.scp_texts = []
        self.count = 0

        try:
            self.audio = AudioSegment.from_wav(self.folder + '/' + self.filename)
        except TypeError as type_error:
            self.audio = AudioSegment.empty()
    
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        
        # metadata
        path = os.path.join(self.folder, 'normal_split', self.filename[:-4]+str(self.count).rjust(2, '0')+self.filename[-4:])[2:]
        split_audio.export(path, format="wav")

        name = path.split('/')[-1][:-4]
        self.scp_texts.append(" ".join([name, path]) + '\n')
        self.count += 1
        return path
    
    def multiple_split(self, min_per_split):
        os.makedirs(os.path.join(self.folder, 'normal_split'), exist_ok=True)
        total_mins = math.ceil(self.get_duration() / 60)
        paths = []
        for i in np.arange(0, total_mins, min_per_split):
            paths.append(self.single_split(i, i+min_per_split))
            if i == total_mins - min_per_split:
                print('All splited successfully')
        return paths

    def single_silent_split(self, filepath, min_silence_len=1000, silence_thresh=-32):
        audio = AudioSegment.from_wav(filepath)
        os.makedirs(os.path.join(self.folder, 'silent_split'), exist_ok=True)
        nonsilence_range = silence.detect_nonsilent(audio, min_silence_len, silence_thresh, seek_step=100)

        chunks = []
        for i, chunk in enumerate(nonsilence_range):
            path = os.path.join(
                self.folder, 'silent_split' ,self.filename[:-4]+str(self.count).rjust(2, '0')+self.filename[-4:]
            )[2:]
            if i == 0:
                start = chunk[0]
                end = (chunk[1] + nonsilence_range[i+1][0]) / 2 
                audio[:end].export(path, format='wav')
            elif i == len(nonsilence_range)-1:
                start = (nonsilence_range[i-1][1] + chunk[0]) / 2
                end = chunk[1] + 1000.
                audio[start:].export(path, format='wav')
            else:
                start = (nonsilence_range[i-1][1] + chunk[0]) / 2
                end = (chunk[1] + nonsilence_range[i+1][0]) / 2
                audio[start:end].export(path, format='wav')

            start = round(start/1000, 1)
            end = round(end/1000, 1)
            chunks.append((start, end))

            # save metadata
            name = path.split('/')[-1][:-4]
            self.scp_texts.append(" ".join([name, path]) + '\n')
            self.count += 1
              
    
    def multiple_silent_split(self, min_per_split):
        paths = self.multiple_split(min_per_split=min_per_split)
        self.scp_texts.clear()
        self.count = 0
        
        os.makedirs(os.path.join(self.folder, 'silent_split'), exist_ok=True)
        for path in paths:
            self.single_silent_split(path)
        shutil.rmtree(os.path.join(self.folder, 'normal_split'))
        print("silent splited successfully")

    def make_scp_file(self):
        with open(os.path.join(self.folder, 'wav.scp'), 'w+') as f:
            f.writelines(self.scp_texts)

    def make_split_scp_file(self, split=4):
        div = math.ceil(len(self.scp_texts) / split)

        left, right = 0, div
        idx = 0
        scps = []
        while True:
            scp = os.path.join(self.folder, f'wav{idx}.scp')
            if len(self.scp_texts[left:right]) > 0:
                with open(scp, 'w+') as f:
                    f.writelines(
                        self.scp_texts[left:right] 
                        if right >= 0 else self.scp_texts[left:]
                    )
                scps.append(scp)
            
            if right < 0 or split == 1: break

            left = right
            if right + div < len(self.scp_texts):
                right += div
            else:
                right = -1
            idx += 1
        return scps


# not splited wav file
def change_sampling_rate_file_path(file_path, resample_sr=16000):
    filename = file_path.split('/')[-1].split('.')[0]

    if not os.path.exists(f'./download/{filename}/{filename}.wav'):    
        data, sr = librosa.load(file_path, sr=48000)
        resample = librosa.resample(data, sr, resample_sr)

        os.makedirs(f'./download/{filename}', exist_ok=True)
        sf.write(f'./download/{filename}/{filename}.wav', resample, resample_sr, format='WAV')
    return filename


def main(filepath):
    start_time = time.time()
    cfg = OmegaConf.load('./decode_conf.yaml')

    filename = change_sampling_rate_file_path(filepath)
    folder = f'./download/{filename}'
    file = f'{filename}.wav'
    
    split_wav = SplitWavAudioMubin(folder, file)
    # split_wav.multiple_split(min_per_split=1)                     # 1분 단위 split
    split_wav.single_silent_split(os.path.join(folder, file))       # wav 파일 전체를 침묵 기준으로 분리
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
    args.output_dir = f'./output/{filename}'

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

def change_sampling_rate_file_paths(file_paths, resampling_sr=16000):
    no_error_file_paths = []
    for file_path in file_paths:
        try:
            data, sr = librosa.load(file_path)
            resample = librosa.resample(data, sr, resampling_sr)
            sf.write(file_path, resample, resampling_sr, format='WAV')
            no_error_file_paths.append(file_path)
        except EOFError as e:
            print(f"error occurred - {file_path}")
    return no_error_file_paths

def dev():
    start_time = time.time()
    cfg = OmegaConf.load('./decode_conf.yaml')

    label_path = os.path.join('dataset', 'train', 'raw_data')
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
    # args.output_dir = f'./output/{filename}'

    kwargs = vars(args)
    kwargs.update(o)

    del args.mdl_file
    del args.wav_scp    
    kwargs.pop('config', None)

    # output_dir = args.output_dir
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

def make_new_dataset(output_file_paths: str, labeled_file_paths: str):
    # 1. create dictionary and add labeled data
    output = defaultdict(list)
    idx = labeled_file_paths[0].split('/')[-5]      # KlecSpeech_train_D12_label_0
    folder = labeled_file_paths[0].split('/')[-2]   # S001007
    print(labeled_file_paths)
    print('\n\n')
    for labeled_file_path in labeled_file_paths:
        i = labeled_file_path.split('/')[-1].split('.')[0]
        extension = labeled_file_path.split('/')[-1].split('.')[1]

        domain = labeled_file_path.split('/')[-4]
        subdomain = labeled_file_path.split('/')[-3]

        # json 파일 제외
        if extension == 'txt':
            with open(labeled_file_path, 'r+') as f:
                line = f.readline().strip()

            # step  (그러니까)/(긍게*) 짧으니까 쓰기가 쉽죠.
            for match in set(re.findall(r'\([^)]*\)[\s]*[/][\s]*\([^)]*\)', line)):
                repl = match.split('/')[0][1:-1]
                line = line.replace(match, repl)
            
            # step 이 작품이 쓰여진 건 (60)/)육십) 년대 와서야 이루어졌습니다. 왜냐하면 전쟁 때는,
            for match in set(re.findall(r'\([^)]*\).*\)', line)):
                repl = re.findall(r'\([^)]*\)', match)[0][1:-1]
                line = line.replace(match, repl)
            
            # step  (그러니까)(긍게*) 짧으니까 쓰기가 쉽죠.
            for match in set(re.findall(r'\([^)]*\)[^)]*\)', line)):
                repl = re.findall(r'\([a-zA-Z가-힣]+\)', line)[0][1:-1]
                line = line.replace(match, repl)
            
            # step . change '[kr]/' -> [kr] 
            for match in set(re.findall(r'[가-힣][/]', line)):
                repl = match[:-1]
                line = line.replace(match, repl)

            # step . change '[en]/' -> ''
            for match in set(re.findall(r'[a-zA-Z][/]', line)):
                repl = ''
                line = line.replace(match, repl)

            # step . change '[kr, en]+' -> ''
            for match in set(re.findall(r'[a-zA-Z가-힣]{1}[+]\s', line)):
                repl = ''
                line = line.replace(match, repl)

            # step . change '[kr, en]*' -> ''
            repl = ''
            line = line.replace('*', repl)

            # step 4. @/ -> ''
            line = re.sub(r'[@][/]', '', line)
            
            # step 4. /, ／ -> ''
            line = line.replace('/', '')
            line = line.replace('／', '')
            
            # step 7. '  ' -> ' '
            repl = ''
            line = line.replace('  ', ' ')

            # step 8. '+' -> ' '
            repl = ' '
            line = line.replace('+', ' ')
            line = line.replace('＋', ' ')

            # step
            line = line.replace('ｌ', '')

            # key : KlecSpeech_train_D12_label_0-D12-G02-S001007-000625
            key = idx + '-' + domain + '-' + subdomain + '-' + folder + '-' + i
            output[key].append(line.strip())

    # 2. add output data in dictionary
    for output_file_path in output_file_paths:
        _, domain, subdomain = output_file_path.split('/')[3:6]
        with open(output_file_path, 'r+') as f:
            lines = f.readlines()
        for line in lines:
            line_split = line.split(' ') 
            i, line = line_split[0], " ".join(line_split[1:])

            key = idx + '-' + domain + '-' + subdomain + '-' + folder + '-' + i
            output[key].append(line.strip())
    
    # 3. add null values
    for key, value in output.items():
        if len(value) < 2:
            output[key].append('')
    df = pd.DataFrame(output).T
    df.rename(columns={0: 'label', 1: 'output'}, inplace=True)
    return df


def dataset():
    label_path = os.path.join('dataset', 'train', 'labeled_data')
    output_path = os.path.join('output', 'train', 'raw_data')

    dfs = pd.DataFrame({'label': [], 'output': []})
    for idx in sorted(os.listdir(output_path)):
        for domain in sorted(os.listdir(os.path.join(output_path, idx))):
            for subdomain in sorted(os.listdir(os.path.join(output_path, idx, domain))):
                for directory in sorted(os.listdir(os.path.join(output_path, idx, domain, subdomain))):

                    output_folder_path = os.path.join(output_path, idx, domain, subdomain, directory)
                    output_file_paths = sorted([
                        os.path.join(output_folder_path, file, '1best_recog', 'text')
                        for file in os.listdir(output_folder_path)
                    ])

                    labeled_idx = idx.split('_')
                    labeled_idx[-2] = 'label'
                    labeled_idx = "_".join(labeled_idx)
                    labeled_folder_path = os.path.join(label_path, labeled_idx ,domain, subdomain, directory)
                    labeled_file_paths = sorted([
                        os.path.join(labeled_folder_path, file)
                        for file in os.listdir(labeled_folder_path)
                    ])
                    df = make_new_dataset(output_file_paths, labeled_file_paths)
                    dfs = pd.concat([dfs, df])
    dfs.to_csv('./dataset.csv', encoding='utf-8-sig')

def test():
    cfg = OmegaConf.load('./decode_conf.yaml')
    d = ModelDownloader('.cache/espnet')
    o = d.download_and_unpack(cfg.mdl)
    pass

if __name__ == '__main__':
    main(filepath='./bert.wav')
    # dev()
    # dataset()
    # test()