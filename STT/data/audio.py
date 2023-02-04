from typing import Optional, Union
from pydub import AudioSegment, silence

import os
import math
import numpy as np
import pandas as pd
import shutil
import soundfile as sf


class SplitWavAudio():
    '''
    ## SplitWavAudio
    description:
    - 음성 처리와 관련된 메소드들이 정의되어있습니다.
    
    args:
    - folder:
        - values
            - str: inference할 wav파일을 사용할 경우, 해당 파일이 존재하는 폴더의 위치.
            - None: GPT 모델 학습에 사용할 데이터셋을 제작할 시에 사용.
    - filename:
        - values
            - str: inference할 wav파일을 사용할 경우, wav 파일의 이름.
            - None: GPT 모델 학습에 사용할 데이터셋을 제작할 시에 사용.
    '''
    def __init__(self, 
            folder: Optional[str], 
            filename:  Optional[str]
        ) -> None:
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
        path = os.path.join(
            self.folder, 
            'normal_split', 
            self.filename[:-4]+'-'+str(self.count).rjust(2, '0')+self.filename[-4:]
        )[2:]
        split_audio.export(path, format="wav")

        name = path.split('/')[-1][:-4]
        self.scp_texts.append(" ".join([name, path]) + '\n')
        self.count += 1
        return path
    
    def multiple_split(self, min_per_split):
        '''
        description:
            min_per_split 분 단위로 쪼개는 작업을 수행하는 함수
        '''
        os.makedirs(os.path.join(self.folder, 'normal_split'), exist_ok=True)
        total_mins = math.ceil(self.get_duration() / 60)
        paths = []
        for i in np.arange(0, total_mins, min_per_split):
            paths.append(self.single_split(i, i+min_per_split))
            if i == total_mins - min_per_split:
                print('All splited successfully')
        return paths

    def multiple_silent_split(self, min_per_split, min_silence_len):
        '''
        description:
            먼저 min_per_split 분 단위로 쪼갠 후,
            쪼개진 각각의 wav 파일에서 min_silence_len 기준으로 추가로 분리
        '''
        paths = self.multiple_split(min_per_split=min_per_split)
        self.scp_texts.clear()
        self.count = 0
        
        os.makedirs(os.path.join(self.folder, 'silent_split'), exist_ok=True)
        for path in paths:
            self.single_silent_split(path, min_silence_len=min_silence_len)
        shutil.rmtree(os.path.join(self.folder, 'normal_split'))
        print("silent splited successfully")

    def single_silent_split(self, filepath, min_silence_len=1000):
        '''
        description:
            wav 파일 하나에 대해 min_silence 기준으로 silence split 진행하는 함수
        '''
        audio = AudioSegment.from_wav(filepath)
        os.makedirs(os.path.join(self.folder, 'silent_split'), exist_ok=True)

        chunks = silence.split_on_silence(
            audio, 
            min_silence_len=min_silence_len,            # split on silence longer than min_silence_len(ms)
            silence_thresh=audio.dBFS*1.9,             # min dBPS 
            seek_step=50
        )

        target_length = 5 * 1000                        # ms
        output_chunks = [chunks[0]]
        for chunk in chunks[1:]:
            
            if len(output_chunks[-1]) < target_length:
                output_chunks[-1] += chunk
            else:
                output_chunks.append(chunk)

        for output_chunk in output_chunks:
            # save splited wav file
            path = os.path.join(
                self.folder, 'silent_split' ,self.filename[:-4]+'-'+str(self.count).rjust(2, '0')+self.filename[-4:]
            )[2:]
            output_chunk.export(path, format='wav')

            # save metadata
            name = path.split('/')[-1][:-4]
            self.scp_texts.append(" ".join([name, path]) + '\n')
            self.count += 1
              
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

def change_sampling_rate(
        file_path: str, 
        resampling_sr: Optional[Union[int, str]]=None
    ):
    '''
    ## change_sampling_rate()
    description:
    - 파일의 sampling rate를 변경해주는 함수입니다.
    
    args:
    - file_path: sampling rate를 변경하고자 하는 파일의 위치를 받습니다.
    - resampling_sr: 변경하고자 하는 sampling rate를 입력합니다.
        - values
            - int : 변경하고자 하는 sampling rate. 음수일 경우 'None'으로 처리.
            - None or 'None' : sampling rate를 변경하지않음.
    '''

    filename = file_path.split('/')[-1].split('.')[0].replace(' ', '')
    filename = filename.replace('–', '_').replace('-', '_')
    new_path = os.path.join('output/STT', 'download', filename)
    if os.path.exists(new_path):
        shutil.rmtree(new_path)

    os.makedirs(new_path, exist_ok=True)
    
    if type(resampling_sr) != str:
        if resampling_sr < 0 :
            resampling_sr = 'None'
        if resampling_sr == None:
            resampling_sr = 'None'

    output_path = os.path.join(new_path, filename+'.wav')
    if resampling_sr != 'None':
        sound = AudioSegment.from_wav(file_path)
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        sound.export(output_path, format='wav')
    else:
        sound = AudioSegment.from_wav(file_path)
        sound = sound.set_channels(1)
        sound.export(output_path, format='wav')
    return filename