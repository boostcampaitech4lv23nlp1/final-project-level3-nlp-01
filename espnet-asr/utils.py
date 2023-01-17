from typing import Optional
from pydub import AudioSegment, silence

import os
import math
import numpy as np
import pandas as pd
import shutil
import librosa
import soundfile as sf


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
                self.folder, 'silent_split' ,self.filename[:-4]+'-'+str(self.count).rjust(2, '0')+self.filename[-4:]
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
              
    
    def multiple_silent_split(self, min_per_split, min_silence_len):
        paths = self.multiple_split(min_per_split=min_per_split)
        self.scp_texts.clear()
        self.count = 0
        
        os.makedirs(os.path.join(self.folder, 'silent_split'), exist_ok=True)
        for path in paths:
            self.single_silent_split(path, min_silence_len=min_silence_len)
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

def change_sampling_rate_file_path(
        file_path, 
        resampling_sr: Optional[int]=16000
    ):
    filename = file_path.split('/')[-1].split('.')[0].replace(' ', '')
    new_path = os.path.join('download', filename)
    os.makedirs(new_path, exist_ok=True)
    
    if resampling_sr is not None:
        data, sr = librosa.load(file_path)
        resample = librosa.resample(data, sr, resampling_sr)
        sf.write(os.path.join(new_path, filename+'.wav'), resample, resampling_sr, format='WAV')
    else:
        shutil.copy(file_path, os.path.join(new_path, filename+'.wav'))

    return filename