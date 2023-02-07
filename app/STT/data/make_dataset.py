import os
import time
import warnings
import whisper
import torch
import asyncio
from concurrent.futures import ProcessPoolExecutor

from omegaconf import OmegaConf
from tqdm import tqdm

from .audio import SplitWavAudio, change_sampling_rate
from ..inference.inference import Inference

from transformers import AutoProcessor
from multiprocessing import Process, set_start_method

HOMEPATH = '/opt/ml'
IGNORE_FOLDERS = ['.DS_Store']
warnings.filterwarnings(action='ignore', category=UserWarning)

# TODO: GPT2에 학습할 데이터셋 생성 모듈 제작 -> MakeDatasetUsingAIHUB()
class MakeDatasetUsingAIHUB(object):
    def __init__(self, model) -> None:
        self.model = model
        self.cfg = OmegaConf.load('./STT/data/conf.yaml')        
        self.split_wav = SplitWavAudio(folder=None, filename=None)
        self.all_scps = []

    def __call__(
        self,
        download_dir: str='./STT/dataset/train'
    ):
        raw_data_path = os.path.join(download_dir, 'raw_data')
        self.all_scps = []
        num_process = self.cfg.default.num_process
        for idx in [t for t in os.listdir(raw_data_path) if t not in IGNORE_FOLDERS]:
            for domain in [t for t in os.listdir(os.path.join(raw_data_path, idx)) if t not in IGNORE_FOLDERS]:
                for subdomain in [t for t in os.listdir(os.path.join(raw_data_path, idx, domain)) if t not in IGNORE_FOLDERS]:      
                    for directory in tqdm(tl:=[t for t in os.listdir(os.path.join(raw_data_path, idx, domain, subdomain)) if t not in IGNORE_FOLDERS], desc=subdomain, total=len(tl)):
                        folder_path = os.path.join(raw_data_path, idx, domain, subdomain, directory)                            # folder
                        file_paths = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.split('.')[1] == 'wav'])     # files

                        self.split_wav.scp_texts.clear()
                        for file_path in file_paths:
                            name = file_path.split('/')[-1][:-4]
                            self.split_wav.scp_texts.append(" ".join([name, file_path]) + '\n')
                        
                        self.split_wav.folder = folder_path
                        self.all_scps.append(self.split_wav.make_split_scp_file(split=num_process))

        # parameter setting
        kwargs = dict(getattr(self.cfg, 'whisper'))
        
        processes = []
        model = self.model
        for scps in self.all_scps:
            if len(scps) > 1:
                model.share_memory()
                inferences = []
                for i, scp in enumerate(scps):
                    scp_split = scp.split('/')
                    output_dir = os.path.join('output', *scp_split[1:-1], scp_split[-2])
                    kwargs.update({
                        'model': model,
                        'data_path_and_name_and_type': [(scp, 'speech', 'sound')],
                        'output_dir': output_dir + f'_{i}'
                    })
                    inferences.append(Inference(**kwargs))

                for inference in inferences:
                    process = Process(target=inference)
                    process.start()
                    processes.append(process)
                    time.sleep(0.1)
                
                for process in processes:
                    process.join()
                processes.clear()
            else:
                scp_split = scps[0].split('/')
                output_dir = os.path.join(
                    'output', *scp_split[1:-1], scp_split[-2]
                )
                kwargs.update({
                    'model': model,
                    'data_path_and_name_and_type': [(scps[0], 'speech', 'sound')],
                    'output_dir': output_dir
                })
                inference = Inference(**kwargs)
                inference()
        
class MakeInferenceDataset(object):
    def __init__(self, model, inference_wav_path: str) -> None:
        self.model = model
        self.cfg = OmegaConf.load('/opt/ml/level3_productserving-level3-nlp-01/app/STT/data/conf.yaml') # original : ./STT/data/conf.yaml
        resampling_sr = self.cfg.default.resampling_sr

        self.filename = change_sampling_rate(file_path=inference_wav_path, resampling_sr=resampling_sr)
        self.folder = f'./output/STT/download/{self.filename}'
        self.file = f'{self.filename}.wav'
        self.split_wav = SplitWavAudio(self.folder, self.file)
        
    def __call__(self, min_per_split=None, min_silence_len=None) -> str:
        if min_per_split is None:
            # wav 파일 전체에 대해서 slience 기준으로 분리
            self.split_wav.single_silent_split(
                filepath=os.path.join(self.folder, self.file),
                min_silence_len=min_silence_len
            )
        else:
            # min_per_split 분 단위로 split
            self.split_wav.multiple_split(min_per_split=min_per_split)

        start_time = time.time()
        num_process = self.cfg.default.num_process
        batch_size = self.cfg.default.batch_size

        print(f'num process : {num_process}')
        print(f'batch_size : {batch_size}')
        scps = self.split_wav.make_split_scp_file(split=num_process)

        # parameter setting
        kwargs = dict(getattr(self.cfg, 'whisper'))
        model_size = kwargs.pop('model_size')

        # set model init
        model_checkpoint = f'openai/whisper-{model_size}'
        processor = AutoProcessor.from_pretrained(model_checkpoint)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language='korean', task='transcribe')
        model = self.model  

        output_dir = f'./output/STT/{self.filename}/{self.filename}'

        processes = []
        if len(scps) > 1:
            model.share_memory()
            inferences = []
            for i, scp in enumerate(scps):
                kwargs.update({
                    'processor': processor,
                    'forced_decoder_ids': forced_decoder_ids,
                    'model': model,
                    'batch_size': batch_size,
                    'data_path_and_name_and_type': [(scp, 'speech', 'sound')],
                    'output_dir': output_dir + f'_{i}'
                })
                inferences.append(Inference(**kwargs))

            
            for inference in inferences:
                process = Process(target=inference)
                process.start()
                processes.append(process)
            
            # 모든 프로세스가 끝날 때 까지 wait 합니다.
            for process in processes:
                process.join()
            processes.clear()
        else:
            kwargs.update({
                    'processor': processor,
                    'forced_decoder_ids': forced_decoder_ids,
                    'model': model,
                    'batch_size': batch_size,
                    'data_path_and_name_and_type': [(scps[0], 'speech', 'sound')],
                    'output_dir': output_dir,
                })
            
            inference = Inference(**kwargs)
            inference()

        print("-"*50)
        print(f"end(sec) : {time.time()-start_time:.2f}")
        print("-"*50)
        
        return self.filename

    def process(self, min_per_split=None, min_silence_len=None):
        asyncio.run(self.run(min_per_split, min_silence_len))
        return self.filename