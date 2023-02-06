import pandas as pd

from .data.make_dataset import MakeDatasetUsingAIHUB, MakeInferenceDataset
from .data.utils.output_to_dataframe import aihub_dataset, inference_dataset
from typing import Optional

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# TODO: GPT2에 학습할 데이터셋 생성 모듈 제작 -> make_dataset()
# TODO: GPT2에 넘겨줄 inference 모듈 제작 -> inference()

def stt_setup(model, make_dataset: bool, download_dir='./STT/dataset/train', **kwargs) -> Optional[pd.DataFrame]:
    '''
    ## stt_setup
    description:
    - stt 작업을 시작하는 함수입니다.
    
    args:
    - make_dataset:
        - '참'일 경우 aihub 한국어 강의 음성 데이터셋에 대해 STT를 진행.
        - '거짓'일 경우 wav 파일 1개에 대해 STT를 진행.
    - download_dir:
        - make_dataset이 '참'일 경우 음성 데이터셋의 train 폴더 위치를 지정.
        - make_dataset이 '거짓'일 경우 사용하지 않음
    - kwargs:
        - inference_wav_file:
            - STT를 진행할 wav 파일 위치를 지정.
    '''
    if make_dataset is True:
        MakeDatasetUsingAIHUB(model=model)(download_dir=download_dir)
        aihub_dataset(stage='train')
        return None
    else:
        assert kwargs['inference_wav_file'], "you must input inference wav file's path."
        inference_wav_file = kwargs['inference_wav_file']
        filename = MakeInferenceDataset(model=model, inference_wav_path=inference_wav_file)(
            min_per_split=None,
            min_silence_len=500,
        )
        
        df = inference_dataset(filename=filename)
        return df
