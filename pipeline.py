import torch
from STT.setup import stt_setup
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')     # multiprocess mode
    output = stt_setup(
        make_dataset=False, 
        inference_wav_file='./[심화별개념5]_2-1구석기_신석기시대_2강선사시대.wav'
    )