import torch
from STT.setup import stt_setup


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')     # multiprocess mode
    output = stt_setup(
        make_dataset=False, 
        inference_wav_file='./[심화별개념5]_ 2-1 구석기_신석기 시대_2강 선사시대.wav'
    )
    