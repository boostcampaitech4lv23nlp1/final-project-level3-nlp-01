import os
import torch
import whisper
import asyncio

from typing import Optional, Tuple, Sequence
from transformers import pipeline

class Inference():
    def __init__(
            self,
            processor,
            forced_decoder_ids,
            model,
            output_dir: str,
            fp16: bool,
            beam_size: int,
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
            language: Optional[str],
        ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.processor = processor
        self.forced_decoder_ids = forced_decoder_ids
        self.model = model

        self.output_dir = output_dir
        self.scp_path, _, _ = data_path_and_name_and_type[0]

    async def run(self):
        print('whisper inference')
        with open(self.scp_path, 'r+') as f:
            lines = f.readlines()

        output = []
        for line in lines:
            i, path = line.strip().split(' ')

            # TODO: apt-get install ffmpeg README.md 파일에 명시하기
            audio = whisper.load_audio(path)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(self.device)
            
            predicted_ids = self.model.generate(inputs, forced_decoder_ids=self.forced_decoder_ids)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            print(i, ' ', transcription)
            output.append(" ".join([i, transcription]) + '\n')
    
        # make directory
        os.makedirs(path:=os.path.join(self.output_dir, '1best_recog'), exist_ok=True)
        with open(os.path.join(path, 'text'), 'w+') as f:
            for s in output:
                f.write(s)
            f.close()

        print(f"end {self.scp_path}")
        
    def process(self):
        asyncio.run(self.run())