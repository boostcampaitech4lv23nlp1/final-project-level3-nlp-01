import os
import torch
import whisper
import asyncio
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from typing import Optional, Tuple, Sequence
from transformers import pipeline

class Inference():
    def __init__(
            self,
            processor,
            forced_decoder_ids,
            model,
            output_dir: str,
            beam_size: int,
            batch_size: int,
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
        ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.processor = processor
        self.forced_decoder_ids = forced_decoder_ids
        self.model = model

        self.beam_size = beam_size

        self.output_dir = output_dir
        self.scp_path, _, _ = data_path_and_name_and_type[0]

        self.batch_size = batch_size

    def __call__(self):
        print('whisper inference')
        with open(self.scp_path, 'r+') as f:
            lines = f.readlines()

        output = []
        batch_metadatas = []
        for line in lines:
            i, path = line.strip().split(' ')
            if len(batch_metadatas) < self.batch_size:
                batch_metadatas.append((i, path))
            else:
                datas = [whisper.load_audio(path) for _, path in batch_metadatas]

                input_features = self.processor.feature_extractor(
                    datas,
                    padding='max_length',
                    max_length=480_000,
                    sampling_rate=16000,
                    return_tensors='pt'
                ).input_features.to(self.device)
                
                predicted_idss = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)
                transcriptions = self.processor.batch_decode(predicted_idss, skip_special_tokens=True)
                
                for batch_metadata, transcription in zip(batch_metadatas, transcriptions):
                    _i, _ = batch_metadata
                    output.append(result:=" ".join([_i, transcription]) + '\n')
                    print(result)
                batch_metadatas.clear()
                batch_metadatas.append((i, path))
        
        if len(batch_metadatas) > 0:
            datas = [whisper.load_audio(path) for _, path in batch_metadatas]

            input_features = self.processor.feature_extractor(
                datas,
                sampling_rate=16000,
                return_tensors='pt'
            ).input_features.to(self.device)
            
            predicted_idss = self.model.generate(
                input_features, 
                forced_decoder_ids=self.forced_decoder_ids,
                num_beams=self.beam_size
            )
            transcriptions = self.processor.batch_decode(predicted_idss, skip_special_tokens=True)
            
            for batch_metadata, transcription in zip(batch_metadatas, transcriptions):
                i, _ = batch_metadata
                output.append(result:=" ".join([i, transcription]) + '\n')
                print(result)
            batch_metadatas.clear()
    
        # make directory
        os.makedirs(path:=os.path.join(self.output_dir, '1best_recog'), exist_ok=True)
        with open(os.path.join(path, 'text'), 'w+') as f:
            for s in output:
                f.write(s)
            f.close()

        print(f"end {self.scp_path}")