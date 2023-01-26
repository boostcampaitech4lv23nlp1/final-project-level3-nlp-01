import os
import torch
import whisper

from typing import Optional, Tuple, Sequence

class Inference():
    def __init__(
            self, 
            model,
            output_dir: str,
            fp16: bool,
            beam_size: int,
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
            language: Optional[str],
        ):
        self.model = model
        self.output_dir = output_dir

        self.scp_path, _, _ = data_path_and_name_and_type[0]
        self.options = {
            'language': language,
            'beam_size': beam_size,
            'fp16': fp16,
        }
        
    def __call__(self):
        print('whisper inference')

        with open(self.scp_path, 'r+') as f:
            lines = f.readlines()

        output = []
        for line in lines:
            i, path = line.strip().split(' ')

            # TODO: apt-get install ffmpeg README.md 파일에 명시하기
            result = self.model.transcribe(
                path,
                temperature=0,
                no_speech_threshold=0.6,
                **self.options
            )
            print(i, ' ', result['text'])
            output.append(" ".join([i, result['text']]) + '\n')
    
        # make directory
        os.makedirs(path:=os.path.join(self.output_dir, '1best_recog'), exist_ok=True)
        with open(os.path.join(path, 'text'), 'w+') as f:
            for s in output:
                f.write(s)
            f.close()

        print(f"end {self.scp_path}")