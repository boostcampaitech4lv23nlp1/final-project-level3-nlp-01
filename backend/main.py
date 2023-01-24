from fastapi import FastAPI, File, UploadFile
from STT.setup import stt_setup

from typing import Optional
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import torch

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get('/speechToText/', description='start stt inference')
def start_stt_inference():
    try:
        filename = app.filename
        # torch.multiprocessing.set_start_method('spawn')     # multiprocess mode
        # app.stt_output = stt_setup(
        #     make_dataset=False, 
        #     inference_wav_file=app.filename
        # )
        return {'response': 'success'}
    except AttributeError as e:
        return {'error': 'start_stt_inference error'}

@app.get('/postProcessing/', description='start postprocessing')
def start_post_processing():
    try:
        stt_output = app.stt_output
        # TODO: post processing pipeline
        return {'response': 'success'}
    except AttributeError as e:
        return {'error': 'start_post_processing error'}

@app.post('/saveWavFile/', description='save wav file')
def save_wav_file(file: UploadFile=File(...)):
    if file is None:
        return {'output': None}
    else:
        with open(str(file.filename), 'wb') as f:
            shutil.copyfileobj(file.file, f)
        app.filename = file.filename
        return {'filename': file.filename}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)