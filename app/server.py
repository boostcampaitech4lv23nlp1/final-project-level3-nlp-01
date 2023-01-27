# TODO: validation check

# TODO: keyword extraction function
# TODO: question generation function
# TODO: async test

import json
import shutil
import pickle
import torch
import uvicorn
from pydantic import BaseModel

from typing import Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware


from summary.main import segment, summarize
from stt_postprocessing.main import postprocess
from STT.setup import stt_setup


app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# def model_init(model_path):
#     return model, tokenizer

# app.stt_model, app.stt_tokenizer = model_init(fmalsdfjkadskl)

class FileName(BaseModel):
    file:str

class STTOutput(BaseModel):
    stt_output:str

class STTPostprocessed(BaseModel):
    stt_postprocessed:str

# input WAV file to save
@app.post('/saveWavFile/', description='save wav file')
def save_wav_file(file: UploadFile=File(...)):
# def save_wav_file(file: FileName): # for streamlit test
    if file is None:
        return {'output': None}
    else:
        with open(file.filename, 'wb') as f:
            shutil.copyfileobj(file.file, f) ## commit 할때는 주석 풀기
        app.wav_filename = file.filename
        return {JSONResponse(
            status_code = 200,
            content = {
            "output": file.filename
            })}

# STT inference
@app.get('/speechToText/', description='stt inference')
def stt_inference():
    try:
        filename = app.wav_filename
        
        output = stt_setup(
            make_dataset=False, 
            inference_wav_file=filename
        )
        app.stt_output = output
        return {'status' : True}
    except AttributeError as e:
        return {'status': False}


# STT postprocess
@app.get('/sttPostProcessing/', description='stt postprocessing')
def stt_postprocess():
    try:
        input = app.stt_output
        output = postprocess(model_path='./GPT_2', df = input)
        app.stt_postprocessed = output

        output = " ".join(output)
        return {'text': output}

    except AttributeError as e:
        return {'error':'start STT inference error'}

# STT 후에 바로 처리하는 segmentation
@app.get('/segmentation/', description='make phrase')
def preprocess():
    try:
        input = app.stt_postprocessed
        output = segment(input)
        return {'output': output}
    except BaseException as e:
        return {'error': e}


#######################################

# Summarization
# @app.get('/summarization/', description='start summarization')
# def summary():
#     try:
#         input = app.preprocessed
#         output = summarize(data = input,
#                             sum_model_path='/opt/ml/project_models/summarization/kobart_all_preprocessed_without_news',
#                             sum_model= 'kobart')
#         print('finish summarization')
#         return JSONResponse(
#             status_code = 200,
#             content = {
#             "output": json.dumps(output)
#             }
#         )
#     except AttributeError as e:
#         return {'error':'start summarization error'}

# ########################################


# TODO: add keyword_extraction function - async ?
# async def keyword_extraction(docs):
#     input = json.loads(docs)
#     output = main_test(input)
#     return JSONResponse(
#         status_code = 200,
#         content = {
#         "output": json.dumps(output)
#         }
#     )

# TODO: add question_generation function

# @app.get("/service")
# def main(docs):
#     stt_post_processed = STT_postprocessing(docs)
#     with open('/opt/ml/stt_postprocessed.pickle', 'wb') as f:
#         pickle.dump(stt_post_processed, f)
#     summarized = summary(stt_post_processed)
#     # extracted_keyword = keyword_extraction(stt_post_processed)
#     return summarized

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')     # multiprocess mode
    uvicorn.run(app, host="127.0.0.1", port=8001)