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


from summary.main import summ_preprocess, summarize
from stt_postprocessing.main import stt_postprocess
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

class FileName(BaseModel):
    file:str

class STTOutput(BaseModel):
    stt_output:str

class STTPostprocessed(BaseModel):
    stt_postprocessed:str

# STT + postprocess + segmentation
@app.post('speechToText', description = 'stt inference')
def stt_inference(file: FileName):
    if file is None:
        return {'output':None}
    else:
         # with open(str(file.file), 'rb') as f:
        #     shutil.copyfileobj(file.file, f) ## commit 할때는 주석 풀기
        app.wav_filename = file.file
        
        torch.multiprocessing.set_start_method('spawn')     # multiprocess mode
        app.stt_output = stt_setup(
            make_dataset=False, 
            inference_wav_file=app.wav_filename
        )
        postprocessed = stt_postprocess(model_path='/opt/ml/project_models/stt/postprocessing_gpt',max_len=64, data = app.stt_output)


# input WAV file to save
@app.post('/saveWavFile/', description='save wav file')
# def save_wav_file(file: UploadFile=File(...)): ## commit 할때는 이 형식으로 데이터 불러오기
def save_wav_file(file: FileName):
    if file is None:
        return {'output': None}
    else:
        # with open(str(file.file), 'rb') as f:
        #     shutil.copyfileobj(file.file, f) ## commit 할때는 주석 풀기
        app.wav_filename = file.file
        return JSONResponse(
            status_code = 200,
            content = {
            "output": json.dumps(file.file)
            })

# STT inference
@app.get('/speechToText/', description='stt inference')
def stt_inference(file: FileName):
    try:
        filename = file
        torch.multiprocessing.set_start_method('spawn')     # multiprocess mode
        output = stt_setup(
            make_dataset=False, 
            inference_wav_file=filename
        )
        return JSONResponse(
            status_code = 200,
            content = {
            "output": json.dumps(output)
            })
    except AttributeError as e:
        return {'error':'start STT inference error'}

## AttributeError: 'FastAPI' object has no attribute 'stt_output' -> error 발생
## client에서 requests 보낼때의 문제 ~ json 구성 바꾸기

# STT postprocess
@app.get('/sttPostProcessing/', description='stt postprocessing')
def stt_postprocess(stt_output: STTOutput):
    try:
        input = stt_output
        output = stt_postprocess(model_path='/opt/ml/project_models/stt/postprocessing_gpt',max_len=64, data = input)
        return JSONResponse(
            status_code = 200,
            content = {
            "output": json.dumps(output)
            }
        )
    except AttributeError as e:
        return {'error':'start STT inference error'}

# Make phrase
@app.get('/segmentation/', description='make phrase')
def preprocess(stt_postprocessed: STTPostprocessed):
    try:
        input = stt_postprocessed
        output = summ_preprocess(input)
        return JSONResponse(
            status_code = 200,
            content = {
            "output": json.dumps(output)
            }
        )
    except AttributeError as e:
        return {'error':'start preprocessing error'}


#######################################

# Summarization
@app.get('/summarization/', description='start summarization')
def summary():
    try:
        input = app.preprocessed
        output = summarize(data = input,
                            sum_model_path='/opt/ml/project_models/summarization/kobart_all_preprocessed_without_news',
                            sum_model= 'kobart')
        print('finish summarization')
        return JSONResponse(
            status_code = 200,
            content = {
            "output": json.dumps(output)
            }
        )
    except AttributeError as e:
        return {'error':'start summarization error'}

########################################


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
    
    uvicorn.run(app, host="127.0.0.1", port=8001)