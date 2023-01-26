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
import pandas as pd

# from .summary.main import segment, summarize
# from .stt_postprocessing.main import postprocess
# from .STT.setup import stt_setup
from app.keyword_extraction.main import main_extraction
from app.keyword_extraction.filtering import main_filtering
from app.question_generation.main import generation


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
    stt_output:list

# input WAV file to save
@app.post('/saveWavFile/', description='save wav file')
# def save_wav_file(file: UploadFile=File(...)):
def save_wav_file(file: FileName): # for streamlit test
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
def stt_inference():
    try:
        filename = app.wav_filename
        # torch.multiprocessing.set_start_method('spawn')     # multiprocess mode
        output = stt_setup(
            make_dataset=False, 
            inference_wav_file=filename
        )
        app.stt_output = output
        return {'response': 'success'}
    except AttributeError as e:
        return {'error':'start STT inference error'}


# STT postprocess
@app.get('/sttPostProcessing/', description='stt postprocessing')
def stt_postprocess():
    try:
        input = app.stt_output
        output = postprocess(model_path='/opt/ml/project_models/stt/postprocessing_gpt', df = input)
        app.stt_postprocessed = output
        
        print('<<<<<<<<postprocess passed>>>>>>>>')
        return {'response': 'success'}
        
    except AttributeError as e:
        return {'error':'start STT inference error'}


# Make phrase
@app.get('/segmentation/', description='make phrase')
def preprocess():
    try:
        input = app.stt_postprocessed
        print('<<<<<<<<<<<<segmentation start>>>>>>>>>>>>>')
        output = segment(input)
        print('<<<<<<<<<<<<segmentation passed>>>>>>>>>>>>>')
        return JSONResponse(
            status_code = 200,
            content = {
            "output": json.dumps(output)
            }
        )
    except AttributeError as e:
        return {'error':'start preprocessing error'}

# Summarization
@app.post('/summarization/', description='start summarization')
def summary(segments):
    print('<<<<<<<<here>>>>>>>>')

    stt_output = json.loads(segments)
    print(stt_output)
    try:
        input = stt_output
        output = summarize(preprocessed = input,
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

# ########################################


# Keyword Extraction
@app.get("/keyword") #input = seg&summary docs, output = context, keyword dataframe() to json
def keyword_extraction(seg_docs, summary_docs):
    seg_docs = json.loads(seg_docs)
    temp_keywords = main_extraction(seg_docs) #1차 키워드 추출

    summary_docs = json.loads(summary_docs)
    keywords = main_filtering(summary_docs, temp_keywords) #2차 키워드 추출
    keywords = keywords.to_json()
    
    return JSONResponse(
        status_code = 200,
        content = {
        "output": keywords
        }
    )

# Question Generation
@app.get("/qg")
def qg_task(keywords):
    input = json.loads(keywords)
    input = pd.DataFrame(input)
    output = generation("kobart", input)

    return JSONResponse(
        status_code = 200,
        content = {
            "output": json.dumps(output)
        }
    )

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