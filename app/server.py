# TODO: add validation check class
import os
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

from .summary.main import segment, summarize
from .stt_postprocessing.main import postprocess
from .STT.setup import stt_setup
from .keyword_extraction.main import main_extraction
from .keyword_extraction.filtering import main_filtering
from .question_generation.main import question_generate

from .utils.stt_model_init import stt_model_init, stt_post_model_init, segment_model_init
from .utils.summary_model_init import summary_model_init
from .utils.keyword_model_init import ner_model_init, kw_model_init, filtering_model_init
from .utils.qg_model_init import qg_model_init

os.environ['TOKENIZERS_PARALLELISM'] = 'True'

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# model init
app.stt_model = stt_model_init()
app.stt_post_model, app.stt_post_tokenizer = stt_post_model_init(
            model_path = '/opt/ml/project_models/stt/postprocessing_gpt')
app.segment_model = segment_model_init()

app.summary_model, app.summary_tokenizer, app.summary_model_name = summary_model_init(
    model_path = '/opt/ml/project_models/summarization/kobart_all_preprocessed_without_news_05',
    model_name = 'kobart')

app.ner_model = ner_model_init()
app.kw_model = kw_model_init()
app.filter_model = filtering_model_init()

app.qg_model, app.qg_tokenizer = qg_model_init()

# input validation
class FileName(BaseModel):
    file: str

class STTOutput(BaseModel):
    stt_output: list

class SegmentsOutput(BaseModel):
    segments: list

class SummaryOutput(BaseModel):
    summarization: list

class KeywordOutput(BaseModel):
    keywords: dict


# STT : input WAV file to save
@app.post('/saveWavFile/', description='save wav file') # input : WAV -> output : str
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

# STT : STT inference
@app.get('/speechToText/', description='stt inference') # input : None -> output : None
def stt_inference():
    try:
        filename = app.wav_filename
        output = stt_setup(
            model= app.stt_model,
            make_dataset=False, 
            inference_wav_file=filename
        )
        app.stt_output = output
        return {'response': 'success'}
    except AttributeError as e:
        return {'error':'start STT inference error'}


# STT : STT postprocess
@app.get('/sttPostProcessing/', description='stt postprocessing') # input : None -> output : None
def stt_postprocess():
    try:
        input = app.stt_output
        output = postprocess(model = app.stt_post_model,
                            tokenizer = app.stt_post_tokenizer,
                            df = input)
        app.stt_postprocessed = output
        
        print('<<<<<<<<postprocess passed>>>>>>>>')
        return {'response': 'success'}
        
    except AttributeError as e:
        return {'error':'start STT inference error'}


# STT : Make phrase
@app.get('/segmentation/', description='make phrase') # input : None -> output : list
def preprocess():
    try:
        input = app.stt_postprocessed
        print('<<<<<<<<<<<<segmentation start>>>>>>>>>>>>>')
        output = segment(app.segment_model, input)
        print('<<<<<<<<<<<<segmentation passed>>>>>>>>>>>>>')
        return JSONResponse(
            status_code = 200,
            content = {
            "output": json.dumps(output)
            }
        )
    except AttributeError as e:
        return {'error':'start preprocessing error'}

# Summarization: Summarization
@app.post('/summarization/', description='start summarization') # input : list -> output : list
def summary(segments):

    stt_output = json.loads(segments)
    # try:
    input = stt_output
    output = summarize(model = app.summary_model,
                        tokenizer = app.summary_tokenizer,
                        postprocess_model = app.segment_model,
                        preprocessed = input,
                        sum_model = app.summary_model_name)
    print('finish summarization')
    return JSONResponse(
        status_code = 200,
        content = {
        "output": json.dumps(output)
        }
    )
    # except AttributeError as e:
    #     return {'error':'start summarization error'}

# ########################################


# Keyword Extraction : Keyword Extraction
@app.post("/keyword") #input = seg&summary docs, output = context, keyword dataframe() to json
def keyword_extraction(seg_docs, summary_docs): #TODO: input 형식 확인해서 validation 추가
    seg_docs = json.loads(seg_docs)
    temp_keywords = main_extraction(ner_model = app.ner_model, 
                                    kw_model = app.kw_model,
                                    docs = seg_docs) #1차 키워드 추출

    summary_docs = json.loads(summary_docs)
    keywords = main_filtering(filter_model = app.filter_model,
                                summary_datas = summary_docs, 
                                keyword_datas = temp_keywords) #2차 키워드 추출
    
    return JSONResponse(
        status_code = 200,
        content = {
        "output": json.dumps(keywords)
        }
    )

# ########################################    

# QG : Question Generation
@app.post("/qg")
def qg_task(keywords):
    input = json.loads(keywords)
    output = question_generate("t5", "question-generation", input, app.qg_model, app.qg_tokenizer) 

    return JSONResponse(
        status_code = 200,
        content = {
            "output": json.dumps(output)
        }
    )
# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method('spawn', force=True)     # multiprocess mode
#     uvicorn.run(app, host="127.0.0.1", port=8001)