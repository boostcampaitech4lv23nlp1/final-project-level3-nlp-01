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

from typing import Optional, List, Tuple, Dict, Union
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import pandas as pd

from summary.main import segment, summarize
from stt_postprocessing.main import postprocess
from STT.setup import stt_setup

from keyword_extraction.main import main_extraction
from keyword_extraction.filtering import main_filtering
from question_generation.main import generation


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
    file: str

class STTOutput(BaseModel):
    '''
    input:
        {"stt_output" : ["I'm baby", "I'm 26 years old]}
    '''
    stt_output: List[str]

class KeyWordInput(BaseModel):
    '''
    example:
        {
            seg_docs: [],
            summary_docs: [],
        }
    '''
    seg_docs: List[str]
    summary_docs: List[str]

# input WAV file to save
@app.post('/saveWavFile/', description='save wav file')
def save_wav_file(file: UploadFile=File(...)):
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
        output = postprocess(model_path='./models/stt/postprocessing_gpt', df = input)
        app.stt_postprocessed = output

        output = " ".join(output)
        return {'text': output}

    except AttributeError as e:
        return {'error':'start STT inference error'}

# STT 후에 바로 처리하는 segmentation (분리)
@app.get('/segmentation/', description='make phrase')
def preprocess():
    try:
        input = app.stt_postprocessed
        print('<<<<<<<<<<<<segmentation start>>>>>>>>>>>>>')
        # output: list = segment(input)

        output = ['나는 아기다.', 'segmentation 요청에 대한 결과입니다.']
        return {'text': output}
    except BaseException as e:
        return {'error': e}

# Summarization
@app.post('/summarization/', description='start summarization')
def summary(segments: STTOutput):
    '''
    description:
        summarization 하는 요청을 받는 함수
    '''

    stt_output = segments.stt_output
    print(stt_output)
    try:
        input = stt_output
        # output: list = summarize(preprocessed = input,
        #                     sum_model_path='/opt/ml/project_models/summarization/kobart_all_preprocessed_without_news',
        #                     sum_model= 'kobart')
        output = ['나는 아기다.', 'summarization 요청에 대한 결과']
        
        print('finish summarization')
        return {'summarization_output': output}
    except AttributeError as e:
        return {'error':'start summarization error'}

# Keyword Extraction
@app.post("/keyword", description='start keyword extraction') #input = seg&summary docs, output = context, keyword dataframe() to json
def keyword_extraction(req: KeyWordInput):
    '''
    input:
        seg_docs: list
        summary_docs: list
    output:
        keywords: List[Dict[str, List[Tuple[int, str]]]]
    '''
    seg_docs = req.seg_docs
    summary_docs = req.summary_docs

    # temp_keywords = main_extraction(seg_docs) #1차 키워드 추출
    # keywords = main_filtering(summary_docs, temp_keywords) #2차 키워드 추출
    
    keywords = [{
        'context': "나는 아기다",
        'keyword': [(1, "아기"), (2, "신생아")]
    },
    {
        'context': "나는 어른이다",
        'keyword': [(1, "어른"), (2, "성인")]
    }
    ]
    return {
        'output': keywords
    }

class QuestionGenerationInput(BaseModel):
    '''
    type:
        List[Dict[str, Union[str, List]]]
    example:
        [{context: "string", keyword: [(18, "신사임당"), (19, '다른거')]}]
    '''
    keywords: List[Dict[str, Union[str, List]]]
    

# Question Generation
@app.post("/questionGeneration", description="start question generation")
def qg_task(req: QuestionGenerationInput):
    input = req.keywords
    for idx in range(len(input)):
        input[idx]['keyword'] = [tuple(keyword) for keyword in input[idx]['keyword']]
        
    # output = generation("kobart", input)
    '''
    example:
        [{'question': "string" ,'answer': "string"}, {'question': "string" ,'answer': "string"}]
    '''
    output = [
        {'question': "신생아는 왜 귀여운가?", "answer": "그냥 귀엽기 때문이다"},
        {'question': "아기는 왜 귀여운가?", "answer": "그냥 귀엽기 때문이다"}
    ]

    return {'output': output}

# @app.get("/service")
# def main(docs):
#     stt_post_processed = STT_postprocessing(docs)
#     with open('/opt/ml/stt_postprocessed.pickle', 'wb') as f:
#         pickle.dump(stt_post_processed, f)
#     summarized = summary(stt_post_processed)
#     # extracted_keyword = keyword_extraction(stt_post_processed)
#     return summarized

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)     # multiprocess mode
    uvicorn.run(app, host="127.0.0.1", port=8001)