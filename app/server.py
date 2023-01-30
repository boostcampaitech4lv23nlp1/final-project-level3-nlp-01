# TODO: add validation check class
import os
import json
import shutil
import pickle
import torch
import uvicorn
import io
import pandas as pd
from pydantic import BaseModel

from typing import Optional, List, Dict, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
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

class QuestionGenerationInput(BaseModel):
    '''
    type:
        List[Dict[str, Union[str, List]]]
    example:
        [{context: "string", keyword: [(18, "신사임당"), (19, '다른거')]}]
    '''
    keywords: List[Dict[str, Union[str, List]]]

# STT : input WAV file to save
@app.post('/saveWavFile/', description='save wav file') # input : WAV -> output : str
def save_wav_file(file: UploadFile=File(...)):

    if file is None:
        return {'output': None}
    else:
        filename = file.filename
        with open(filename, 'wb') as f:
            shutil.copyfileobj(file.file, f) # for streamlit test -> to be comment
        app.wav_filename = filename
        return {JSONResponse(
            status_code = 200,
            content = {
            "output": filename
            })}

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
        return {'status' : True}
    except AttributeError as e:
        return {'status': False}


# STT : STT postprocess
@app.get('/sttPostProcessing/', description='stt postprocessing') # input : None -> output : None
def stt_postprocess():
    try:
        input = app.stt_output
        output = postprocess(model = app.stt_post_model,
                            tokenizer = app.stt_post_tokenizer,
                            df = input)
        app.stt_postprocessed = output

        output = " ".join(output)
        return {'text': output}

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
        return {'text': output}
    except BaseException as e:
        return {'error': e}

# Summarization: Summarization
@app.post('/summarization/', description='start summarization') # input : list -> output : list
def summary(segments: STTOutput):

    stt_output = segments.stt_output
    print(stt_output)
    try:
        input = stt_output
        output = summarize(model = app.summary_model,
                            tokenizer = app.summary_tokenizer,
                            postprocess_model = app.segment_model,
                            preprocessed = input,
                            sum_model = app.summary_model_name)
        print('finish summarization')
        return {'summarization_output': output}
    except AttributeError as e:
        return {'error':'start summarization error'}

# ########################################


# Keyword Extraction : Keyword Extraction
@app.post("/keyword") #input = seg&summary docs, output = context, keyword dataframe() to json
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
    temp_keywords = main_extraction(ner_model = app.ner_model, 
                                    kw_model = app.kw_model,
                                    docs = seg_docs) #1차 키워드 추출

    keywords = main_filtering(filter_model = app.filter_model,
                                summary_datas = summary_docs, 
                                keyword_datas = temp_keywords) #2차 키워드 추출
    
    return {
        'output': keywords
    }

# ########################################    



# QG : Question Generation
@app.post("/questionGeneration", description="start question generation")
def qg_task(req: QuestionGenerationInput):
    '''
    type:
        List[Dict[str, Union[str, List]]]
    example:
        [{context: "string", keyword: [(18, "신사임당"), (19, '다른거')]}]
    '''
    input = req.keywords
    for idx in range(len(input)):
        input[idx]['keyword'] = [tuple(keyword) for keyword in input[idx]['keyword']]

    output = question_generate("t5", "question-generation", input, app.qg_model, app.qg_tokenizer) 

    result = {'questions': [], 'answers' : []}
    for dictionary in output:
        result['questions'].append(dictionary['question'])
        result['answers'].append(dictionary['answer'])
    
    filename = app.wav_filename.split('.')[0]
    app.result_filepath = f'./{filename}.csv'
    pd.DataFrame(result).to_csv(app.result_filepath)

    return {'output': output}

@app.get('/downloadResult/', description='download result')
def download_result():
    try:
        filepath = app.result_filepath
        return FileResponse(filepath)
    except:
        return {'error': 'download_result error'}

# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method('spawn', force=True)     # multiprocess mode
#     uvicorn.run(app, host="127.0.0.1", port=8001)
