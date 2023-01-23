from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .summary.main import main_test
from .stt_postprocessing.test import main_inference
import json

import pickle

app = FastAPI()

def STT_postprocessing(docs):
    input = json.loads(docs)
    output = main_inference(model_path='/opt/ml/project_models/stt/postprocessing_gpt',max_len=64, data = input)
    print('finish postprocessing')
    return output


def summary(data):
    output = main_test(data = data,
                        sum_model_path='/opt/ml/project_models/summarization/kobart_all_preprocessed_without_news',
                        sum_model= 'kobart')
    print('finish summarization')
    return JSONResponse(
        status_code = 200,
        content = {
        "output": json.dumps(output)
        }
    )
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

@app.get("/service")
def main(docs):
    stt_post_processed = STT_postprocessing(docs)
    with open('/opt/ml/stt_postprocessed.pickle', 'wb') as f:
        pickle.dump(stt_post_processed, f)
    summarized = summary(stt_post_processed)
    # extracted_keyword = keyword_extraction(stt_post_processed)
    return summarized

