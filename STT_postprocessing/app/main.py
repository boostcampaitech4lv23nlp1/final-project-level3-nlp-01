import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from test import main_inference
import json

app = FastAPI()


@app.post("/STT")
def keyword_extraction(docs):
    input = json.loads(docs)
    output = main_inference(model_path='/opt/ml/espnet-asr/final/GPT_2',max_len=64, df = input)
    return {'output' : json.dumps(output)}

# JSONResponse(
#         status_code = 200,
#         content = {
#         "output": json.dumps(output)
#         }
#     )

