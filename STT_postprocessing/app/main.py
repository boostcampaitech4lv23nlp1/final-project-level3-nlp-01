import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from inference import postprocess
from .utils.stt_model_init import stt_post_model_init
import json

app = FastAPI()


@app.post("/STT")
def STT_postprocessing(docs):
    input = json.loads(docs)
    output = postprocess(model_path='/opt/ml/espnet-asr/final/GPT_2', df = input)
    return {'output' : json.dumps(output)}

# JSONResponse(
#         status_code = 200,
#         content = {
#         "output": json.dumps(output)
#         }
#     )

