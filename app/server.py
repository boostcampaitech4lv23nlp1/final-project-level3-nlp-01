from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .summary.main import main_test
import json

app = FastAPI()

# @app.get("/")
# def hello_world():
#     return {"hello": "world"}

@app.get("/summary")
def summary(docs):
    input = json.loads(docs)
    output = main_test(input)
    return JSONResponse(
        status_code = 200,
        content = {
        "output": json.dumps(output)
        }
    )


