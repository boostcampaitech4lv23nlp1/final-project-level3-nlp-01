import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qg.main import main
import json
import pickle

app = FastAPI()


'''
model_type = "kobart" or "t5
task = "question_generation" or "e2e-qg"
main(model_type, task, input, t5_model, t5_tokenizer)
'''

@app.get("/qg")
def qg_task(keyword_docs):
    input = json.loads(keyword_docs)
    output = main("t5", "question-generation", input, t5_model, t5_tokenizer) 
    print("Finish Question Generation")

    return output
    # return JSONResponse(
    #     status_code = 200,
    #     content = {
    #         "questions": json.dumps(output)
    #     }
    # )


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8001, reload=True)