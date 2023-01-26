import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from qg.main import main
import json
import pickle

app = FastAPI()

# data_path = '/opt/ml/input/data/KE/keyword_for_qg.pickle'
# with open(data_path, 'rb') as f:
#     datas = pickle.load(f)

@app.get("/qg")
def qg_task(docs):
    input = json.loads(docs)
    output = main("kobart", input)
    print("Finish Question Generation")
    print(type(output))
    return output
    # return JSONResponse(
    #     status_code = 200,
    #     content = {
    #         "questions": json.dumps(output)
    #     }
    # )


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8001, reload=True)