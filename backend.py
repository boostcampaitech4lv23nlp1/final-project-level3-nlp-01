import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from first_keyword_extraction.keyword_extraction import main_extraction
import json

app = FastAPI()

@app.get("/keyword")
def keyword_extraction(docs):
    input = json.loads(docs)
    output = main_extraction(input)
    return JSONResponse(
        status_code = 200,
        content = {
        "output": json.dumps(output)
        }
    )

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8001, reload=True)

#make -j 2 run_app