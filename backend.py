import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from first_keyword_extraction.keyword_extraction import main_extraction
from second_keyword_extraction.keyword_filtering import main_filtering
import json
import pandas as pd

app = FastAPI()

@app.get("/first_keyword")
def keyword_extraction(docs):
    input = json.loads(docs)
    output = main_extraction(input)
    return JSONResponse(
        status_code = 200,
        content = {
        "output": json.dumps(output)
        }
    )

@app.get("/second_keyword")
def keyword_extraction(summary_docs, temp_keywords):
    summary_input = json.loads(summary_docs)
    keywords_input = json.loads(temp_keywords)

    keywords_input = pd.DataFrame(keywords_input)

    output = main_filtering(summary_input, keywords_input)
    output = output.to_json()

    return JSONResponse(
        status_code = 200,
        content = {
        "output": output
        }
    )

if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8001, reload=True)