import requests
import json
import streamlit as st
import pandas as pd

backend_address = "http://localhost:8001"

# def stt(docs):
#     data = {
#         'docs':json.dumps(docs)
#     }
#     response = requests.post(
#         url = f'{backend_address}/saveWavFile/',
#         params = data
#     )
#     response = requests.get(
#         url=f'{backend_address}/sppechToText/',
#     )
#     return response

# def stt_postprocess(docs):
#     data = {
#         'docs':json.dumps(docs)
#     }
#     response = requests.get(
#         url = f'{backend_address}/sppechToText/',
#         params = data
#     )
#     return response

def stt(docs):
    data = {"file": docs}
    headers = {"Content-type": "application/json"}

    response = requests.post(
        url = f"{backend_address}/saveWavFile/",
        data = json.dumps(data),
        headers = headers
    )

    response = requests.get(
        url = f"{backend_address}/speechToText/"
    )

    response = requests.get(
        url = f"{backend_address}/sttPostProcessing/",
        data = response
    )

    response = requests.get(
        url = f"{backend_address}/segmentation/"
    )

    response = requests.get(
        url = f"{backend_address}/summarization/",
        data = response
    )
    print('>>>>>>>>>>>> finish summarization')

    results = response.json()['output']
    json_data = json.loads(results)
    st.write(json_data)
    

    return response

def main():
    st.title("STT -> postprocessing -> preprocessing -> Summary")

    uploaded_file = '/opt/ml/stt/backend/stt_example_1.wav'
    if uploaded_file:
        with st.spinner('wait for summarization'):
            result = stt(uploaded_file)
    st.write(result)
    
if __name__ == '__main__':        
    main()