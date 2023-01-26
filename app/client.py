import requests
import json
import streamlit as st
import pandas as pd

backend_address = "http://localhost:8001"

def stt(docs):
    data = {"file": docs}
    headers = {"Content-type": "application/json"}

    response = requests.post(
        url = f"{backend_address}/saveWavFile/",
        data = json.dumps(data),
        headers = headers
    )
    print('>>>>>>>>>>>> finish load WAV file')
    response = requests.get(
        url = f"{backend_address}/speechToText/"
    )
    print('>>>>>>>>>>>> finish STT inference')
    response = requests.get(
        url = f"{backend_address}/sttPostProcessing/",
    )
    print('>>>>>>>>>>>> finish STT postprocess')
    response = requests.get(
        url = f"{backend_address}/segmentation/"
    )
    print('>>>>>>>>>>>> finish segment')

    results = response.json()['output']
    json_data = json.loads(results)
    st.write(json_data)
    

    return response

def main():
    st.title("STT -> postprocessing -> preprocessing -> Summary")

    uploaded_file = '/opt/ml/level3_productserving-level3-nlp-01/history-03.wav'
    if uploaded_file:
        with st.spinner('wait for stt'):
            result = stt(uploaded_file)
    st.write(result)
    
if __name__ == '__main__':        
    main()