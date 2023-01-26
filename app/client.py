import requests
import json
import streamlit as st
import pandas as pd

backend_address = "http://localhost:8001"

def stt(docs:str): # for streamlit test -> str (filepath)
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
    )

    response = requests.get(
        url = f"{backend_address}/segmentation/"
    )

    results = response.json()['output']
    json_data = json.loads(results)
    return json_data

def summary(docs:list):
    data = {'segments':json.dumps(docs)}
    headers = {"Content-type": "application/json"}
    response = requests.post(
        url = f"{backend_address}/summarization/",
        data = data,
        headers = headers
    )

    results = response.json()['output']
    json_data = json.loads(results)
    return json_data

def main():
    st.title("BACKEND test")

    uploaded_file = '/opt/ml/level3_productserving-level3-nlp-01/history-03.wav'
    if uploaded_file:
        with st.spinner('wait for stt'):
            stt_inferenced = stt(uploaded_file)
    st.subheader("STT segments")
    st.write(stt_inferenced)
    print(type(stt_inferenced))

    summarized = summary(stt_inferenced)
    st.subheader("Summarization result")
    st.write(summarized)
    
if __name__ == '__main__':        
    main()