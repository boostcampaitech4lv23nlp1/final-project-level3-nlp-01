import requests
import json
import streamlit as st
import pandas as pd
import pickle
import time

backend_address = "http://localhost:8001"

def stt(docs:str): # for streamlit test -> str (filepath)
    data = {"file": docs}
    headers = {"Content-type": "application/json"}

    page = ''
    while page == '':
        try:
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

        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue


def summary(segments):
    data = {
        'segments' : segments
    }
    response = requests.post(
        url = f"{backend_address}/summarization/",
        params = json.dumps(data),
    )

    results = response.json()['output']
    json_data = json.loads(results)
    return json_data


def keyword(segments, summarized):
    data = {
        'segments': segments,
        'summarized': summarized
    }

    page = ''
    while page == '':
        try:
            response = requests.post(
            url = f"{backend_address}/keyword",
            params = json.dumps(data),
            verify = False
            )
            results = response.json()['output']
            json_data = json.loads(results)
            return json_data
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue


def qg(keywords):
    data = {
        'keywords': keywords
    }
    response = requests.post(
        url = f"{backend_address}/qg",
        params = data,
        verify=False
    )

    results = response.json()['output']
    qg_result = json.loads(results)
    return qg_result

def main():
    st.title("BACKEND test")

    uploaded_file = '/opt/ml/history.wav'
    if uploaded_file:
        with st.spinner('wait for stt'):
            segments = stt(uploaded_file)
    st.subheader("STT segments")
    st.write(segments)
    with st.spinner('wait for summarization'):
        st.write(type(segments))
        summarized = summary(segments)
    st.subheader("Summarization Result")
    st.write(summarized)

    #### keyword test
    st.subheader("Keyword Extraction Result")
    with st.spinner('wait for keyword extraction'):
        keywords = keyword(segments, summarized)
    st.write(keywords)

    ##### qg test
    st.subheader("Question Generation, Result")
    with st.spinner('wait for question generation'):
        qg_result = qg(keywords)
    st.write(qg_result)

    
if __name__ == '__main__':        
    main()