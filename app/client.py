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
    # headers = {"Content-type": "application/json"}
    response = requests.post(
        url = f"{backend_address}/summarization/",
        params = data,
    )

    results = response.json()['output']
    json_data = json.loads(results)
    return json_data

def keyword(seg_docs, summary_docs):
    data = {
        'seg_docs': json.dumps(seg_docs),
        'summary_docs': json.dumps(summary_docs)
    }

    page = ''
    while page == '':
        try:
            response = requests.get(
            url = f"{backend_address}/keyword",
            params = data,
            verify = False
            )
            results = response.json()['output']
            json_data = json.loads(results)
            return pd.DataFrame(json_data)
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue


# def keyword(seg_docs, summary_docs):
#     data = {
#         'seg_docs': json.dumps(seg_docs),
#         'summary_docs': json.dumps(summary_docs)
#     }

#     response = requests.get(
#         url = f"{backend_address}/keyword",
#         params = data,
#         verify = False
#     )

#     results = response.json()['output']
#     json_data = json.loads(results)
#     return pd.DataFrame(json_data)


def qg(keywords):
    keywords = keywords.to_json()
    data = {
        'keywords': keywords
    }

    response = requests.get(
        url = f"{backend_address}/qg",
        params = data,
        verify=False
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
    st.write(type(stt_inferenced))
    st.write(stt_inferenced[0])
    print(type(stt_inferenced))

    summarized = summary(stt_inferenced)
    st.subheader("Summarization result")
    st.write(summarized)

    #### keyword test
    st.title("keyword extraction")

    with open('seg_0125.pickle', 'rb') as f:
        seg_docs = pickle.load(f)
    
    with open("seg_0125.pickle", "rb" ) as f:
        summary_docs = pickle.load(f)

    keywords = keyword(seg_docs, summary_docs)
    st.dataframe(keywords, 50000, 500)

    ##### qg test
    st.title("qg")
    # with open('0126_keywords.pickle', 'rb') as f:
    #     keywords = pickle.load(f)

    qg_result = qg(keywords)
    st.write(qg_result)

    
if __name__ == '__main__':        
    main()