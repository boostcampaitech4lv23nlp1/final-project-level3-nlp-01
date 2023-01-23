import requests
import pickle
import json
import streamlit as st
import pandas as pd

backend_address = "http://localhost:8001"

def stt_sum(docs):
    data = {
        'docs': json.dumps(docs)
    }

    response = requests.get(
        url = f"{backend_address}/service",
        params = data
    )

    results = response.json()['output']
    json_data = json.loads(results)
    st.write(json_data)
    
    return


def main():
    st.title("STT postprocessing + Summarization")
    uploaded_file = st.file_uploader("Choose a file!", type=['csv'])
    if uploaded_file:
        docs = pd.read_csv(uploaded_file)
        sentences = ' '.join(list(docs['output']))
        st.write(docs)
        st.subheader("Summary")
        with st.spinner('Wait for Summarization ...'):
            stt_sum(sentences)

main()