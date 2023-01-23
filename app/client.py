import requests
import pickle
import json
import streamlit as st
import pandas as pd

backend_address = "http://localhost:8001"

def summarization(docs):
    data = {
        'docs': json.dumps(docs)
    }

    response = requests.get(
        url = f"{backend_address}/summary",
        params = data
    )

    results = response.json()['output']
    json_data = json.loads(results)
    st.dataframe(pd.DataFrame(json_data), 50000, 500)
    return


def main():
    st.title("Text Summarization")
    uploaded_file = st.file_uploader("Choose a file!", type=["pickle"])
    if uploaded_file:
        docs = pickle.load(uploaded_file)
        st.write(docs)
        st.subheader("Summary")
        with st.spinner('Wait for Summarization ...'):
            summarization(docs)

main()