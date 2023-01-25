import requests
import pickle
import json
import streamlit as st
import pandas as pd

backend_address = "http://localhost:8001"

def first_keyword(docs):
    data = {
        'docs': json.dumps(docs)
    }

    response = requests.get(
        url = f"{backend_address}/first_keyword",
        params = data
    )

    results = response.json()['output']
    json_data = json.loads(results)
    return  pd.DataFrame(json_data)


def second_keyword(summary_docs, temp_keywords):

    temp_keywords = temp_keywords.to_json()
    data = {
        'summary_docs': json.dumps(summary_docs),
        'temp_keywords': temp_keywords
    }

    response = requests.get(
        url = f"{backend_address}/second_keyword",
        params = data
    )

    results = response.json()['output']
    json_data = json.loads(results)
    return pd.DataFrame(json_data)


def main():
    st.title("keyword extraction")
    uploaded_file = st.file_uploader("Choose a file!", type=["pickle"])
    if uploaded_file:
        docs = pickle.load(uploaded_file)
        st.write(docs)
        st.subheader("First Keywords")
        temp_keywords = first_keyword(docs)
        st.dataframe(temp_keywords, 50000, 500)

        st.subheader("Second Keywords")
        
        #테스트 용도
        with open("seg_0125.pickle", "rb" ) as f:
            summary_docs = pickle.load(f)

        keywords = second_keyword(summary_docs, temp_keywords)
        st.dataframe(keywords, 50000, 500)


main()