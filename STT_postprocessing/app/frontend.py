import requests
import pickle
import json
import streamlit as st
import pandas as pd



def STT_postprocessing(docs):
    data = {
        'docs': json.dumps(docs)
    }

    response = requests.post(
        url = f"http://127.0.0.1:8001/STT",
        params = data
    )

    results = response.json()['output']
    json_data = json.loads(results)
    st.write(json_data)
    
    return


def main():
    st.title("STT postprocessing")
    # uploaded_file = st.text_input("텍스트 입력")
    uploaded_file = st.file_uploader("Choose a file!", type=['csv'])
    if uploaded_file:
        # docs = uploaded_file
        sentences= []
        docs = pd.read_csv(uploaded_file)
        for idx, item in docs.iterrows():
            sentences.append(item['output'])
        st.write(docs)
        st.subheader("csv")
        STT_postprocessing(sentences)

main()