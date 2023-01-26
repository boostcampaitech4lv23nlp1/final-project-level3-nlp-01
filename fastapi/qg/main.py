from kobart_qg import main_qg
from t5_pipeline import pipeline

T5_TOKENIZER_PATH = "/opt/ml/input/data/question_generation/t5_qg_tokenizer"
T5_MODEL_PATH = ""
KOBART_MODEL_PATH = ""

def main(model_type, docs):
    if model_type == "kobart":
        kobart_result = main_qg(docs)
        return kobart_result
    else:
        nlp = pipeline("multitask-qa-qg", tokenizer=T5_TOKENIZER_PATH)

        t5_result = []
        for doc in docs['context']:
            t5_result.append(nlp(doc))
        
        return t5_result
