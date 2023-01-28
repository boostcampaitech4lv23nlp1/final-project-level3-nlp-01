from kobart_qg import main_qg
from t5_pipeline import pipeline

# KOBART_MODEL_PATH = "Sehong/kobart-QuestionGeneration"


def main(model_type, task, docs, model, tokenizer):
    if model_type == "kobart":
        kobart_result = main_qg(docs)
        return kobart_result
    else:
        nlp = pipeline(task, model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer)

        t5_result = []
        for data in docs:
            t5_result.append(nlp(data['context'], data['keyword']))
        
        return t5_result
