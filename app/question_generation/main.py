from kobart_qg import main_qg
from t5_pipeline import pipeline

# KOBART_MODEL_PATH = "Sehong/kobart-QuestionGeneration"


def question_generate(model_type, task, docs, model, tokenizer):
    '''
    ## question_generate
    description:
    - 주어진 segment에서 추출한 answer word를 답변으로 하는 질문을 생성합니다.
    
    args:
    - model_type : 질문 생성시, 사용하는 모델 종류
        - str ['t5', 'kobart'] 중 하나
    - task : 요약 태스크 이름
        - str = 'question-generation'
    - docs : answer extraction 부분에서 추출한 context-answer 쌍
        - list(dict)
    - model : 질문 생성시 사용하는 모델
        - t5 사용시 : T5ForConditionalGeneration
        - kobart 사용시 : None - 코드 내에서 불러옴
    - tokenizer : 질문 생성시 사용하는 tokenizer 모델
        - t5 사용시 : T5TokenizerFast
        - kobart 사용시 : None - 코드 내에서 불러옴
    '''

    if model_type == "kobart":
        kobart_result = main_qg(docs)
        return kobart_result
    else:
        nlp = pipeline(task, model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer)

        t5_result = []
        for data in docs:
            t5_result.append(nlp(data['context'], data['keyword']))

        qg_result = []
        for data in t5_result:
            if '?' in data['question'] and len(data['question']) > 10:
                qg_result.append(data)
        qg_result = list({data['question']: data for data in qg_result}.values())
        
        return qg_result
