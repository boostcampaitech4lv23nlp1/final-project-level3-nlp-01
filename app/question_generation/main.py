from .kobart_qg import main_qg
from .t5_pipeline import pipeline
from .qg_filtering import calculate, filtering_topN

# KOBART_MODEL_PATH = "Sehong/kobart-QuestionGeneration"


def question_generate(model_type, task, docs, model, tokenizer, filter_model):
    '''
    ## question_generate
    description:
    - 주어진 segment에서 추출한 answer word를 답변으로 하는 질문을 생성합니다.
    
    args:
    - model_type : 질문 생성시, 사용하는 모델 종류
        - str ['t5', 'kobart'] 중 하나
    - task : 질문생성 태스크 이름
        - str = 'question-generation'
    - docs : answer extraction 부분에서 추출한 context-answer 쌍
        - list(dict)
    - model : 질문 생성시 사용하는 모델
        - t5 사용시 : T5ForConditionalGeneration
        - kobart 사용시 : None - 코드 내에서 불러옴
    - tokenizer : 질문 생성시 사용하는 tokenizer 모델
        - t5 사용시 : T5TokenizerFast
        - kobart 사용시 : None - 코드 내에서 불러옴
    - filter_model : 질문 생성 후 top_n 개 필터링 유사도 계산에 사용하는 SBERT 모델
    '''

    if model_type == "kobart":
        kobart_result = main_qg(docs)
        return kobart_result
    else:
        nlp = pipeline(task, model=model, tokenizer=tokenizer, ans_model=model, ans_tokenizer=tokenizer)

        t5_result = []
        for idx, data in enumerate(docs):
            t5_result.extend(nlp(idx, data['context'], data['keyword']))

        qg_result = []
        for data in t5_result:
            if '?' in data['question'] and len(data['question']) > 10:
                qg_result.append(data)
        qg_result = list({data['question']: data for data in qg_result}.values())

        tuple_lst = calculate(filter_model, qg_result, docs)

        topN_result, total_result = filtering_topN(tuple_lst, qg_result, k=20)

        return topN_result, total_result
