
from sklearn.metrics.pairwise import cosine_similarity

def get_use_keyword(filter_model, docs, keywords, top_n):
    if keywords == []:
        return []

    # load model
    model = filter_model

    # calculate embeddings
    doc_embedding = model.encode([docs]) # (1, 384)

    only_keywords = [] #튜플에서 키워드만 빼서 구성
    for keyword in keywords:
        only_keywords.append(keyword[1])
    keywords_embeddings = model.encode(only_keywords)

    distances = cosine_similarity(doc_embedding, keywords_embeddings)

    new_keywords = []

    similarity = distances.tolist()[0]
    for index, sim in enumerate(similarity):
        keywords[index] = keywords[index] + (sim,) #튜플에 유사도 추가
        new_keywords.append(keywords[index])

    new_keywords.sort(key=lambda x: x[2], reverse=True)

    return new_keywords[:top_n]


def main_filtering(filter_model, summary_datas, keyword_datas):
    '''
    ## main_filtering
    description:
    - 추출한 answer words 중, 요약문과의 유사도를 계산하여 주요 답안을 걸러냅니다.
    
    args:
    - filter_model : filtering 진행 시 사용하는 모델
        - KobertCRF
    - summary_datas : 요약된 결과물
        - list
    - keyword_datas : 1차 answer 추출 후 결과물
        - list(dict)
    '''

    summary_docs = ""
    for data in summary_datas:
        summary_docs += str(data)

    # keyword 리스트 - 문서간 유사도 계산
    list_of_key_word = []
    for data in keyword_datas:
        new_keywords = get_use_keyword(filter_model, summary_docs, data['keyword'], len(data['keyword'])//2)
        if(new_keywords):
            list_of_key_word.append({"context" : data['context'], "keyword": new_keywords})
    
    return list_of_key_word