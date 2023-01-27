
from sklearn.metrics.pairwise import cosine_similarity

def get_use_keyword(filter_model, docs, keywords, top_n):
    if keywords == []:
        return []
    
    # load model
    model = filter_model

    # calculate embeddings
    doc_embedding = model.encode([docs]) # (1, 384)
    keywords_embeddings = model.encode(keywords) # (len(keywords), 384)

    # cosine similarity
    distances = cosine_similarity(doc_embedding, keywords_embeddings)
    use_keyword = [(keywords[index], (distances[0][index])) for index in distances.argsort()[0][::-1][:top_n]] # (keyword, distance) 형태로 재생성

    return [word[0] for word in use_keyword if word[1] > 0.2] # 유사도 0.2 이상


def main_filtering(filter_model, summary_datas, keyword_datas):
    summary_docs = ""
    for data in summary_datas:
        summary_docs += str(data)

    # keyword 리스트 - 문서간 유사도 계산
    for idx, keyword in enumerate(keyword_datas["keyword"]):
        new_keyword = get_use_keyword(filter_model, summary_docs, keyword, len(keyword)//2)
        keyword_datas["keyword"][idx] = new_keyword
    
    return keyword_datas