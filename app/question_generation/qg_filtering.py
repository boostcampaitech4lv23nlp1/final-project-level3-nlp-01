from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# MODEL_NAME = 'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
# model = SentenceTransformer(MODEL_NAME)


def get_similarity(model, data1, data2):
    data1_embeddings = model.encode([data1]) 
    data2_embeddings = model.encode([data2])
    distances = cosine_similarity(data1_embeddings, data2_embeddings)
    return distances[0][0]


def calculate(model, generated_data, context_data):
    tuple_lst = []
    for idx, data in enumerate(generated_data):
        # qa_sim = get_similarity(model, data['answer'], data['question']) # keyword & question
        qo_sim = get_similarity(model, context_data[data['idx']]['context'], data['question']) # context & question

        # score = ((qa_sim*0.2) + (qo_sim*0.4) + (data['k_sim']*0.4))
        tuple_lst.append((idx, qo_sim))

    return sorted(tuple_lst, key=lambda x: x[1], reverse=True)


def filtering_topN(tuple_lst, qg_result, k):
    top_n, answer = [], []
    for idx, _ in tuple_lst:
        if len(top_n) == k:
            break
        
        if qg_result[idx]['answer'] not in qg_result[idx]['question']:
            if qg_result[idx]['answer'] not in answer:
                answer.append(qg_result[idx]['answer'])
                del qg_result[idx]['idx']
                top_n.append(qg_result[idx])
    
    for idx, data in enumerate(qg_result): 
        if 'idx' in data.keys():
            del qg_result[idx]['idx']
    
    return top_n, qg_result

        