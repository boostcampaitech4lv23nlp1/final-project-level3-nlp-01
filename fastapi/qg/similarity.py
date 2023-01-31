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
        qa_sim = get_similarity(model, data['answer'], data['question']) # keyword & question
        qo_sim = get_similarity(model, context_data[data['idx']]['context'], data['question']) # context & question

        score = (qa_sim + qo_sim + data['k_sim'])/3
        # score = ((qa_sim*0.2) + (qo_sim*0.4) + (data['sim']*0.4))
        tuple_lst.append((idx, score))
    tuple_lst = sorted(tuple_lst, key=lambda x: x[1], reverse=True)

    return tuple_lst