from app.keyword_extraction.main import ner
from sentence_transformers import SentenceTransformer
from gluonnlp.data import SentencepieceTokenizer
from app.keyword_extraction.keybert_model import KeyBERT
from transformers import BertModel, AutoModel

def ner_model_init():
    # TODO: change model path in ner function
    ner_model = ner()
    return ner_model

def kw_model_init():
    kw_model = KeyBERT(
        AutoModel.from_pretrained(
        'sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens',)
    )
    return kw_model

def filtering_model_init():
    filter_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    return filter_model