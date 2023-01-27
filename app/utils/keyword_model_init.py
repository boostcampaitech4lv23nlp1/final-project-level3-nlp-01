from ..keyword_extraction import ner
from app.keyword_extraction.keybert_model import KeyBERT
from transformers import BertModel
from sentence_transformers import SentenceTransformer

def ner_model_init():
    ner_model = ner()
    return ner_model

def kw_model_init():
    kw_model = KeyBERT(BertModel.from_pretrained('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'))
    return kw_model

def filtering_model_init():
    filter_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')