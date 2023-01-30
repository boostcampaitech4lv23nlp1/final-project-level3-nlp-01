from __future__ import absolute_import, division, print_function, unicode_literals
from gluonnlp.data import SentencepieceTokenizer

from .ner_config.net import KobertCRF
from .keybert_model import KeyBERT
from .data_utils.utils import Config
from .data_utils.vocab_tokenizer import Tokenizer
from .data_utils.pad_sequence import keras_pad_fn

from transformers import BertModel
from pathlib import Path

from konlpy.tag import *

import torch
import pickle
import json
import pandas as pd


class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != "":
                    list_of_ner_word.append(entity_word.replace("▁", " ").lstrip())

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append(entity_word.replace("▁", " ").lstrip())
                entity_word, entity_tag, prev_entity_tag = "", "", ""


        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for token_str, pred_ner_tag_str in zip(input_token, pred_ner_tag):
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:] # 첫번째 예측을 기준으로 하겠음
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence

# original NER model

class ner():
    def __init__(self):
        model_dir = '/opt/ml/level3_productserving-level3-nlp-01/app/keyword_extraction/ner_config'

        model_dir = Path(model_dir)
        model_config = Config(json_path=model_dir / 'config.json')

        tok_path = "/opt/ml/project_models/ner/tokenizer_78b3253a26.model"
        ptr_tokenizer = SentencepieceTokenizer(tok_path)

        import sys # for path error handling
        sys.path.append('/opt/ml/level3_productserving-level3-nlp-01/app/keyword_extraction') # data_utils가 위치하는 path 추가

        with open("app/keyword_extraction/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)

        self.tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

        with open(model_dir / "ner_to_index.json", 'rb') as f:
            ner_to_index = json.load(f)
            index_to_ner = {v: k for k, v in ner_to_index.items()}

        self.model = KobertCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)
        model_dict = self.model.state_dict()
        checkpoint = torch.load("/opt/ml/project_models/ner/ner_model_epoch_12.bin", map_location="cuda")
        convert_keys = {}
        for k, v in checkpoint['model_state_dict'].items():
            new_key_name = k.replace("module.", '')
            if new_key_name not in model_dict:
                print("{} is not int model_dict".format(new_key_name))
                continue
            convert_keys[new_key_name] = v
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model.load_state_dict(convert_keys, strict=False)
        self.model.to(device)
        self.model.eval()

        self.decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=self.tokenizer, index_to_ner=index_to_ner)
    
    def extraction(self, doc):
        input_text = doc
        list_of_input_ids = self.tokenizer.list_of_string_to_list_of_cls_sep_token_ids([input_text])
        x_input = torch.tensor(list_of_input_ids).long().to(torch.device('cuda'))

        list_of_pred_ids = self.model(x_input)

        list_of_ner_word, decoding_ner_sentence = self.decoder_from_res(list_of_input_ids=list_of_input_ids, list_of_pred_ids=list_of_pred_ids)

        return(list_of_ner_word)


def get_nouns(old_input):
    new_input = []

    for i in old_input:
        tagged = Mecab().pos(i)
        flag = 1
        for _, t in tagged:
            if(t=='JKS' or t=='JKC' or t=='JKG' or t=='JKO' or t=='JKB' or t=='JKV' or t=='JKG' or t=='JC' or t=='JX'):# and len(s)>1:
                flag = 0
                break
        if(flag):
            new_input.append(i)

    return new_input

def get_nouns_sentence(text):
    tokenized_doc = Mecab().pos(text)
    nouns_sentence = ' '.join([word[0] for word in tokenized_doc if (word[1] == 'NNG' or word[1] == 'NNP')])
    return nouns_sentence

def main_extraction(ner_model, kw_model, docs):
    '''
    ## main_extraction
    description:
    - answer word를 추출합니다.
    
    args:
    - ner_model : ner 진행 시 사용하는 모델
        - KobertCRF
    - kw_model : keyword extraction 진행 시 사용하는 모델
        - KeyBERT
    - docs : stt 이후 생성된 문단 리스트
        - list
    '''
    stopwords = []
    with open('app/keyword_extraction/stopwords.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            stopwords.append(line.replace('\n',''))

    #ner model
    ner_model = ner_model

    #keybert model
    kw_model = kw_model

    list_of_key_word = []

    for doc in docs: #doc = 하나의 context
        keywords_check = [] #중복 키워드 방지

        sen_list = []
        for i in doc.split('.'):
            if '?' in i:
                for j in i.split('?'):
                    if(len(j)>1):
                        sen_list.append(j)
            else:
                if(len(i)>1):
                    sen_list.append(i+'.')

        keywords = []
        for index, sen in enumerate(sen_list):
            # ner output
            temp_output = ner_model.extraction(sen)
            # 조사 제거
            temp_output = get_nouns(temp_output)

            output = []
            # 한글자 제거 
            for i, j in enumerate(temp_output):
                if len(j)>1:
                    output.append(temp_output[i])
            
            #keybert output
            keyword = kw_model.extract_keywords(get_nouns_sentence(sen), keyphrase_ngram_range=(1,1), top_n=1)

            if(keyword):
                output.append(keyword[0][0])

            # 중복 제거
            temp = set(output)
            output = list(temp)

            # 불용어 제거  & 하나의 context에서 키워드 중복되지 않도록 진행
            final_output = []
            for token in output: 
                if (token not in stopwords) and (token not in keywords_check): 
                    final_output.append(token)
                    keywords_check.append(token)


            for word in final_output:
                keywords.append((index, word))
            
        list_of_key_word.append({"context" : doc, "keyword": keywords})

    return list_of_key_word # dict list