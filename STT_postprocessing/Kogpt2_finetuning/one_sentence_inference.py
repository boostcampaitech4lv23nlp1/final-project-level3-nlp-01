import numpy as np
import pandas as pd
import torch
import time
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
import warnings
warnings.filterwarnings("ignore")


def inference(model_path,text):

    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(model_path,
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK) 
    model = GPT2LMHeadModel.from_pretrained(model_path)

    sent = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    a=""

    with torch.no_grad():
        if text!="":
            while 1:
                print(Q_TKN + text + SENT + sent + A_TKN + a)
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + text + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
                print(input_ids)
                input_ids = input_ids.to(device)
                pred = model(input_ids)
                pred = pred.logits
                print(pred.shape)
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
    
    return a

if __name__ == '__main__' :
    text = input('텍스트를 입력해 주세요.')
    answer = inference(model_path = '/opt/ml/espnet-asr/final/GPT_2', text = text)
    print(f'query : {text}')
    print(f'answer : {answer}')