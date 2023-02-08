import numpy as np
import pandas as pd
import torch
import time
from .dataloader import Datasets
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
import warnings
warnings.filterwarnings("ignore")


def postprocess(model, tokenizer, df):
    
    inference_dataset = Datasets(df, tokenizer)
    tokenized_dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size=32)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sentences= []
    
    for batch in tokenized_dataloader:
        output = model.generate(input_ids = batch['input_ids'].to(device),
                            attention_mask =batch['attention_mask'].to(device),
                            max_length = 64,)
        
        outputs = []
        for out in output:
            x = torch.where(out == 4)[0].tolist()[0]
            out = out[x:]
            outputs.append(out)

        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for answer in decoded_output:
            if answer[0] == '아' and answer[1] == ' ':
                    answer = answer[2:]
            elif answer[0] == '어' and answer[1] == ' ':
                answer = answer[2:]
            
            if '<pad>' in answer :
                answer = answer.replace('<pad>', '')
                sentences.append(answer)
                print(f"answer : {answer.strip()}")
            elif '<unk>' in answer : 
                answer = answer.replace('<unk>','')
                print(f"answer : {answer.strip()}")
                sentences.append(answer.strip())
            else:
                sentences.append(answer.strip())
                print(f"answer : {answer.strip()}")
    # df = pd.DataFrame({'result' : sentences})
    # df.to_csv('inference.csv',index=False)
    return sentences

if __name__ == '__main__':

    model_path = '/opt/ml/espnet-asr/final/GPT_2'
    OUTPUT_TKN = "<usr>"
    RESULT_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'
    start_time = time.time()
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path,
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK,padding_side='left')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    data_path = '설민석_일제강점기_inference_small_beam10.csv'
    df = pd.read_csv(data_path)

    result = postprocess(model = model, tokenizer=tokenizer, df = df)

    print(f'sec : {time.time() - start_time}')