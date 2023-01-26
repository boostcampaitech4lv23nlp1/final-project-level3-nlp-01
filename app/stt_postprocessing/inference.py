import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")


def inference(model, tokenizer, data_path , max_len, output_dir):

    OUTPUT_TKN = "<usr>"
    RESULT_TKN = "<sys>"
    EOS = '</s>'
    SENT = '<unused1>'

    sent = '0'
    sentences = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():

        df = pd.read_csv(data_path)
        for idx, item in df.iterrows():
            answer=""
            query = item['output']
            print(f'query : {query}')

            if query is not np.nan:
                while True:
                    input_ids = torch.LongTensor(tokenizer.encode(OUTPUT_TKN + query + SENT + sent + RESULT_TKN + answer)).unsqueeze(dim=0)
                    input_ids = input_ids.to(device)
                    pred = model(input_ids)
                    pred = pred.logits
                    if pred.shape[1]>max_len:
                        print("error!!")
                        break
                    gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    answer += gen.replace("▁", " ")
                answer = answer.strip()
                if answer[0] == '아' and answer[1] == ' ':
                    answer = answer[2:]
                elif answer[0] == '어' and answer[1] == ' ':
                    answer = answer[2:]

                if '<pad>' in answer :
                    sentences.append(query)
                    print(f"answer : {query.strip()}")
                elif '<unk>' in answer : 
                    answer = answer.replace('<unk>','')
                    print(f"answer : {answer.strip()}")
                    sentences.append(answer.strip())
                else:
                    sentences.append(answer.strip())
                    print(f"answer : {answer.strip()}")

            else:
                sentences.append(query)
                print(f"answer : {query.strip()}")

    df = pd.DataFrame({
        'output' : df['output'],
        'result' : sentences,
    })
    df.to_csv(f'{output_dir}.csv', index=False)

    return sentences