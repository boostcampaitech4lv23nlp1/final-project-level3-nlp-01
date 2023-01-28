import pandas as pd
import torch
import time
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
import warnings
from multiprocessing import Process
from .inference import inference
from .split_data import split_file
warnings.filterwarnings("ignore")


def postprocess(model, tokenizer, df):

    '''
    ## postprocess
    description:
    - stt 작업 후 후처리를 진행합니다.
    
    args:
    - model: 후처리에 사용하는 모델
        - GPT2LMHeadModel
    - tokenizer: 후처리에 사용하는 tokenizer 모델
        - PreTrainedTokenizerFast
    - df: 후처리 입력으로 활용되는 stt output
        - pd.DataFrame
    '''

    num_process = 1
    
    print(f'num process : {num_process}')
    
    scps = split_file(df, split = num_process)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = tokenizer
    model = model.to(device)
    max_len = 64
    processes = []
    sentences = []
    # outputs = [] # for making csv file
    if len(scps) > 1:
        model.share_memory()

        for i, scp in enumerate(scps):
            output_dir= f'./output/inference_{i}'
            args = [model, tokenizer, scp, max_len, output_dir]
 
            process = Process(target=inference,args=args)
            process.start()
            processes.append(process)
            time.sleep(0.1)

        for process in processes:
            process.join()
        processes.clear()
    else:
        output_dir = './output/inference_0'
        num_process = 1
        args = [model, tokenizer, scps[0], max_len, output_dir]
        inference(*args)
    
    for i in range(num_process):
        df = pd.read_csv(f'./output/inference_{i}.csv')
        for _, item in df.iterrows():
            sentences.append(item['result'])
    #         outputs.append(item['output']) # for making csv file
    # data = pd.DataFrame({
    #     'output' : outputs,
    #     'result' : sentences
    # })
    # data.to_csv('./inference.csv', index=False)
    
    return sentences


if __name__ == '__main__':
    from dataloader import DataLoader
    start_time = time.time()
    data_path = '/opt/ml/espnet-asr/STT_postprocessing/history_dataset.csv'
    loader = DataLoader(data_path=data_path)
    df = loader.load()
    torch.multiprocessing.set_start_method('spawn',force=True)
    results = postprocess(model_path = '/opt/ml/espnet-asr/final/GPT_2', df = df)
    print(results)
    print(f'sec : {time.time()-start_time}')