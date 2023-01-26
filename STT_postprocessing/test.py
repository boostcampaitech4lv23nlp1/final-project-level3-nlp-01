import pandas as pd
import torch
import time
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
import warnings
from multiprocessing import Process
from inference import inference
from split_data import split_file
warnings.filterwarnings("ignore")


def main_inference(model_path, df):
    
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    PAD = '<pad>'

    num_process = 8
    
    print(f'num process : {num_process}')
    
    scps = split_file(df, split = 8)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path,
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK) 
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    max_len = 64
    processes = []
    sentences = []
    outputs = []
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
        args = [model, tokenizer, scp, max_len, output_dir]
        inference(*args)
    
    for i in range(num_process):
        df = pd.read_csv(f'./output/inference_{i}.csv')
        for idx, item in df.iterrows():
            sentences.append(item['result'])
    #         outputs.append(item['output'])
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
    results = main_inference(model_path = '/opt/ml/espnet-asr/final/GPT_2', df = df)
    print(results)
    print(f'sec : {time.time()-start_time}')
