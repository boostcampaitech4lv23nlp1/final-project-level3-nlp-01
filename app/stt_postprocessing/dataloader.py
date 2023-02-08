import pandas as pd
from torch.utils.data import Dataset
import torch


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self):
        data = pd.read_csv(self.data_path)
        return data
    
    
class Datasets(Dataset):
    def __init__(self, df, tokenizer):
        OUTPUT_TKN = "<usr>"
        RESULT_TKN = "<sys>"
        SENT = '<unused1>'
        inputs = []
        sent = '0'
        answer=""
        for idx, item in df.iterrows():
            data = OUTPUT_TKN + item['output'] + SENT + sent + RESULT_TKN + answer
            input_ids = tokenizer(data, max_length = 32, padding = 'max_length',truncation = True)

            inputs.append(input_ids)

        self.inputs = inputs
        
    def __getitem__(self, idx) -> dict:
        
        X = {key: torch.tensor(value) for key, value in self.inputs[idx].items()}
        return X

    def __len__(self):
        return len(self.inputs)
    
    
if __name__ == '__main__':
    data_path = '/opt/ml/espnet-asr/STT_postprocessing/history_dataset.csv'
    loader = DataLoader(data_path=data_path)
    loader.load()