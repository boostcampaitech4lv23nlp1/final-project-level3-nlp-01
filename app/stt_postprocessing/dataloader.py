import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self):
        data = pd.read_csv(self.data_path)
        return data

if __name__ == '__main__':
    data_path = '/opt/ml/espnet-asr/STT_postprocessing/history_dataset.csv'
    loader = DataLoader(data_path=data_path)
    loader.load()