import pandas as pd


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self):
        data = pd.read_csv(self.data_path, index_col = 0)
        data['result'] = data['result'].astype(str)
        data = ' '.join(data['result']) # 하나의 덩어리로 뭉쳐서 반환
        return data

if __name__ == '__main__':
    data_path = '/opt/ml/Segmentation/data/dataset_0118.csv'
    loader = DataLoader(data_path=data_path)
    loader.load()