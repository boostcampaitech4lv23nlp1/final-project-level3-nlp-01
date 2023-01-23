import pandas as pd


class DataLoader:
    def __init__(self, data):
        self.data = data

    def load(self):
        data = self.data
        data = ' '.join(data)
        return data