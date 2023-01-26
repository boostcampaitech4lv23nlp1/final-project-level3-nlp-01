import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")


OUTPUT_TKN = "<usr>"
RESULT_TKN = "<sys>" 
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

class STTDataset(Dataset):
    def __init__(self, datasets, tokenizer, max_len=64):  # 데이터셋의 전처리를 해주는 부분
        self._data = datasets
        self.max_len = max_len
        self.output_token = OUTPUT_TKN
        self.result_token = RESULT_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = tokenizer

    def __len__(self):  # STTdata 의 길이를 리턴한다.
        return len(self._data)

    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        df = self._data.iloc[idx]
        query = df["output"]  # 질문을 가져온다.

        answer = df["label"]  # 답변을 가져온다.

        q_tokenized = self.tokenizer.tokenize(self.output_token + str(query) + self.sent_token)
        q_len = len(q_tokenized)

        a_tokenized = self.tokenizer.tokenize(self.result_token + str(answer) + self.eos)
        a_len = len(a_tokenized)

        #질문의 길이가 최대길이보다 크면
        if q_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_tokenized = q_tokenized[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_tokenized)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_tokenized = a_tokenized[:a_len]
            a_len = len(a_tokenized)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_tokenized = q_tokenized[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_tokenized)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_tokenized = a_tokenized[:a_len]
            a_len = len(a_tokenized)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask,] * q_len + a_tokenized[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.
        token_ids = self.tokenizer.convert_tokens_to_ids(q_tokenized + a_tokenized)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)
    
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)