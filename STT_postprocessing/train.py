import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time



SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



class STTDataset(Dataset):
    def __init__(self, datasets, max_len=64):  # 데이터셋의 전처리를 해주는 부분
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
                q_tokenized = q_tokenized[:(int(self.max_len / 2))]   #질문길이를 최대길이의 반으로 
                q_len = len(q_tokenized)
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_tokenized = a_tokenized[:a_len]
            a_len = len(a_tokenized)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_tokenized = q_tokenized[:(int(self.max_len / 2))]   #질문길이를 최대길이의 반으로 
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

def train(args, tokenizer, model, train_dataloader, test_dataloader, epoch, learning_rate, criterion, optimizer, scheduler, device, save_dir):
    
    model.to(device)
    global_val_loss = int(1e9)
    
    for epoch in range(epoch):
        model.train()
        print(f'EPOCH : {epoch}')
        batch_idx=1
        loss_sum = 0
   
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            token_ids, mask, label = batch
            token_ids = token_ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            out = model(token_ids)
            out = out.logits      #Returns a new tensor with the logit of the elements of input
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, args.Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
            avg_loss = loss.sum() / mask.sum()
            
            loss_sum+=avg_loss
            avg_loss.backward()
            optimizer.step()


            if batch_idx%1600 == 0:
                val_loss=validation(args, model, test_dataloader, criterion, device)
                scheduler.step(val_loss)

                if global_val_loss>val_loss:
                    global_val_loss=val_loss
                    print("[model save]")
                    tokenizer.save_pretrained('/opt/ml/espnet-asr/final/'+save_dir)
                    model.save_pretrained('/opt/ml/espnet-asr/final/'+save_dir)
                    
            print(f'idx : {batch_idx}, train avg_loss : {loss_sum/batch_idx}')

            if args.wandb=='True':
                wandb.log({"train_loss": loss_sum/batch_idx})
                
            batch_idx+=1
            
def validation(args, model, test_dataloader, criterion, device):
    
    model.to(device)
    global_loss = 0
    model.eval()
    
    print(f'Start validation..')
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            token_ids, mask, label = batch
            token_ids = token_ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            out = model(token_ids)
            out = out.logits      #Returns a new tensor with the logit of the elements of input
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, args.Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
            # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
            avg_loss = loss.sum() / mask.sum()
            global_loss+=avg_loss

        print(f'validation avg_loss : {global_loss/len(test_dataloader)}')
        if args.wandb=='True':
            wandb.log({"validation_loss": global_loss/len(test_dataloader)})
        
    return global_loss/len(test_dataloader)
                
                
def inference(model_path, query):

    OUTPUT_TKN = "<usr>"
    RESULT_TKN = "<sys>"
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path,
                bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                pad_token=PAD, mask_token=MASK) 
    model = GPT2LMHeadModel.from_pretrained(model_path)

    sent = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    answer=""

    with torch.no_grad():
        if query!="":
            while 1:
                print(OUTPUT_TKN + query + SENT + sent + RESULT_TKN + answer)
                input_ids = torch.LongTensor(tokenizer.encode(OUTPUT_TKN + query + SENT + sent + RESULT_TKN + answer)).unsqueeze(dim=0)
                print(input_ids)
                input_ids = input_ids.to(device)
                pred = model(input_ids)
                pred = pred.logits
                print(pred.shape)
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break
                answer += gen.replace("▁", " ")
    
    return answer


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--lr', type=float, default=5e-5, help='')
    parser.add_argument('--batch', type=int, default=64, help='')
    parser.add_argument('--n_epochs', type=int, default=5, help='')
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/espnet-asr/merge3.csv', help='')
    parser.add_argument('--model_name', type=str, default='skt/kogpt2-base-v2', help='')
    parser.add_argument('--report_name', type=str, default='', help='')
    parser.add_argument('--Sneg', type=float, default=-1e18, help='')
    parser.add_argument('--save_dir', type=str, default="GPT_2", help='')
    parser.add_argument('--wandb', type=str, default="False", help='')
    parser.add_argument('--max_len', type=int, default=64, help='')
    parser.add_argument('--inference', type=str, default="True", help='')

    args = parser.parse_args()
    
    # try : 
    #     wandb.login(key = '3e00a171508ab88512c57afafb441f5ee2b4864b')
    # except:
    #     annoy = 'must'
    
    # if args.wandb=='True':
    #     wandb.init(project="STT preprocessing", entity="jjjjjun")

    print(args)
    
    OUTPUT_TKN = "<usr>"
    RESULT_TKN = "<sys>" 
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    if args.inference == 'True':
        args.model_name = '/opt/ml/espnet-asr/final/GPT_2'
        
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name,
               bos_token=BOS, eos_token=EOS, unk_token='<unk>',
               pad_token=PAD, mask_token=MASK)
    
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    dataset = pd.read_csv(args.dataset_path)
    trainset, testset = train_test_split(dataset, test_size=0.1, shuffle=True, random_state=42)
    
    train_set = STTDataset(trainset, args.max_len)
    test_set = STTDataset(testset, args.max_len)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch, num_workers=2, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_set, batch_size=args.batch, num_workers=2, shuffle=False, collate_fn=collate_batch)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
    
    Path('/opt/ml/espnet-asr/final/'+args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.inference=='False':
        train(args, tokenizer, model, train_dataloader, test_dataloader, args.n_epochs, args.lr, criterion, optimizer, scheduler, device, args.save_dir)
    
    
    if args.inference=='True':
        sent = '0'
        sentences = []
        model.to(device)
        with torch.no_grad():
            
            df = pd.read_csv('/opt/ml/espnet-asr/final_dataset.csv')
            for idx, item in df.iterrows():
                query = item['output']
                print(f'query : {query}')
                answer = ""
                
                if query is not np.nan:
                    while 1:
                        # print(OUTPUT_TKN + query + SENT + sent + RESULT_TKN + answer)
                        input_ids = torch.LongTensor(tokenizer.encode(OUTPUT_TKN + query + SENT + sent + RESULT_TKN + answer)).unsqueeze(dim=0)
                        input_ids = input_ids.to(device)
                        pred = model(input_ids)
                        pred = pred.logits
                        if pred.shape[1]>args.max_len:
                            print("error!!")
                            break
                        gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                        if gen == EOS:
                            break
                        answer += gen.replace("▁", " ")
                    print(f"answer : {answer.strip()}")
                    sentences.append(answer.strip())
                else:
                    sentences.append(query)
                    
        df = pd.DataFrame({
            'output' : df['output'],
            'result' : sentences,
        })
        df.to_csv('./inference.csv', index=False)
        print(f'sec : {time.time()-start_time}')