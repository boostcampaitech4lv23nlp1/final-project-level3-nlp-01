import pandas as pd
import torch
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
from dataloader import STTDataset
from dataloader import collate_batch


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
            out = out.logits  
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, args.Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)
  
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
            out = out.logits 
            mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
            mask_out = torch.where(mask_3d == 1, out, args.Sneg * torch.ones_like(out))
            loss = criterion(mask_out.transpose(2, 1), label)

            avg_loss = loss.sum() / mask.sum()
            global_loss+=avg_loss

        print(f'validation avg_loss : {global_loss/len(test_dataloader)}')
        if args.wandb=='True':
            wandb.log({"validation_loss": global_loss/len(test_dataloader)})
        
    return global_loss/len(test_dataloader)
                
                
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=5e-5, help='')
    parser.add_argument('--batch', type=int, default=64, help='')
    parser.add_argument('--n_epochs', type=int, default=5, help='')
    parser.add_argument('--dataset_path', type=str, default='/opt/ml/espnet-asr/merge.csv', help='')
    parser.add_argument('--model_name', type=str, default='skt/kogpt2-base-v2', help='')
    parser.add_argument('--report_name', type=str, default='', help='')
    parser.add_argument('--Sneg', type=float, default=-1e18, help='')
    parser.add_argument('--max_len', type=int, default=64, help='')
    parser.add_argument('--save_dir', type=str, default="GPT_2", help='')
    parser.add_argument('--wandb', type=str, default="True", help='')

    args = parser.parse_args()
    
    try : 
        wandb.login(key = '3e00a171508ab88512c57afafb441f5ee2b4864b')
    except:
        annoy = 'must'
    
    if args.wandb=='True':
        wandb.init(project="STT preprocessing", entity="jjjjjun")

    print(args)
    
    OUTPUT_TKN = "<usr>"
    RESULT_TKN = "<sys>" 
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    SENT = '<unused1>'
    PAD = '<pad>'

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name,
               bos_token=BOS, eos_token=EOS, unk_token='<unk>',
               pad_token=PAD, mask_token=MASK)
    
    model = GPT2LMHeadModel.from_pretrained(args.model_name)

    dataset = pd.read_csv(args.dataset_path)
    trainset, testset = train_test_split(dataset, test_size=0.1, shuffle=True, random_state=42)
    
    train_set = STTDataset(trainset, tokenizer, args.max_len)
    test_set = STTDataset(testset, tokenizer, args.max_len)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch, num_workers=2, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_set, batch_size=args.batch, num_workers=2, shuffle=False, collate_fn=collate_batch)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=4, verbose=True)
    
    Path('/opt/ml/espnet-asr/final/'+args.save_dir).mkdir(parents=True, exist_ok=True)
 
    train(args, tokenizer, model, train_dataloader, test_dataloader, args.n_epochs, args.lr, criterion, optimizer, scheduler, device, args.save_dir)
