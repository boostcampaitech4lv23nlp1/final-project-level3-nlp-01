import torch
from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel


def stt_post_model_init(model_path):
    BOS = '</s>'
    EOS = '</s>'
    MASK = '<unused0>'
    PAD = '<pad>'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path,
                    bos_token=BOS, eos_token=EOS, unk_token='<unk>',
                    pad_token=PAD, mask_token=MASK,padding_side='left') 
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    return model, tokenizer