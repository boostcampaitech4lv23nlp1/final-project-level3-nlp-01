import torch
import whisper
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer

from transformers import WhisperForConditionalGeneration
from omegaconf import OmegaConf

def stt_model_init():
    cfg = OmegaConf.load('app/STT/data/conf.yaml')
    kwargs = dict(getattr(cfg, 'whisper'))
    model_size = kwargs.pop('model_size')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WhisperForConditionalGeneration.from_pretrained(
            f'openai/whisper-{model_size}',
        ).to(device)

    return model

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

def segment_model_init():
    model = SentenceTransformer('/opt/ml/project_models/stt/segments_sentencetransformer')
    return model
