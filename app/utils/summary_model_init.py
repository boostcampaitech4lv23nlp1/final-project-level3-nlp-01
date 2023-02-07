from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def model_init_for_t5(model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, stride=128)

    return model, tokenizer

def model_init_for_bart(model_path):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

    return model, tokenizer

def summary_model_init(model_path, model_name):
    if model_name == 't5':
        model, tokenizer = model_init_for_t5(model_path)

    if model_name == 'kobart':
        model, tokenizer = model_init_for_bart(model_path)
    return model, tokenizer, model_name