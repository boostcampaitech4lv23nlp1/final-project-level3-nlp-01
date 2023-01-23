from tqdm import tqdm
import nltk
nltk.download('punkt')
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def model_init_for_t5(model_path, max_target_length):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, stride=128)
    
    model.config.max_length = max_target_length
    tokenizer.model_max_length = max_target_length
    return model, tokenizer

def model_init_for_bart(model_path):
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

    return model, tokenizer

class Summarizer:
    def __init__(self, data, model_path, max_input_length, max_target_length, model_):
        self.data = data
        self.model_path = model_path
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_ = model_

    def summarize(self):
        result_list = []
        if self.model_ == 't5':
            model, tokenizer = model_init_for_t5(self.model_path, self.max_target_length)
            
            for paragraph in tqdm(self.data):
                inputs = ['summarize: ' + paragraph]
                inputs = tokenizer(inputs, max_length=self.max_input_length, truncation=True, return_tensors="pt")
                output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=self.max_target_length)
                decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                result = nltk.sent_tokenize(decoded_output.strip())[0]
                result_list.append(result)
        
        else:
            model, tokenizer = model_init_for_bart(self.model_path)

            for paragraph in tqdm(self.data):
                raw_input_ids = tokenizer.encode(paragraph)
                input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
                summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=5,  max_length=self.max_target_length,  eos_token_id=1)
                result = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
                result_list.append(result)

        return result_list



