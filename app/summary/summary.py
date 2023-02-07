from tqdm import tqdm
import nltk
# nltk.download('punkt')
import torch


class Summarizer:
    def __init__(self, data, model, tokenizer, max_input_length: int, max_target_length: int, model_: str):
        self.data = data
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.model_ = model_

    def summarize(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        result_list = []
        if self.model_ == 't5':
            model, tokenizer = self.model, self.tokenizer
            model.to(device)
            model.config.max_length = self.max_target_length
            tokenizer.model_max_length = self.max_target_length
            
            for paragraph in tqdm(self.data):
                inputs = ['summarize: ' + paragraph]
                inputs = tokenizer(inputs, max_length=self.max_input_length, truncation=True, return_tensors="pt")
                output = model.generate(**inputs, num_beams=3, do_sample=True, min_length=10, max_length=self.max_target_length)
                decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                result = nltk.sent_tokenize(decoded_output.strip())[0]
                result_list.append(result)
        
        else:
            model, tokenizer = self.model, self.tokenizer
            model.to(device)
            
            for paragraph in tqdm(self.data):
                raw_input_ids = tokenizer.encode(paragraph)
                input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
                summary_ids = model.generate(torch.tensor([input_ids]).to("cuda"),  num_beams=5,  max_length=self.max_target_length,  eos_token_id=1)
                result = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
                result_list.append(result)

        return result_list



