import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

    
MODEL_NAME = 'Sehong/kobart-QuestionGeneration'

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

sep_token = "<unused0>"

def generation(text):
    raw_input_ids = tokenizer.encode(text)
    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

    question_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
    gen_ans = tokenizer.decode(question_ids.squeeze().tolist(), skip_special_tokens=True)

    return gen_ans

def main_qg(docs):
    gen_questions = []

    for idx in range(len(docs)):
        for keyword in docs["keyword"][idx]:
            text = docs['context'][idx] + sep_token + keyword
            question = generation(text)
            gen_questions.append({'question', question, 'answer', keyword})
   

    return gen_questions
