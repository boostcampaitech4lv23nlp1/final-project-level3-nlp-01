from transformers import(
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

def qg_model_init():
    model = T5ForConditionalGeneration.from_pretrained(ans_model)
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer)
    return model, tokenizer