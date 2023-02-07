from transformers import(
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

def qg_model_init():
    model = T5ForConditionalGeneration.from_pretrained('/opt/ml/project_models/question_generation_t5/qg_models')
    tokenizer = T5TokenizerFast.from_pretrained('/opt/ml/project_models/question_generation_t5/qg_tokenizer')
    return model, tokenizer