import itertools
import logging
from typing import Optional, Dict, Union

from nltk import sent_tokenize

import torch
from transformers import(
    T5ForConditionalGeneration,
    T5TokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

class QGPipeline:
    """only for question generation only. excluding answer extraction."""
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ans_model: PreTrainedModel,
        ans_tokenizer: PreTrainedTokenizer,
        qg_format: str,
        use_cuda: bool
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.ans_model = ans_model
        self.ans_tokenizer = ans_tokenizer

        self.qg_format = qg_format

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        if self.ans_model is not self.model:
            self.ans_model.to(self.device)

        self.model_type = "t5"

    def __call__(self, idx: int, context: str, answers: list):
        # sents = sent_tokenize(context)
        sents = []
        for i in context.split('.'):
            if '?' in i:
                for j in i.split('?'):
                    if(len(j)>1):
                        sents.append(j)
            else:
                if(len(i)>1):
                    sents.append(i+'.')

        flat_answers = list(itertools.chain(*answers))
        
        if len(flat_answers) == 0:
          return []

        qg_examples = self._prepare_inputs_for_qg_from_answers_hl(sents, answers)
        
        qg_inputs = [example['source_text'] for example in qg_examples]
        questions = self._generate_questions(qg_inputs)
        output = [{'idx': idx, 'answer': example['answer'], 'question': que} for example, que in zip(qg_examples, questions)]
        return output
    
    def _generate_questions(self, inputs):
        inputs = self._tokenize(inputs, padding=True, truncation=True)
        
        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device), 
            max_length=32,
            num_beams=4,
        )
        
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions
    
    
    def _tokenize(self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs
    
    
    def _prepare_inputs_for_qg_from_answers_hl(self, sents, answers):
        inputs = []
        for answer in answers:
            idx, answer_text = answer[0], answer[1]
            
            sent = sents[idx]
            sents_copy = sents[:]

            answer_text = answer_text.strip()
            if answer_text not in sent:
                continue

            ans_start_idx = sent.index(answer_text)

            sent = f"{sent[:ans_start_idx]} <hl> {answer_text} <hl> {sent[ans_start_idx + len(answer_text): ]}"
            sents_copy[idx] = sent

            source_text = " ".join(sents_copy)
            source_text = f"generate question: {source_text}"
            source_text = source_text + " </s>"
            inputs.append({"answer": answer_text, "source_text": source_text})
        
        return inputs
    
    
class E2EQGPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        use_cuda: bool
    ) :

        self.model = model
        self.tokenizer = tokenizer

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration", "BartForConditionalGeneration"]
        
        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"
        else:
            self.model_type = "bart"
        
        self.default_generate_kwargs = {
            "max_length": 512,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }
    
    def __call__(self, context: str, **generate_kwargs):
        inputs = self._prepare_inputs_for_e2e_qg(context)

        # TODO: when overrding default_generate_kwargs all other arguments need to be passsed
        # find a better way to do this
        if not generate_kwargs:
            generate_kwargs = self.default_generate_kwargs
        
        input_length = inputs["input_ids"].shape[-1]
        
        # max_length = generate_kwargs.get("max_length", 256)
        # if input_length < max_length:
        #     logger.warning(
        #         "Your max_length is set to {}, but you input_length is only {}. You might consider decreasing max_length manually, e.g. summarizer('...', max_length=50)".format(
        #             max_length, input_length
        #         )
        #     )

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device), 
            attention_mask=inputs['attention_mask'].to(self.device),
            **generate_kwargs
        )

        prediction = self.tokenizer.decode(outs[0], skip_special_tokens=True)
        questions = prediction.split("<sep>")
        questions = [question.strip() for question in questions[:-1]]
        return questions
    
    def _prepare_inputs_for_e2e_qg(self, context):
        source_text = f"generate questions: {context}"
        if self.model_type == "t5":
            source_text = source_text + " </s>"
        
        inputs = self._tokenize([source_text], padding=False)
        return inputs
    
    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=512
    ):
        inputs = self.tokenizer.batch_encode_plus(
            inputs, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs


SUPPORTED_TASKS = {
    "question-generation": {
        "impl": QGPipeline,
        "default": {
            "model": "/opt/ml/input/data/question_generation/t5_models/checkpoint-4000",
            "ans_model": "/opt/ml/input/data/question_generation/t5_models/checkpoint-4000",
        }
    },
    "e2e-qg": {
        "impl": E2EQGPipeline,
        "default": {
            "model": "/opt/ml/input/data/question_generation/t5_models/checkpoint-4000",
        }
    }
}

def pipeline(
    task: str,
    model: Optional[Union[str, PreTrainedModel]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    qg_format: Optional[str] = "highlight",
    ans_model: Optional[Union[str, PreTrainedModel]] = None,
    ans_tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    use_cuda: Optional[bool] = True,
    **kwargs,
):
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError("Unknown task {}, available tasks are {}".format(task, list(SUPPORTED_TASKS.keys())))

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        model = targeted_task["default"]["model"]
    
    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
            )
    
    # Instantiate tokenizer if needed
    if isinstance(tokenizer, (str, tuple)):
        if isinstance(tokenizer, tuple):
            # For tuple we have (tokenizer name, {kwargs})
            tokenizer = T5TokenizerFast.from_pretrained(tokenizer[0], **tokenizer[1])
        else:
            tokenizer = T5TokenizerFast.from_pretrained(tokenizer)
    
    # Instantiate model if needed
    if isinstance(model, str):
        model = T5ForConditionalGeneration.from_pretrained(model)
    
    if task == "question-generation":
        if ans_model is None:
            # load default ans model
            ans_model = targeted_task["default"]["ans_model"]
            ans_tokenizer = T5TokenizerFast.from_pretrained(ans_model)
            ans_model = T5ForConditionalGeneration.from_pretrained(ans_model)
        else:
            # Try to infer tokenizer from model or config name (if provided as str)
            if ans_tokenizer is None:
                if isinstance(ans_model, str):
                    ans_tokenizer = ans_model
                else:
                    # Impossible to guest what is the right tokenizer here
                    raise Exception(
                        "Impossible to guess which tokenizer to use. "
                        "Please provided a PretrainedTokenizer class or a path/identifier to a pretrained tokenizer."
                    )
            
            # Instantiate tokenizer if needed
            if isinstance(ans_tokenizer, (str, tuple)):
                if isinstance(ans_tokenizer, tuple):
                    # For tuple we have (tokenizer name, {kwargs})
                    ans_tokenizer = T5TokenizerFast.from_pretrained(ans_tokenizer[0], **ans_tokenizer[1])
                else:
                    ans_tokenizer = T5TokenizerFast.from_pretrained(ans_tokenizer)

            if isinstance(ans_model, str):
                ans_model = T5ForConditionalGeneration.from_pretrained(ans_model)
    
    if task == "e2e-qg":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
    else: # task == "question-generation":
        return task_class(model=model, tokenizer=tokenizer, ans_model=ans_model, ans_tokenizer=ans_tokenizer, qg_format=qg_format, use_cuda=use_cuda)
