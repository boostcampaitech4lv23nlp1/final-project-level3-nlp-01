from transformers import Trainer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5TokenizerFast, MT5ForConditionalGeneration
from datasets import load_dataset, load_metric
from tqdm import tqdm
import numpy as np
import string

import nltk

import sys

if __name__ == '__main__':

    # print('torch', torch.__version__) 1.12.1+cu113
    # print('transformers', transformers.__version__) 4.25.1
    # print('datasets', datasets.__version__) 2.8.0
    # print('pytorch_lightning', pytorch_lightning.__version__) 1.8.6

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
    print('FINISH MODEL & TOKENIZER LOADING')

    data = load_dataset('squad_kor_v1')
    print('FINISH DATASET LOADING')

    
    def clean_text(text):
        sentences = nltk.sent_tokenize(text.strip())
        sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
        sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                        if len(sent) > 0 and
                                        sent[-1] in string.punctuation]
        text_cleaned = "\n".join(sentences_cleaned_no_titles)
        return text_cleaned

    prefix = 'Korean Context: '
    max_input_length = 512
    max_target_length = 128


    def preprocess_data(examples):
        texts_cleaned = [clean_text(text) for text in examples["context"]]
        #print(texts_cleaned)
        inputs = [prefix + text for text in texts_cleaned]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["question"], max_length=max_target_length, 
                            truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    preprocessed_data = data.map(preprocess_data, batched=True)
    print('FINISH DATSET PREPROCESSING')

    training_args = Seq2SeqTrainingArguments(
        './test',
        evaluation_strategy="steps", eval_steps=30000,
        logging_strategy="steps", logging_steps=30000,
        save_strategy="steps", save_steps=30000,
        learning_rate=3e-5,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=3,
        predict_with_generate=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                        for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                        for label in decoded_labels]
        
        # Compute ROUGE scores
        result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                                use_stemmer=True)

        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length to metrics
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                        for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return {k: round(v, 4) for k, v in result.items()}


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_data["train"],
        eval_dataset=preprocessed_data["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print('finish trainer loading ')

    # Training
    print('Start Training...')
    trainer.train()
