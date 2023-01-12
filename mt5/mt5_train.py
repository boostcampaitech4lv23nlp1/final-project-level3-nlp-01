import argparse
import logging
import pandas as pd
import os
import transformers
from datasets import load_dataset
import nltk
nltk.download('punkt')
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5TokenizerFast, MT5ForConditionalGeneration
from datasets import load_metric
import sys
os.environ["WANDB_DISABLED"] = "true"
# logger = logging.getLogger(__name__)

class QuestionDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=30):
        self.data_dir = data_dir
        self.datasets = load_dataset(self.data_dir)
        self.type_path = type_path
        self.data = pd.DataFrame(self.datasets[self.type_path])

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_text,output_text= self.data.loc[idx, 'context'],self.data.loc[idx, 'question']

            input_ = "Korea Context: %s" % (input_text)
            target = "%s " %(output_text)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=512, padding=True, truncation=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=128, padding=True, truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


model_path = './model'
model_name = 'mt5-kor-qg'
model_dir = f"{model_path}/{model_name}"
model_checkpoint = 'google/mt5-base'
# max_input_length = 512
# max_target_length = 128


# Load model
model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding="max_length")

def get_dataset(tokenizer, type_path, args):
    return QuestionDataset(tokenizer=tokenizer, data_dir="squad_kor_v1", type_path=type_path,  max_len=args)

# Load datset
# datasets = load_dataset("squad_kor_v1")
train_dataset = get_dataset(tokenizer=tokenizer, type_path="train", args=512)
train_dataloader = DataLoader(train_dataset, batch_size=8,  shuffle=True, num_workers=0)

val_dataset = get_dataset(tokenizer=tokenizer, type_path="validation", args=512)
val_dataloader =  DataLoader(val_dataset, batch_size=8, num_workers=0)

# logger.info('finished loading dataset')


training_args = Seq2SeqTrainingArguments(
    model_dir,
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
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Training
print('Start Training...')
trainer.train()

# Saving model
print('Saving Model...')
trainer.save_model()
