import pandas as pd
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
import torch
import numpy as np
import evaluate
from peft import LoKrConfig, get_peft_model, TaskType

train_data_path = 'dataset/ACP_deleted.csv'
model_name = 'model/esm2_35M'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_part = model_name.split('/')[-1]

# dataset
dataset = load_dataset('csv',data_files=train_data_path,split='train')
datasets = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

def process_fucntion(examples):
    tokeinzed_examples = tokenizer(examples['sequence'], max_length=128, truncation=True)
    tokeinzed_examples['labels'] = examples['label']
    return tokeinzed_examples

tokenized_datasets = datasets.map(process_fucntion, batched=True, remove_columns=datasets['train'].column_names)

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']


# evaluate

acc_metric = evaluate.load('metrics/accuracy/accuracy.py')
f1_metric = evaluate.load('metrics/f1/f1.py')
#sn_metric = evaluate.load('')
#sp_metric = evaluate.load('')
#mcc_metric = evaluate.load('')
pre_metric = evaluate.load('metrics/precision/precision.py')
#roc_auc_metric = evaluate.load('metrics/roc_auc/roc_auc.py')

def eval_predict(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    pre = pre_metric.compute(predictions=predictions, references=labels)
    #roc_auc = roc_auc_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    acc.update(pre)
    return acc

# TrainingArguments
train_Args = TrainingArguments(output_dir=f'./checkpionts/{model_part}',
    num_train_epochs=1,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=5e-5,
    weight_decay=0.0001,
    warmup_ratio=0.5,
    gradient_accumulation_steps=2,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=5,
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    greater_is_better=True
    )

# Trainer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

config = LoKrConfig(task_type=TaskType.SEQ_CLS, target_modules=['query'])
model_LoKr = get_peft_model(model, config)

trainer = Trainer(model=model_LoKr,
                  args=train_Args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_predict)
trainer.train()
mtrics = trainer.evaluate()
print(mtrics)
print(model_LoKr.print_trainable_parameters())