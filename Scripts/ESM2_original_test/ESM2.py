import pandas as pd
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
import torch
import numpy as np
import evaluate
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

train_data_path = r'C:\Users\KQDtianxiaK\PycharmProjects\PEFT-PLM\datasets\Protein\ACP\Anticancer-Peptides-CNN\acp740_all.csv'
model_name = 'model/esm2_35M'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_part = model_name.split('/')[-1]

# dataset
dataset = load_dataset('csv',data_files=train_data_path,split='train')
datasets = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

def process_fucntion(examples):
    tokeinzed_examples = tokenizer(examples['sequence'], max_length=128, padding=True,truncation=True)
    tokeinzed_examples['labels'] = examples['label']
    return tokeinzed_examples

tokenized_datasets = datasets.map(process_fucntion, batched=True, remove_columns=datasets['train'].column_names)

train_dataset = tokenized_datasets['train']
test_dataset = tokenized_datasets['test']


# evaluate

acc_metric = evaluate.load('metrics/accuracy/accuracy.py')
f1_metric = evaluate.load('metrics/f1/f1.py')
recall_metric = evaluate.load('metrics/recall/recall.py')
pre_metric = evaluate.load('metrics/precision/precision.py')
roc_auc_metric = evaluate.load('metrics/roc_auc/roc_auc.py')

def eval_predict(eval_predict):
    logits, labels = eval_predict
    predictions = logits.argmax(axis=-1)
    pred_proba = np.argmax(logits, axis=-1)

    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    pre = pre_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    roc_auc = roc_auc_metric.compute(prediction_scores=pred_proba, references=labels)

    acc.update(f1)
    acc.update(pre)
    acc.update(recall)
    acc.update(roc_auc)
    return acc

# TrainingArguments
train_Args = TrainingArguments(output_dir=f'./checkpoints/{model_part}',
    num_train_epochs=10,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    learning_rate=2e-5,
    weight_decay=0.0001,
    warmup_ratio=0.5,
    gradient_accumulation_steps=2,
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=0,
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    greater_is_better=True
    )

# Trainer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print(sum(param.numel() for param in model.parameters()))
trainer = Trainer(model=model,
                  args=train_Args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_predict)
trainer.train()
mtrics = trainer.evaluate()
print(mtrics)

# 评估模型并收集预测概率和标签
test_predictions, test_pred_proba, test_labels = trainer.predict(test_dataset).predictions

# 将预测概率和标签转换为numpy数组以进行ROC曲线绘制
test_pred_proba = test_pred_proba.numpy()
test_labels = test_labels.numpy()

# 计算ROC AUC分数
roc_auc = roc_auc_score(test_labels, test_pred_proba)
print(f"ROC AUC: {roc_auc}")

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(test_labels, test_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()